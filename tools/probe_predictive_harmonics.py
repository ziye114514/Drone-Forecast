"""Frozen three-window comparison, including original current arc contours."""
from pathlib import Path
import sys,os,json,time,hashlib
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[variable]='1'
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from live_decode import native_batches
from predictive_harmonics import predictive_harmonics
from probe_harmonic_pixels import count_mask,summary_mask
from event_drone_forecast.geometry import fit_rotor_geometry


def current_candidates(mask,box):
    x,y,X,Y=np.rint(box).astype(int);x,y=max(0,x-30),max(0,y-30);X,Y=min(1280,X+30),min(720,Y+30)
    local=mask[y:Y,x:X]
    local=cv2.morphologyEx(local,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    local=cv2.morphologyEx(local,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(local,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours=sorted([c for c in contours if len(c)>=16 and cv2.contourArea(c)>=20],key=cv2.contourArea,reverse=True)[:6]
    candidates=[]
    for contour in contours:
        r=fit_rotor_geometry(contour,max_major=min(500.,1.5*max(X-x,Y-y)),allow_arc=False)
        if r is not None:
            candidates.append(dict(center=(np.asarray(r['center'])+[x,y]).tolist(),major=r['major'],minor=r['minor'],
                mode=r['mode'],fit_residual_px=r.get('fit_residual_px'),visible_arc_fraction=r.get('visible_arc_fraction')))
    return candidates


def process(out,name,frame,t,e,controls_hash):
    target=out/name;target.mkdir()
    np.savez_compressed(target/'native_window.npz',events=e,t=t)
    start=time.perf_counter();r=predictive_harmonics(e,t);elapsed=(time.perf_counter()-start)*1000
    count=count_mask(e,t,20000);loose=count_mask(e,t,20000)//255
    # Geometry keeps the ORIGINAL density contour. Frequency is nearby support,
    # never a new physical rim. Unsupported weak areas cannot grow arbitrarily.
    y,x=np.nonzero(r['mask']);near=cv2.dilate(r['mask'],np.ones((5,5),np.uint8))
    supported=cv2.bitwise_and(count,near)
    body=frame['body'];box=np.asarray(body['bbox'])
    record=dict(name=name,t=t,upstream_mode=frame['structure']['mode'],input_events=len(e),
        eligible_pixels=r['eligible_pixels'],accepted_pixels=r['accepted_pixels'],CPU_ms=elapsed,
        count20=summary_mask(count,box),predictive=summary_mask(r['mask'],box),
        frequency_supported_original_contour=summary_mask(supported,box),
        current_geometry_candidates={kind:current_candidates(mask,box) for kind,mask in [('count20',count),('supported_original_contour',supported)]},
        parameters=r['parameters'],controls_frozen_sha256=controls_hash,
        camera_noise_probability_calibrated=False,frequency_is_not_verified_motor_RPM=True,
        candidate_geometry_is_not_independent_truth=True)
    for key,mask in [('predictive_mask',r['mask']),('supported_original_contour',supported),('count20_mask',count)]:cv2.imwrite(str(target/(key+'.png')),mask)
    np.savez_compressed(target/'evidence.npz',frequency_hz=r['frequency_hz'],spatial_phase=r['spatial_phase'],evidence=r['evidence'],temporal_coverage=r['temporal_coverage'])
    (target/'record.json').write_text(json.dumps(record,indent=2),'utf-8')
    lo=np.maximum(0,np.floor(box[:2]-40)).astype(int);hi=np.minimum([1280,720],np.ceil(box[2:]+40)).astype(int)
    native=np.full((720,1280,3),52,np.uint8);e20=e[e['t']>=round(t*1e6)-20000]
    native[e20['y'],e20['x']]=np.where(e20['p'][:,None]>0,np.array([255,255,255]),np.array([220,110,40]))
    panels=[]
    for title,img in [('Native ON/OFF 20ms',native),('Original count geometry',cv2.cvtColor(count,cv2.COLOR_GRAY2BGR)),
        ('Past-half discovery + later-half check',cv2.cvtColor(r['mask'],cv2.COLOR_GRAY2BGR)),
        ('Original contour near qualified evidence',cv2.cvtColor(supported,cv2.COLOR_GRAY2BGR))]:
        crop=img[lo[1]:hi[1],lo[0]:hi[0]];scale=min(470/crop.shape[1],300/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
        panel=np.full((350,490,3),245,np.uint8);panel[40:40+crop.shape[0],10:10+crop.shape[1]]=crop;cv2.putText(panel,title,(10,25),0,.45,(30,30,30),1);panels.append(panel)
    cv2.imwrite(str(target/'comparison.jpg'),np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]))
    print(json.dumps(record),flush=True);return record


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=ROOT.parent/'output/three_continuation_20261026';controls=out.parent/'controls/controls.json'
    controls_hash=hashlib.sha256(controls.read_bytes()).hexdigest();records=[]
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ['predictive_harmonics.py','tools/probe_predictive_harmonics.py']:
        p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    for folder,selected in [('raw_release_clear',{13:'clear',44:'partial'}),('raw_release_side',{22:'side'})]:
        prior=source/folder;protocol=json.loads((prior/'protocol.json').read_text('utf-8'))
        rows=[json.loads(s) for s in (prior/'predictions.jsonl').read_text('utf-8').splitlines()]
        raw=Path(protocol['raw']);before=raw.stat()
        iterator=native_batches(raw,protocol['actual_iterator_start'],protocol['actual_duration'],False,protocol['native_step_us'])
        for k,(t,e) in enumerate(iterator):
            if k in selected:
                assert abs(t-rows[k]['t'])<1e-8;records.append(process(out,selected[k],rows[k],t,e,controls_hash))
            if k>=max(selected):break
        iterator.close()
        assert before.st_size==raw.stat().st_size and before.st_mtime_ns==raw.stat().st_mtime_ns
    result=dict(records=records,RAW_unchanged=True,saved_boxes_used_only_for_display_and_geometry_ROI=True,
        periodic_pixel_inference_whole_frame=True,labels_read=False,one_frozen_setting=True,
        default_prediction_unchanged=True,not_full_DDHF_reproduction=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8')


if __name__=='__main__':main(sys.argv[1])
