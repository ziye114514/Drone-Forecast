"""Reuse cached native windows and exact stage21 past-only event translation."""
from pathlib import Path
import sys,json,time,os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[variable]='1'
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from predictive_harmonics import predictive_harmonics
from native_event_alignment import past_warp
from probe_harmonic_pixels import count_mask,summary_mask
from probe_predictive_harmonics import current_candidates


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    prior=out.parent/'native_pilot';records=[]
    for name,folder in [('clear','raw_release_clear'),('partial','raw_release_clear'),('side','raw_release_side')]:
        packet=np.load(prior/name/'native_window.npz');e=packet['events'];t=float(packet['t'])
        rows=[json.loads(s) for s in (ROOT.parent/'output/three_continuation_20261026'/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        i=min(range(len(rows)),key=lambda i:abs(rows[i]['t']-t));old=rows[i-1];body=rows[i]['body']
        warped,diag=past_warp(e,t,old['coefficient']['velocity'],old['t'])
        zero,_=past_warp(e,t,[0.,0.],old['t']);np.testing.assert_array_equal(zero,e)
        refused=False
        try:past_warp(e,t,[1.,0.],t)
        except ValueError:refused=True
        assert refused
        start=time.perf_counter();result=predictive_harmonics(warped,t);elapsed=(time.perf_counter()-start)*1000
        count=count_mask(warped,t,20000);record=json.loads((prior/name/'record.json').read_text('utf-8'))
        directory=out/name;directory.mkdir()
        np.savez_compressed(directory/'aligned_events.npz',events=warped,t=t)
        np.savez_compressed(directory/'evidence.npz',frequency_hz=result['frequency_hz'],evidence=result['evidence'],spatial_phase=result['spatial_phase'])
        cv2.imwrite(str(directory/'predictive_mask.png'),result['mask']);cv2.imwrite(str(directory/'count20_mask.png'),count)
        candidates=current_candidates(count,body['bbox'])
        for candidate in candidates:
            cx,cy=candidate['center'];xx,yy=np.meshgrid(np.arange(1280),np.arange(720))
            # A generous local bounding support count only; not a rotor label.
            support=(abs(xx-cx)<=candidate['major']/2)&(abs(yy-cy)<=candidate['major']/2)
            candidate['nearby_qualified_period_pixels']=int(np.count_nonzero(support&(result['mask']>0)))
        r=dict(name=name,t=t,causal_translation=diag,native_events_cached_not_new_RAW=True,
            original_accepted_pixels=record['accepted_pixels'],aligned_accepted_pixels=result['accepted_pixels'],
            aligned_CPU_ms=elapsed,aligned_count20=summary_mask(count,body['bbox']),
            aligned_periodicity=summary_mask(result['mask'],body['bbox']),
            original_candidates=record['current_geometry_candidates']['count20'],aligned_candidates=candidates,
            zero_velocity_exact=True,future_velocity_rejected=True,new_RPM_or_pose_truth=False)
        records.append(r);(directory/'record.json').write_text(json.dumps(r,indent=2),'utf-8');print(json.dumps(r),flush=True)
        box=np.asarray(body['bbox']);lo=np.maximum(0,np.floor(box[:2]-45)).astype(int);hi=np.minimum([1280,720],np.ceil(box[2:]+45)).astype(int)
        panels=[]
        for title,image in [('Original per-pixel periodic evidence',cv2.imread(str(prior/name/'predictive_mask.png'))),
            ('Past-velocity-aligned periodic evidence',cv2.cvtColor(result['mask'],cv2.COLOR_GRAY2BGR)),
            ('Original20ms geometry contours',cv2.imread(str(prior/name/'count20_mask.png'))),
            ('Aligned20ms geometry contours',cv2.cvtColor(count,cv2.COLOR_GRAY2BGR))]:
            crop=image[lo[1]:hi[1],lo[0]:hi[0]];scale=min(470/crop.shape[1],300/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            panel=np.full((350,490,3),245,np.uint8);panel[40:40+crop.shape[0],10:10+crop.shape[1]]=crop;cv2.putText(panel,title,(10,25),0,.44,(30,30,30),1);panels.append(panel)
        cv2.imwrite(str(directory/'comparison.jpg'),np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]))
    (out/'summary.json').write_text(json.dumps(dict(records=records,settings_frozen=True,labels_read=False,
        same_events_as_previous_pilot=True,geometry_or_prediction_default_changed=False),indent=2),'utf-8')


if __name__=='__main__':main(sys.argv[1])
