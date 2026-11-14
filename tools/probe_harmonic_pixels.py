"""Three native RAW snapshots, no threshold search, labels, training or GPU."""
from pathlib import Path
import sys,os,json,time,hashlib
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[variable]='1'
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from live_decode import native_batches
from harmonic_pixels import harmonic_pixels


def count_mask(e,t,window_us):
    e=e[e['t']>=round(t*1e6)-window_us]
    idx=e['y'].astype(np.int64)*1280+e['x'];v=np.bincount(idx+e['p'].astype(np.int64)*1280*720,minlength=2*1280*720).reshape(2,720,1280).astype(np.float32)
    feature=np.minimum(cv2.GaussianBlur(v[0],(3,3),.8),cv2.GaussianBlur(v[1],(3,3),.8));feature[v.sum(0)<2]=0
    return np.uint8(cv2.GaussianBlur(feature,(3,3),.8)>=4.*window_us/20000)*255


def summary_mask(mask,box):
    x,y,X,Y=np.rint(box).astype(int);x,y=max(x,0),max(y,0);X,Y=min(X,1280),min(Y,720)
    inside=np.count_nonzero(mask[y:Y,x:X]);total=np.count_nonzero(mask)
    return dict(retained_pixels=int(total),inside_saved_body_box=int(inside),outside_saved_body_box=int(total-inside),
        fraction_inside_saved_body_box=inside/max(total,1),this_is_not_labelled_precision_or_recall=True)


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);cases=[]
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ['harmonic_pixels.py','tools/probe_harmonic_pixels.py']:
        p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    specs=[('clear','raw_release_clear',13),('partial','raw_release_clear',44),('side','raw_release_side',22)]
    source=ROOT.parent/'output/three_continuation_20261026'
    for name,folder,index in specs:
        prior=source/folder;protocol=json.loads((prior/'protocol.json').read_text('utf-8'))
        rows=[json.loads(s) for s in (prior/'predictions.jsonl').read_text('utf-8').splitlines()]
        raw=Path(protocol['raw']);before=raw.stat();frame=rows[index]
        iterator=native_batches(raw,protocol['actual_iterator_start'],protocol['actual_duration'],False,protocol['native_step_us'])
        for k,(t,e) in enumerate(iterator):
            if k!=index:continue
            assert abs(t-frame['t'])<1e-8
            a=time.perf_counter();result=harmonic_pixels(e,t,protocol['native_step_us']);elapsed=(time.perf_counter()-a)*1000
            count20=count_mask(e,t,20000);count33=count_mask(e,t,protocol['native_step_us'])
            directory=out/name;directory.mkdir()
            cv2.imwrite(str(directory/'harmonic_mask.png'),result['mask']);cv2.imwrite(str(directory/'count20_mask.png'),count20)
            cv2.imwrite(str(directory/'count33_mask.png'),count33)
            np.savez_compressed(directory/'frequency_evidence.npz',frequency_hz=result['frequency_hz'],comb_score=result['comb_score'],spectral_flatness=result['spectral_flatness'])
            body=frame['body'];assert body.get('detected')
            box=np.asarray(body['bbox']);lo=np.maximum(0,np.floor(box[:2]-40)).astype(int);hi=np.minimum([1280,720],np.ceil(box[2:]+40)).astype(int)
            onoff=np.full((720,1280,3),52,np.uint8)
            recent=e[e['t']>=round(t*1e6)-20000]
            onoff[recent['y'],recent['x']]=np.where(recent['p'][:,None]>0,np.array([255,255,255]),np.array([220,110,40]))
            views=[onoff,cv2.cvtColor(count20,cv2.COLOR_GRAY2BGR),cv2.cvtColor(count33,cv2.COLOR_GRAY2BGR),cv2.cvtColor(result['mask'],cv2.COLOR_GRAY2BGR)]
            panels=[]
            for title,img in zip(['Native ON/OFF 20ms','Count threshold 20ms','Count rate threshold 33ms','DDHF-like NDFT 33ms'],views):
                crop=img[lo[1]:hi[1],lo[0]:hi[0]];scale=min(470/crop.shape[1],300/crop.shape[0])
                resize=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
                panel=np.full((350,490,3),245,np.uint8);panel[40:40+resize.shape[0],10:10+resize.shape[1]]=resize
                cv2.putText(panel,title,(10,25),0,.5,(30,30,30),1);panels.append(panel)
            cv2.imwrite(str(directory/'comparison.jpg'),np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]))
            cases.append(dict(name=name,t=t,source_mode=frame['structure']['mode'],raw_unchanged=before.st_size==raw.stat().st_size and before.st_mtime_ns==raw.stat().st_mtime_ns,
                raw_events=len(e),ndft_CPU_ms=elapsed,eligible_pixels=result['eligible_pixels'],parameters=result['parameters'],
                count20=summary_mask(count20,body['bbox']),count33=summary_mask(count33,body['bbox']),
                event_count_only_min6=summary_mask(result['eligible_mask'],body['bbox']),
                harmonic=summary_mask(result['mask'],body['bbox']),
                frequency_is_detection_fingerprint_not_calibrated_motor_RPM=True))
            print(json.dumps(cases[-1]),flush=True);break
        iterator.close()
    summary=dict(cases=cases,author_code_used=False,full_author_reproduction=False,
        RAW_only_segmentation=True,saved_NN_boxes_used_only_for_display_and_proxy_counts=True,
        labels_read=False,model_or_prediction_default_changed=False,end_to_end_FPS_measured=False,
        no_parameter_sweep=True,reference='https://arxiv.org/abs/2603.08386')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')


if __name__=='__main__':run(sys.argv[1])
