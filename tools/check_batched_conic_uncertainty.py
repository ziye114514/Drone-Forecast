"""Compare current conic uncertainty against the frozen finite-difference code."""
from pathlib import Path
import sys,json,time,importlib.util
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src')]
from event_drone_forecast import geometry as current
from fast_structure_front import count_pair,roi_bounds,NativeROI
from live_decode import native_batches


def run(source,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);source=Path(source)
    frozen=ROOT.parent/'output/three_continuation_20261026/geometry_before_speed.py'
    spec=importlib.util.spec_from_file_location('frozen_conic_geometry',frozen)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    prot=json.loads((source/'protocol.json').read_text('utf-8'))
    data=list(map(json.loads,(source/'predictions.jsonl').read_text('utf-8').splitlines()))
    selector=NativeROI();cases=[];line_error=0.
    for row,(t,events) in zip(data,native_batches(prot['raw'],prot['actual_iterator_start'],prot['actual_duration'],False,prot['native_step_us'])):
        assert abs(row['t']-t)<1e-8
        if not row['body'].get('detected'):continue
        bounds=roi_bounds(row['body']['bbox']);on,off=count_pair(events,bounds,t,selector)
        feature=np.minimum(cv2.GaussianBlur(on,(3,3),.8),cv2.GaussianBlur(off,(3,3),.8));feature[(on+off)<2]=0
        mask=np.uint8(cv2.GaussianBlur(feature,(3,3),.8)>=4.)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours=sorted([c for c in contours if len(c)>=16 and cv2.contourArea(c)>=20],key=cv2.contourArea,reverse=True)[:6]
        for c in contours:
            points=np.asarray(c,float).reshape(-1,2);p=current._fit(points)
            a=old._line(points,500.);b=current._line(points,500.)
            assert (a is None)==(b is None)
            if a is not None:
                for key in a:
                    if isinstance(a[key],(float,int,np.ndarray)):
                        line_error=max(line_error,float(np.max(np.abs(np.asarray(a[key])-b[key]))))
            if p is not None:cases.append((points,p))
    errors=[];cost={name:[] for name in ['old','batched']}
    for i,(points,p) in enumerate(cases):
        for rep in range(3):
            values={}
            for name in (['old','batched'] if (i+rep)%2==0 else ['batched','old']):
                start=time.perf_counter();values[name]=(old if name=='old' else current)._uncertainty(points,p)
                cost[name].append((time.perf_counter()-start)*1000)
            errors.append(float(np.max(np.abs(np.asarray(values['old'])-values['batched']))))
    assert max(errors)<1e-10 and line_error==0.,(max(errors),line_error)
    result=dict(current_conic_cases=len(cases),calls_per_method=len(cost['old']),
        max_uncertainty_absolute_difference=max(errors),line_outputs_max_difference=line_error,
        no_cached_uncertainty=True,no_qualifications_or_fit_thresholds_changed=True,
        timing_scope='Uncertainty component on actual current contour fits; not full FPS',
        timing_ms={n:dict(median=float(np.median(v)),p95=float(np.percentile(v,95))) for n,v in cost.items()})
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
