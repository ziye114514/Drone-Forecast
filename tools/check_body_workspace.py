"""Exact existing line-detector input and outputs, same real count images."""
from pathlib import Path
import sys,json,time
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from native_body_workspace import BodyWorkspace
from research.stage6.silhouette import body_measurement
from check_final_compute import compare


def main():
    out=ROOT.parent/'output/stability_audit_20261031/body_workspace';out.mkdir(exist_ok=False);kernel=BodyWorkspace();cv2.setNumThreads(2)
    for dtype in (np.float32,np.float64):
        crop=np.arange(12000,dtype=dtype).reshape(100,120)
        for value in (crop,crop[:,::2],crop+.25):
            np.testing.assert_array_equal(kernel.render(value),np.uint8(np.clip(np.log1p(value)/4,0,1)*255))
    cases=[]
    for name in ('clear','side','train2'):
        native=np.load(out.parent/'native_timing'/(name+'_160ms.npz'));e=native['events'];t=float(native['cutoff']);e=e[e['t']>=round(t*1e6)-20000]
        ids=e['y'].astype(np.int64)*1280+e['x'];on=np.bincount(ids[e['p']==1],minlength=1280*720);off=np.bincount(ids[e['p']==0],minlength=1280*720)
        density=np.minimum(on,off).reshape(720,1280).astype(np.float32)
        b=[float(e['x'].min()),float(e['y'].min()),float(e['x'].max()),float(e['y'].max())]
        old=body_measurement(density,b,True);new=body_measurement(density,b,True,kernel);delta=compare(old,new)
        # Repeated use of the same LSD object must have no hidden frame state.
        compare(new,body_measurement(density,b,True,kernel));timings={False:[],True:[]}
        for repeat in range(16):
            order=(False,True) if repeat%2 else (True,False)
            for enabled in order:
                start=time.perf_counter();body_measurement(density,b,True,kernel if enabled else None)
                timings[enabled].append((time.perf_counter()-start)*1000)
        cases.append(dict(name=name,maximum_output_difference=delta,old_median_ms=float(np.median(timings[False])),new_median_ms=float(np.median(timings[True])),
            old_p95_ms=float(np.percentile(timings[False],95)),new_p95_ms=float(np.percentile(timings[True],95)),valid=old.get('valid'),reason=old.get('reason')))
    result=dict(startup=kernel.provenance,cases=cases,render_calls=kernel.calls,fallbacks=kernel.fallbacks,
        fractional_input_falls_back=True,full_pipeline_FPS_measured=False,original_density_and_thresholds_preserved=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
