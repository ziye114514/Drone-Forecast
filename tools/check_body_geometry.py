"""Compare residuals and complete geometry on identical real ON/OFF packets."""
from pathlib import Path
import os,sys,json,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from native_conic_residual import ConicResidual
from event_drone_forecast.geometry import _distance,ellipse_points
from fast_geometry_compute import configure_residual
from fast_structure_front import FastStructure
from check_final_compute import compare


def main(out):
    out.mkdir(exist_ok=False);backend=ConicResidual();rng=np.random.default_rng(3110);samples=[];max_error=0.
    for i in range(64):
        p=np.r_[rng.uniform(10,1200,2),rng.uniform(10,120),rng.uniform(2,9),rng.uniform(0,np.pi)]
        points=ellipse_points(p,np.linspace(0,2*np.pi,180))+rng.normal(0,.3,(180,2))
        a=_distance(points,p);b=backend(points,p);max_error=max(max_error,float(np.max(abs(a-b))))
        np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-9);samples.append((points,p))
    times={'numpy':[],'fused':[]}
    for repeat in range(4):
        order=[('numpy',_distance),('fused',backend)]
        if repeat%2:order.reverse()
        for name,fn in order:
            start=time.perf_counter()
            for _ in range(10):
                for points,p in samples:fn(points,p)
            times[name].append((time.perf_counter()-start)*1000/640)
    from live_decode import native_batches
    from sdk_activity_backend import SDKActivityAVX2
    import cv2
    cv2.setNumThreads(2);cases=[]
    rows=[json.loads(s) for s in (ROOT.parent/'output/target25_continuity_20261031/RAW_clear_center4s/predictions.jsonl').read_text('utf-8').splitlines()][:60]
    models=[FastStructure(optimized_compute=True,jit_counts=True,jit_conics=True,body_workspace=bool(i)) for i in range(2)]
    noise=SDKActivityAVX2(1280,720,10000);buffer=noise.get_empty_output_buffer()
    begin=rows[0]['t']-.033333;elapsed=[[],[]]
    for i,(t,events) in enumerate(native_batches(Path('E:/dataset/train/6/media/event.raw'),begin,len(rows)*.033333,False,33333)):
        if i>=len(rows):break
        row=rows[i];assert abs(t-row['t'])<1e-7,(t,row['t'])
        noise.process_events(events,buffer);packet=buffer.numpy();values=[]
        for enabled,model in zip((False,True),models):
            configure_residual(enabled);start=time.perf_counter()
            values.append(model.step(packet,t,row['body'],None,row['coefficient']['velocity'] if row['coefficient'] else [0.,0.],True,defer_rates=True))
            elapsed[int(enabled)].append((time.perf_counter()-start)*1000)
        cases.append(dict(t=t,max_output_difference=compare(*values)))
    result=dict(randomized_contours=len(samples),max_residual_difference_px=max_error,
        actual_native_cases=cases,median_component_ms={k:float(np.median(v)) for k,v in times.items()},
        startup=backend.provenance,full_pipeline_FPS_measured=False,
        geometry_median_ms=[float(np.median(v)) for v in elapsed],geometry_p95_ms=[float(np.percentile(v,95)) for v in elapsed])
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
