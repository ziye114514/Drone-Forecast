"""The same finite-difference geometry calculation with a fused native loop."""
from pathlib import Path
import sys,json,time,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from native_geometry_numba import JITUncertainty
from event_drone_forecast.geometry import _uncertainty,ellipse_points
from fast_structure_front import FastStructure
from check_final_compute import compare


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);kernel=JITUncertainty();cv2.setNumThreads(2)
    rng=np.random.default_rng(271031);samples=[];max_difference=0.
    for i in range(40):
        a=rng.uniform(12,100);b=rng.uniform(max(2.,a*.16),a*.98)
        p=np.r_[rng.uniform(100,900,2),a,b,rng.uniform(0,np.pi)]
        phases=np.linspace(0,2*np.pi,150);points=ellipse_points(p,phases)+rng.normal(0,.3,(150,2))
        old=_uncertainty(points,p);new=kernel(points,p)
        np.testing.assert_allclose(old,new,rtol=1e-8,atol=1e-8)
        max_difference=max(max_difference,float(np.max(abs(np.asarray(old)-new))));samples.append((points,p))
    timing={k:[] for k in ['numpy','fused']}
    for repeat in range(3):
        order=[('numpy',_uncertainty),('fused',kernel)] if repeat%2==0 else [('fused',kernel),('numpy',_uncertainty)]
        for label,fn in order:
            for points,p in samples:
                begin=time.perf_counter();fn(points,p);timing[label].append((time.perf_counter()-begin)*1000)
    dtype=np.dtype([('x','u2'),('y','u2'),('t','i8'),('frequency','f4')]);actual=[]
    for name,folder in [('clear','raw_release_clear'),('partial','raw_release_clear'),('side','raw_release_side')]:
        native=np.load(ROOT.parent/'output/predictive_harmonics_20261027/native_pilot'/name/'native_window.npz');t=float(native['t'])
        rows=[json.loads(s) for s in (ROOT.parent/'output/three_continuation_20261026'/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        row=min(rows,key=lambda r:abs(r['t']-t))
        models=[FastStructure(optimized_compute=True),FastStructure(optimized_compute=True,jit_counts=True,jit_conics=True)]
        values=[m.step(native['events'],t,row['body'],np.zeros(0,dtype),row['coefficient']['velocity'],True) for m in models]
        delta=compare(*values)
        actual.append(dict(name=name,max_output_difference=delta))
    result=dict(controlled_ellipses=len(samples),maximum_covariance_diagnostic_difference=max_difference,native_cases=actual,
        median_component_ms={k:float(np.median(v)) for k,v in timing.items()},startup=kernel.provenance,
        finite_difference_points_and_steps_unchanged=True,analytic_covariance_not_used=True,full_pipeline_FPS_not_measured=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
