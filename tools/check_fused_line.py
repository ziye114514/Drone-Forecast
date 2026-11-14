"""Numerical/gate equivalence of the existing line qualification, not new fit rules."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from native_geometry_numba import JITLine
from fast_geometry_compute import line_geometry_fast
from check_final_compute import compare


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);native=JITLine();rng=np.random.default_rng(281001)
    samples=[]
    for i in range(120):
        n=int(rng.integers(16,450));a=rng.uniform(6,300);b=rng.uniform(.1,80);angle=rng.uniform(0,np.pi)
        x=rng.uniform(-a,a,n);y=rng.uniform(-b,b,n)
        points=np.column_stack((x*np.cos(angle)-y*np.sin(angle),x*np.sin(angle)+y*np.cos(angle)))+rng.uniform(0,600,2)
        samples.append(points)
    for angle in [0.,np.pi/2,np.pi/4]:
        x=np.linspace(-30,30,61)
        samples.append(np.column_stack((x*np.cos(angle),x*np.sin(angle))))
    difference=0.;accepted=0
    for points in samples:
        a=line_geometry_fast(points,500.);b=native(points,500.)
        assert (a is None)==(b is None)
        if a is not None:
            d=compare(a,b);assert d<1e-8,d;difference=max(difference,d);accepted+=1
    timing={key:[] for key in ['numpy','fused']}
    for repeat in range(3):
        order=[('numpy',line_geometry_fast),('fused',native)] if repeat%2==0 else [('fused',native),('numpy',line_geometry_fast)]
        for key,fn in order:
            start=time.perf_counter()
            for points in samples:fn(points,500.)
            timing[key].append((time.perf_counter()-start)*1000)
    result=dict(samples=len(samples),accepted=accepted,all_acceptance_decisions_same=True,max_numeric_difference=difference,
        median_all_samples_ms={k:float(np.median(v)) for k,v in timing.items()},startup=native.provenance,
        component_only_not_complete_pipeline_FPS=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
