"""Predeclared16/32/64px cells, equal alpha/3; no timing/recording tuning.

Root: a fast rotor can leave a16px fixed cell before the validation half.
Coarser pooling can keep both halves observable without inventing motor RPM.
"""
from pathlib import Path
import os,sys,json,numpy as np
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
import regional_timing_period as m
from native_timing_histogram import NativeTimingHistogram
from pure_visibility170 import window
OUT=ROOT.parent/'output/multiscale224_20261108'
if __name__=='__main__':
    OUT.mkdir(exist_ok=False);engine=NativeTimingHistogram(ROOT.parent/'output/timing208_20261108/timing_histogram.dll')
    m.HISTOGRAM_ENGINE=engine;original=dict(m.PARAMS);results=[]
    try:
        for n,t in [('11',11.8),('11',12.),('12',11.8),('12',12.)]:
            e=window(ROOT.parent/f'output/phase185_20261107/cache_recording{n}',t);scales=[]
            for size in [16,32,64]:
                m.PARAMS.update(cell_px=size,family_alpha=.01/3)
                mask,diag,groups=m.measure(e,t)
                diag=dict(diag,parameters=dict(diag['parameters']))
                scales.append(dict(cell_px=size,groups=groups,diagnostic=diag))
            results.append(dict(recording=n,t=t,scales=scales));print(n,t,[(v['cell_px'],len(v['groups'])) for v in scales],flush=True)
        dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);negative=[]
        for seed in range(8):
            rng=np.random.default_rng(seed);e=np.zeros(20000,dtype);pixel=rng.integers(0,64,len(e))
            e['x']=16+pixel%8;e['y']=16+pixel//8;e['p']=1;e['t']=rng.integers(0,33333,len(e));e.sort(order='t')
            counts=[]
            for size in [16,32,64]:
                m.PARAMS.update(cell_px=size,family_alpha=.01/3);counts.append(len(m.measure(e,.033333,width=128,height=128)[2]))
            negative.append(dict(seed=seed,group_counts=counts))
        (OUT/'probe.json').write_text(json.dumps(dict(results=results,negative=negative,
            equal_family_alpha_split=.01/3,regional_evidence_not_RPM=True,RAW_FPS_measured=False),indent=2),'utf-8')
    finally:
        m.PARAMS.clear();m.PARAMS.update(original);engine.close();m.HISTOGRAM_ENGINE=None
