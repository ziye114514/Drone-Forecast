"""Dense nonperiodic adversaries and exact native-vs-NumPy preprocessing."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import regional_timing_period as module
from native_timing_histogram import NativeTimingHistogram
from pure_visibility170 import window
OUT=ROOT.parent/'output/timing208_20261108'
if __name__=='__main__':
    engine=NativeTimingHistogram(OUT/'timing_histogram.dll');records=[]
    dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')])
    for seed in range(8):
        rng=np.random.default_rng(seed);e=np.empty(20000,dtype);pixel=rng.integers(0,64,len(e));e['x']=16+pixel%8;e['y']=16+pixel//8;e['p']=1;e['t']=rng.integers(0,33333,len(e));e.sort(order='t')
        module.HISTOGRAM_ENGINE=None;a,da,ga=module.measure(e,.033333,width=64,height=64)
        module.HISTOGRAM_ENGINE=engine;b,db,gb=module.measure(e,.033333,width=64,height=64)
        assert np.array_equal(a,b) and ga==gb
        records.append(dict(seed=seed,negative='dense independent nonperiodic events',groups=len(gb),maximum_score=db.get('maximum_score')))
    assert sum(r['groups'] for r in records)<=1,records
    # A real periodic sparse patch still has no six-per-half pixel repetitions.
    e=np.empty(8*64,dtype);rng=np.random.default_rng(194);e['t']=np.repeat(np.arange(8)*1e6/240+300,64)+rng.normal(0,20,len(e));p=np.tile(np.arange(64),8);e['x']=16+p%8;e['y']=16+p//8;e['p']=1;e.sort(order='t')
    assert module.measure(e,.033333,width=64,height=64)[2]
    real=[]
    for n,parent,t in [(15,'motion181_20261107',13.5),(12,'phase185_20261107',12.),(18,'phase185_20261107',13.5)]:
        e=window(ROOT.parent/f'output/{parent}/cache_recording{n}',t);arms={};outputs=[]
        for name,backend in [('numpy',None),('native',engine)]:
            module.HISTOGRAM_ENGINE=backend;st=time.perf_counter();mask,diag,groups=module.measure(e,t);elapsed=1000*(time.perf_counter()-st)
            arms[name]=dict(ms=elapsed,qualified_groups=len(groups),maximum_score=diag.get('maximum_score'));outputs.append((mask,groups))
        assert np.array_equal(outputs[0][0],outputs[1][0]) and outputs[0][1]==outputs[1][1]
        real.append(dict(recording=n,t=t,arms=arms,groups=outputs[1][1]))
        print(n,arms,flush=True)
    engine.close();module.HISTOGRAM_ENGINE=None
    result=dict(dense_negative_controls=records,sparse_periodic_positive_passed=True,native_numpy_outputs_equal=True,real_frames=real,
        no_metric_RPM_claimed=True,camera_FAR_calibrated=False,RAW_FPS_measured=False)
    (OUT/'checks.json').write_text(json.dumps(result,indent=2),'utf-8');print('All native histogram and adversarial checks passed',flush=True)
