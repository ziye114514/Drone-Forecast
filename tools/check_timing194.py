"""Adversarial checks for pooling; not substitute for real video evaluation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from regional_timing_period import measure

if __name__=='__main__':
    dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);rng=np.random.default_rng(194)
    times=[];pixels=[]
    for k in range(8):
        xx=np.arange(64);times.extend(k*1e6/240+300+rng.normal(0,20,64));pixels.extend(xx)
    e=np.zeros(len(times),dtype);e['t']=np.asarray(times,int);p=np.asarray(pixels);e['x']=16+p%8;e['y']=16+p//8;e['p']=1;e.sort(order='t')
    m,d,g=measure(e,.033333,width=64,height=64);assert g,'Sparse monopolar periodic patch should be detected'
    # Each individual pixel has fewer than six events in at least one half.
    assert max(np.bincount((e[e['t']<16666]['y']*64+e[e['t']<16666]['x']).astype(int)))<6
    hot=e.copy();hot['x']=20;hot['y']=20;assert not measure(hot,.033333,width=64,height=64)[2]
    future=np.zeros(50,dtype);future['t']=40000;future['x']=30;future['y']=30
    gm=measure(np.r_[e,future],.033333,width=64,height=64)[2];assert gm==g
    jitter=e.copy();blocks=jitter['t']//1042;jitter['t']=np.minimum(33332,blocks*1042+rng.integers(0,1042,len(jitter)));jitter.sort(order='t')
    random_groups=measure(jitter,.033333,width=64,height=64)[2];assert not random_groups,random_groups
    out=ROOT.parent/'output/timing194_20261107';out.mkdir(exist_ok=True)
    checks=dict(sparse_monopolar_periodic_patch_detected=True,individual_pixel_6_per_half_requirement_not_met=True,
        single_hot_pixel_rejected=True,events_after_cutoff_ignored=True,nuisance_block_shuffled_signal_rejected=True,
        detected_frequencies_hz=[x['frequency_hz'] for x in g],not_a_real_camera_noise_calibration=True)
    (out/'checks.json').write_text(json.dumps(checks,indent=2),'utf-8');print(json.dumps(checks))
