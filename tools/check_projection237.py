"""Numerical equivalence and bounded real-event CPU timing, not RAW FPS."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from angular_projection import project
from angular_phase_candidates import ring_phase_samples
from predictive_harmonics import BASE_FREQUENCIES
from pure_visibility170 import window


def legacy(tt,angle,ww,frequencies,orders):
    output=[]
    for start in range(0,len(frequencies),32):
        fs=frequencies[start:start+32];ms=orders[start:start+32]
        z=np.exp(1j*(angle[:,None]*ms[None]-2*np.pi*tt[:,None]*fs[None]))
        output.extend((ww[:,None]*z).sum(0)/max(float(ww.sum()),1e-12))
    return np.asarray(output)


if __name__=='__main__':
    OUT=ROOT.parent/'output/projection237_20261108';OUT.mkdir(exist_ok=False)
    old=json.loads((ROOT.parent/'output/pose234_20261108/results.json').read_text())['records']
    # First candidate from each of three different recordings, fixed before timing.
    chosen=[]
    for n in ['11','12','15']:
        chosen.append(next(r for r in old if r['recording']==n))
    records=[];positive=np.unique(BASE_FREQUENCIES[:,None]*np.arange(1,5)[None]);f=np.tile(np.r_[positive,-positive],4);m=np.repeat(np.arange(1,5),len(f)//4)
    for r in chosen:
        n=r['recording'];cache=ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}'
        e=window(cache,r['t']);data,_=ring_phase_samples(e,r['ellipse'],r['t'],r.get('velocity'))
        train=data['train'];tt=(data['events']['t'][train]-data['begin'])/1e6;angle=data['theta'][train];ww=data['weight'][train]
        tick=time.perf_counter();a=legacy(tt,angle,ww,f,m);slow=time.perf_counter()-tick
        tick=time.perf_counter();b=project(tt,angle,ww,f,m);fast=time.perf_counter()-tick
        delta=float(np.max(abs(a-b)));assert delta<1e-11,delta
        assert np.argmax(abs(a))==np.argmax(abs(b))
        records.append(dict(recording=n,events=len(tt),max_complex_delta=delta,legacy_s=slow,factorized_s=fast,speedup=slow/fast))
    (OUT/'checks.json').write_text(json.dumps(dict(records=records,GPU_used=False,RAW_FPS_measured=False,all_native_timestamps_retained=True),indent=2),'utf-8');print(json.dumps(records),flush=True)
