"""Interleaved same-objective solver benchmark; not end-to-end RAW speed."""
from pathlib import Path
import argparse,datetime,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage113')]
from study113 import geom,rows,save,sha,SUPPORT
OUT=ROOT.parent/'output/stage116_analytic_solver_benchmark_20261022'
CASES={'clear6':[30,70,110,150,190,230],'continuous6':[100,160,210,310,380,440]}
def sources():return {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),ROOT/'research/stage113/support113.py',ROOT/'research/stage58/geometry58.py']}
def freeze():
    assert not OUT.exists();save(OUT/'protocol116.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),
        indexes=CASES,seed=116,rounds=2,design='Predeclared source indices, same four fitted ellipses, original58 finite-difference Jacobian versus113 analytic Jacobian and covariance. ABBA/BAAB randomly fixed per pair. Report non-four/line inputs and convergence differences; no horizon target read.',
        scope='Single-module interleaved wall time, not RAW/NN throughput, source50fps or end-to-end realtime. Same objective; floating numerical paths can change termination and are reported. No original default change.'))
def run():
    p=json.loads((OUT/'protocol116.json').read_text('utf-8'));assert p['sources']==sources();rng=np.random.default_rng(116);records=[];skipped=[]
    for case,indexes in CASES.items():
        rr=rows(SUPPORT/case/'supports.jsonl')
        for index in indexes:
            r=rr[index];rotors=r['geometry'].get('rotors',[])
            if len(rotors)!=4 or any(q['mode']!='ellipse' for q in rotors):skipped.append(dict(case=case,index=index,t=r['t'],count=len(rotors)));continue
            ellipses=[geom.geometry58.from_rotor(q) for q in rotors]
            for repeat in range(2):
                seq=['fd','analytic','analytic','fd'] if rng.integers(2) else ['analytic','fd','fd','analytic'];times={'fd':[],'analytic':[]};outputs={}
                for mode in seq:
                    tick=time.perf_counter();out=geom.geometry58.joint_fit(ellipses) if mode=='fd' else geom.support_fit(ellipses)
                    elapsed=(time.perf_counter()-tick)*1000;times[mode].append(elapsed);outputs[mode]=out
                a=outputs['fd'];b=outputs['analytic'];dc=None;dr=None
                if 'geometric_center' in a and 'geometric_center' in b:
                    dc=float(np.linalg.norm(np.asarray(a['geometric_center'])-b['geometric_center']));dr=abs(a['rms_px']-b['rms_px'])
                records.append(dict(case=case,index=index,t=r['t'],repeat=repeat,order=seq,
                    fd_ms=float(np.median(times['fd'])),analytic_ms=float(np.median(times['analytic'])),
                    speed_ratio=float(np.median(times['fd'])/np.median(times['analytic'])),center_difference_px=dc,rms_difference_px=dr,
                    fd_success=a.get('optimizer_success'),analytic_success=b.get('optimizer_success')))
    summary=dict(pairs=len(records),skipped=len(skipped),median_pair_speed_ratio=float(np.median([r['speed_ratio'] for r in records])),
        fd_median_ms=float(np.median([r['fd_ms'] for r in records])),analytic_median_ms=float(np.median([r['analytic_ms'] for r in records])),
        maximum_center_difference_px=max(r['center_difference_px'] or 0 for r in records),convergence_disagreements=sum(r['fd_success']!=r['analytic_success'] for r in records))
    save(OUT/'benchmark116.json',dict(summary=summary,records=records,skipped=skipped));print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
