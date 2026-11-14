"""Accuracy-only current joint phase/conic observations, same rule on all6.

Reuse existing author-assisted ring fitting and native late-half qualification.
No new motor-speed thresholds, manual initialization or reference boxes. This
captures actual current refits, unlike311 transported geometry, and is slow CPU
research; no30FPS claim. Do not edit snapshotted modules while running.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from joint_phase_bank import JointPhaseBank
from pure_visibility170 import window

if __name__=='__main__':
    out=ROOT.parent/'output/phase395_20261110';out.mkdir(exist_ok=False)
    files=['joint_phase_bank.py','joint_ring_phase.py','angular_phase_candidates.py','angular_projection.py','tools/capture_phase395.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,same_algorithm_every_source=True,accuracy_only=True,CPU_author_core_reused_not_redistributed=True,NN_used=False,manual_references_read=False,new_rotation_thresholds=False,old244_already_used_normal_refinement_so_not_claimed_as_new_idea=True,purpose='test cumulative native current geometry-only refinement with newer inverse/anisotropic/unknown-response pipeline'),indent=2),'utf8')
    summary={};dll=ROOT.parent/'output/helix_cpu232_20261108/build/Release/helix_reference.dll'
    for n in ('13','15','16','11','3','4'):
        bank=JointPhaseBank(dll);rows=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()]
        cache=ROOT.parent/('output/frozen324_20261109/cache' if n=='4' else 'output/frozen_tangent287_20261108/cache' if n=='3' else 'output/motion181_20261107/cache_recording15' if n=='15' else f'output/phase185_20261107/cache_recording{n}')
        manifest=json.loads((cache/'manifest.json').read_text('utf8'))['frames'] if n in ('3','4') else None;cost=[];counts=[]
        for i,row in enumerate(rows):
            t=row['t'];e=np.load(cache/manifest[i]['file']) if manifest else window(cache,t);begin=time.perf_counter();u,q,meta=bank.step(t,e,row['geometry'].get('rotors',[]),row['motion']);cost.append(1000*(time.perf_counter()-begin));counts.append(meta.get('late_qualified_before_temporal_confirmation',0))
            r=dict(t=t,u=None if u is None else np.asarray(u).tolist(),quality=q,meta=meta)
            with (out/f'{n}_periods.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
            if i%45==0:print(n,i,'current refitted rings',counts[-1],flush=True)
        summary[n]=dict(frames=len(rows),two_or_more_current_rings=sum(k>=2 for k in counts),CPU_phase_ms_mean=float(np.mean(cost)),CPU_phase_ms_p95=float(np.percentile(cost,95)));print(n,'frozen',summary[n],flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (out/'completion.json').write_text(json.dumps(dict(summary=summary,manual_references_read=False,RAW_FPS_measured=False),indent=2),'utf8')
