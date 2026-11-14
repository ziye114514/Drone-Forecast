"""One predeclared force-memory change on the three development sequences."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from force_persistence_rollout import ForcePersistenceRollout
from replay_joint236 import source
from score_rotor164 import score
from run_rotor_only import serial

if __name__=='__main__':
    OUT=ROOT.parent/'output/persistence257_20261108';OUT.mkdir(exist_ok=False)
    files=['force_persistence_rollout.py','adaptive_force_memory.py','physical_publication.py','tools/replay_persistence257.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    protocol=dict(development_recordings=['13','15','16'],predeclared_reserved_raw='E:/dataset/experiment/1/media/event.raw',
        reserved_interval=[10.,16.],new_source_not_opened_during_development=True,
        change='Only causal future net-force persistence; current detection, position/velocity/acceleration unchanged',
        memory_hypotheses_s=[.35,.1,'constant'],credit_parameters='reuse AdaptiveForceMemory.allow_shorter unchanged',
        no_search=True,no_recording_specific_settings=True,NN_used=False,GPU_used=False,source_sha256=hashes)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf-8')
    for n in files:
        dst=OUT/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    paths={};rows_by_case={}
    for n in protocol['development_recordings']:
        rows=[json.loads(s) for s in source(n).read_text('utf-8').splitlines()];model=ForcePersistenceRollout();outputs=[]
        for old in rows:
            r=model.step(old);assert r['coefficient']==old['coefficient'] and r['body']==old['body'] and r['structure']==old['structure']
            assert all(p['target']>r['t']-1e-8 for p in model.credit.pending) or not r['motion'].get('current_observation')
            outputs.append(r)
        if n=='15':
            prefix=ForcePersistenceRollout();assert all(prefix.step(old)['physical']==r['physical'] for old,r in zip(rows[:65],outputs))
        path=OUT/f'{n}.jsonl';path.write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in outputs),'utf-8');paths[n]=path;rows_by_case[n]=outputs
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    # Only after all candidate predictions have been frozen, open references.
    results={}
    for n,path in paths.items():
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]=dict(control=score(source(n),ref,origins),persistence=score(path,ref,origins),
            active_frames=sum(r['state'].get('future_force_persistence',{}).get('weights',{}).get('original',1.)<.999999 for r in rows_by_case[n]))
        print(n,{k:v['error']['mean'] for k,v in results[n].items() if isinstance(v,dict)},flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,protocol=protocol),indent=2),'utf-8')
