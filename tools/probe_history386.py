"""Small physics-response model with the user's past-force-change feature."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from conditional_unknown_force import teachers,fit,ConditionalUnknownForce
from causal_force_history import CausalForceHistory
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/history386_20261110';out.mkdir(exist_ok=False)
    files=['causal_force_history.py','conditional_unknown_force.py','physical_publication.py','tools/probe_history386.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    data={}
    for n in ('13','15','16','11','3','4'):
        m=CausalForceHistory();data[n]=[m.step(json.loads(s)) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()]
    arms=('disk','camera','no_future_plane');samples={a:{n:teachers(rr,a,history=True) for n,rr in data.items()} for a in arms};models={}
    for n,rr in data.items():
        models[n]={}
        for arm in arms:
            model=fit([g for k,g in samples[arm].items() if k!=n],arm,history=True);models[n][arm]=model
            predictor=ConditionalUnknownForce(model);outputs=[predictor.step(r) for r in rr]
            assert all(r['valid']==old['valid'] and r['coefficient']==old['coefficient'] for r,old in zip(outputs,rr))
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8')
        print(n,'inference frozen',flush=True)
    (out/'models.json').write_text(json.dumps(dict(LOO=models,shared_for_future=fit(list(samples['disk'].values()),history=True),teacher_counts={n:len(s) for n,s in samples['disk'].items()}),indent=2),'utf8')
    cases={}
    for n in data:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in arms});print(n,{a:v['error'] for a,v in cases[n].items()},flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=cases,source_sha256=hashes,all_inference_before_manual_reference_reads=True,all_sources_exposed_development=True,no_per_video_runtime_rules=True,no_NN=True,no_RAW_FPS_measurement=True),indent=2),'utf8')
