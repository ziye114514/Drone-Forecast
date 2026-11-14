"""Slow parameter identification, current measured velocity retained, all sources."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from rotor_force_window import RotorForceWindow
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/parameters394_20261110';out.mkdir(exist_ok=False)
    files=['rotor_force_window.py','tools/probe_parameters394.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    counts={}
    for n in ('13','15','16','11','3','4'):
        rr=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()];m=RotorForceWindow(parameter_window=True);values=[m.step(r) for r in rr]
        assert all(r['valid']==old['valid'] and r['body']==old['body'] and (not old['coefficient'] or r['coefficient']['velocity']==old['coefficient']['velocity']) for r,old in zip(values,rr))
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in values),'utf8');counts[n]=sum(r['rotor_force_window']['qualified'] for r in values);print(n,'frozen',counts[n],flush=True)
    cases={}
    for n in counts:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),parameters=score(out/f'{n}.jsonl',ref,origins));print(n,{a:r['error'] for a,r in cases[n].items()},flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=cases,applied=counts,source_sha256=hashes,parameter_history_3s_from_existing_gain_learner=True,no_scan=True,past_only=True,all_inference_before_reference_reads=True),indent=2),'utf8')
