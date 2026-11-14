"""One force-constrained current-state estimator, with matched flat-plane control."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from rotor_force_window import RotorForceWindow
from extend_inverse321 import references
from score_rotor164 import score

if __name__=='__main__':
    out=ROOT.parent/'output/window393_20261110';out.mkdir(exist_ok=False);files=['rotor_force_window.py','tools/probe_window393.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    counts={}
    for n in ('13','15','16','11','3','4'):
        rr=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()];counts[n]={}
        for arm,flat in [('plane',False),('flat',True)]:
            m=RotorForceWindow(flat=flat);values=[m.step(r) for r in rr]
            assert all(r['valid']==old['valid'] and r['body']==old['body'] for r,old in zip(values,rr))
            (out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in values),'utf8');counts[n][arm]=dict(applied=sum(r['rotor_force_window']['qualified'] for r in values),identified=sum(bool(r['rotor_force_window'].get('normal_amplitude_data_identified')) for r in values))
        print(n,'inference frozen',counts[n],flush=True)
    cases={}
    for n in counts:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),**{a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in ('plane','flat')});print(n,{a:r['error'] for a,r in cases[n].items()},flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=cases,counts=counts,source_sha256=hashes,past_only=True,no_video_specific_rules=True,all_inference_before_reference_reads=True,RAW_FPS_measured=False),indent=2),'utf8')
