"""One plane-qualification fix, same current observations and fixed origins."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from adaptive_response_observer import AdaptiveResponseObserver
from plane_axis_uncertainty import apply
from inverse_normal_response import InverseNormalResponse
from anisotropic_response import AnisotropicResponse
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score


if __name__=='__main__':
    out=ROOT.parent/'output/axis381_20261109';out.mkdir(exist_ok=False)
    files=['plane_axis_uncertainty.py','tools/replay_axis381.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    counts={}
    for n in ('13','15','16','11','3','4'):
        rows=[json.loads(s) for s in (ROOT.parent/f'output/inverse326_20261109/{n}_inverse320.jsonl').read_text('utf8').splitlines()]
        m=AdaptiveResponseObserver();inv=InverseNormalResponse();disk=AnisotropicResponse();outputs=[];used=0
        for old in rows:
            r=copy.deepcopy(old);used+=apply(r['structure'],r['geometry']);u=r['force'].get('input_u');u=None if u is None else np.asarray(u);meta=r['period_selection']
            fresh=m.x is None or m.seen is not None and r['t']-m.seen>.5
            if fresh and r['body'].get('detected') and not r['body'].get('center_provisional'):m.two_point_init=bool(r['state'].get('two_point_initialization_selected_by_partial_entry'))
            c,v,f,s=m.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'],structure=r['structure'])
            r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'));r=disk.step(inv.step(r));r['physical']=evaluate(r,[.1,.2,.4]) if c else {};outputs.append(r);assert v==old['valid']
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8');counts[n]=used;print(n,'frozen',used,flush=True)
    cases={}
    for n in counts:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);origins=np.round(np.arange(10.8,15.6001,.2),1)
        cases[n]=dict(control=score(ROOT.parent/f'output/force357_20261109/{n}.jsonl',ref,origins),axis_information=score(out/f'{n}.jsonl',ref,origins));print(n,{k:v['error'] for k,v in cases[n].items()},flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=cases,counts=counts,source_sha256=hashes,one_global_rule=True,NN_used=False,no_parameter_search=True),indent=2),'utf8')
