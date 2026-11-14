"""Matched process control and rate ablation; never used for target fitting."""
from pathlib import Path
import sys,json,numpy as np,copy,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from separated_force_response import SeparatedForceObserver
from force_process_controls201 import MatchedIsotropicControl
from physical_publication import evaluate
from run_rotor_only import serial
BASE=ROOT.parent/'output/separated201_20261107'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recording',choices=['13','14','15','16','side','train2'],required=True);args=p.parse_args()
    out=BASE/f'controls{args.recording}';out.mkdir(exist_ok=False)
    n=args.recording
    source=BASE/'recording13/plane191.jsonl' if n=='13' else ROOT.parent/f'output/{"normal187_20261107/recording15_with_input/normal_legacy_input" if n=="15" else "timing194_20261107/recording"+n+"/plane191"}.jsonl'
    if n in ('side','train2'):source=ROOT.parent/f'output/layout176_20261107/integrated/{n}_fixed.jsonl'
    rows=[json.loads(s) for s in source.read_text('utf-8').splitlines()]
    for arm,cls in [('separated',SeparatedForceObserver),('matched_isotropic',MatchedIsotropicControl)]:
        m=cls();output=[];zero=[]
        for old in rows:
            r=copy.deepcopy(old);meta=r['period_selection'];u=r['force'].get('input_u');u=None if u is None else np.asarray(u)
            if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
            c,v,f,s=m.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'],structure=r['structure'])
            r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'));r['physical']=evaluate(r,[.1,.2,.4]) if c else {};output.append(r)
            if arm=='separated':
                zz=copy.deepcopy(r);zz['force'].pop('external_unknown_rate_px_s3',None);zz['physical']=evaluate(zz,[.1,.2,.4]) if c else {};zero.append(zz)
        (out/f'{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in output),'utf-8')
        if zero:(out/'no_rate_rollout.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in zero),'utf-8')
    files=['separated_force_response.py','force_process_controls201.py','tools/replay_controls201.py'];sha={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    for n in files:
        d=out/'source_snapshot'/n;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/n).read_bytes())
    (out/'completion.json').write_text(json.dumps(dict(source_sha256=sha,labels_used=False,parameters_searched=False),indent=2),'utf-8');print(args.recording,'controls complete',flush=True)
