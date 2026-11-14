"""Reuse matured-error credit for200 vs191 on frozen observations."""
from pathlib import Path
import sys,json,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from causal_response_credit import CausalResponseCredit
from run_rotor_only import serial
BASE=ROOT.parent/'output';OUT=BASE/'credit202_20261108'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['13','14','15','16','side','train2']);args=p.parse_args();n=args.case
    out=OUT/n;out.mkdir(parents=True,exist_ok=False)
    if n=='13':base=BASE/'separated201_20261107/recording13/plane191.jsonl'
    elif n=='15':base=BASE/'normal187_20261107/recording15_with_input/normal_legacy_input.jsonl'
    elif n in ('side','train2'):base=BASE/f'normal187_20261107/{n}/normal_legacy_input.jsonl'
    else:base=BASE/f'timing194_20261107/recording{n}/plane191.jsonl'
    alt=BASE/f'separated201_20261107/controls{n}/separated.jsonl'
    a=[json.loads(s) for s in base.read_text('utf-8').splitlines()];b=[json.loads(s) for s in alt.read_text('utf-8').splitlines()];assert len(a)==len(b)
    model=CausalResponseCredit();rr=[]
    for aa,bb in zip(a,b):assert aa['t']==bb['t'];rr.append(model.step(aa,bb))
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rr),'utf-8')
    ww=[r['state']['causal_response_credit']['weight'] for r in rr]
    (out/'completion.json').write_text(json.dumps(dict(labels_used=False,parameters_searched=False,mean_weight=float(np.mean(ww)),final_weight=ww[-1],mature_rows=len(model.memory.rows)),indent=2),'utf-8')
    print(n,float(np.mean(ww)),ww[-1],len(model.memory.rows),flush=True)
