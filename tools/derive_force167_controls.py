"""Matched-calibration counterfactuals, before any target read."""
from pathlib import Path
import sys,json,copy
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_force167 import OUT
from visible_force_dynamics import evaluate_visible
from replay_gated_nn166 import guard
if __name__=='__main__':
    sys.addaudithook(guard)
    for path in OUT.glob('*_full_learned.jsonl'):
        clip=path.name.removesuffix('_full_learned.jsonl');rows=[json.loads(s) for s in path.read_text('utf-8').splitlines()]
        for control in ['normal_only_learned','rate_only_learned']:
            output=[]
            for old in rows:
                r=copy.deepcopy(old)
                if control=='normal_only_learned':r['force167_mode']='normal_only'
                else:r['visible_force_dynamics']['omega']=0.
                r['physical']=evaluate_visible(r,[.1,.2,.4]);output.append(r)
            (OUT/f'{clip}_{control}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in output),'utf-8')
    print('Derived matched-gain direction/frequency controls without annotations')
