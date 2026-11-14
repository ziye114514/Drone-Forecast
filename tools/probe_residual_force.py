"""One fixed causal residual-force hypothesis across already exposed clips."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from causal_residual_force import CausalResidualForce
from probe_gap_segments import OPTIONS
from compare_final_pipeline import reference_at
from score_fred_transfer import reference
from score_fred_adaptation import matched
from prepare_fred_transfer import parse


if __name__=='__main__':
    base=ROOT.parent/'output';out=base/'stability_audit_20261031/residual_force';out.mkdir(exist_ok=False)
    specs=[('train2','target25_continuity_20261031/RAW_other_center6s','dense_labels_train2_40_46'),
        ('side','reacquisition_motion_20261029/RAW_side_default8s','dense_labels_side_110_1195'),
        ('FRED0','target25_continuity_20261031/RAW_fred0_center4s','fred_transfer_20261029/input/0/coordinates.txt'),
        ('FRED1','fred_domain_adaptation_20261029/RAW_1_verified','fred_transfer_20261029/input/1/coordinates.txt'),
        ('FRED2','fred_transfer_20261029/RAW_2_candidate','fred_transfer_20261029/input/2/coordinates.txt'),
        ('FRED7','fred_domain_adaptation_20261029/RAW_7_baseline','fred_transfer_20261029/unseen7')]
    result={}
    for name,path,label in specs:
        rows=[json.loads(x) for x in (base/path/'predictions.jsonl').read_text().splitlines()]
        model=CausalResidualForce();new=[]
        for r in rows:
            p,f,d=model.step(r);new.append(dict(r,physical=p,force=f,causal_residual_force=d))
        (out/(name+'.jsonl')).write_text('\n'.join(json.dumps(r,allow_nan=False) for r in new)+'\n','utf-8')
        if name.startswith('FRED'):
            lp=next((base/label).rglob('coordinates.txt')) if name=='FRED7' else base/label
            labels=parse(lp.read_text('utf-8-sig'))
            def target(t):
                b=reference(labels,t);return None if b is None else (b[:2]+b[2:])/2
        else:
            labels=sorted(json.loads((base/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
            def target(t):return reference_at(labels,t)
        result[name]=dict(matched=matched(rows,new,target),applied=sum(r['causal_residual_force']['applied'] for r in new))
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:dict(old=v['matched']['old_matched_error'],new=v['matched']['new_matched_error'],applied=v['applied']) for k,v in result.items()},indent=2))
