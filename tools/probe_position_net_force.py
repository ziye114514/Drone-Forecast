"""One fixed fallback hypothesis on exposed scenes, no parameter search."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from position_net_force import PositionNetForce
from compare_final_pipeline import reference_at
from score_fred_adaptation import matched
from prepare_fred_transfer import parse
from score_fred_transfer import reference


def main():
    base=ROOT.parent/'output';out=base/'stability_audit_20261031/position_net_force';out.mkdir(exist_ok=False);result={}
    specs=[('train2','target25_continuity_20261031/RAW_other_center6s','dense_labels_train2_40_46/labels.json'),
        ('side','reacquisition_motion_20261029/RAW_side_default8s','dense_labels_side_110_1195/labels.json'),
        ('FRED0','stability_audit_20261031/RAW_FRED0_recovery4s','fred_transfer_20261029/input/0/coordinates.txt'),
        ('FRED7','stability_audit_20261031/RAW_FRED7_final4s','fred_transfer_20261029/unseen7/input/7/coordinates.txt'),
        ('FRED8','stability_audit_20261031/RAW_FRED8_fused4s','stability_audit_20261031/holdout8/input/coordinates.txt'),
        ('FRED9','stability_audit_20261031/RAW_FRED9_budget4s','stability_audit_20261031/holdout9/input/coordinates.txt')]
    for name,path,label in specs:
        rows=[json.loads(s) for s in (base/path/'predictions.jsonl').read_text().splitlines()];model=PositionNetForce();new=[]
        for r in rows:
            p,f,d=model.step(r);new.append(dict(r,physical=p,force=f,position_net_force=d))
        (out/(name+'.jsonl')).write_text('\n'.join(json.dumps(r,allow_nan=False) for r in new)+'\n','utf-8')
        if name.startswith('FRED'):
            labels=parse((base/label).read_text('utf-8-sig'))
            def target(t):
                b=reference(labels,t);return None if b is None else (b[:2]+b[2:])/2
        else:
            labels=sorted(json.loads((base/label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
            def target(t):return reference_at(labels,t)
        result[name]=dict(matched=matched(rows,new,target),applied=sum(r['position_net_force']['applied'] for r in new))
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps({k:dict(old=v['matched']['old_matched_error'],new=v['matched']['new_matched_error'],applied=v['applied']) for k,v in result.items()},indent=2))


if __name__=='__main__':main()
