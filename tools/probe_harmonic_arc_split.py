"""Bounded original-arc split controls and same three cached window masks."""
from pathlib import Path
import sys,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from harmonic_arc_split import split_original_arcs


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    mask=np.zeros((150,230),np.uint8);period=np.zeros_like(mask)
    centers=[(85,75),(130,75)]
    for c in centers:cv2.ellipse(mask,c,(35,22),0,0,360,255,-1);cv2.ellipse(period,c,(16,10),0,0,360,255,-1)
    controlled=split_original_arcs(mask,period,120.)
    no_period=split_original_arcs(mask,np.zeros_like(period),120.)
    single=np.zeros_like(mask);cv2.ellipse(single,(105,75),(60,22),0,0,360,255,-1)
    single_result=split_original_arcs(single,period,150.)
    (out/'controlled.json').write_text(json.dumps(dict(two_overlapping=controlled,no_period=no_period,
        one_ellipse_with_two_false_seeds=single_result,known_centres=centers,synthetic_only=True),indent=2),'utf-8')
    assert no_period['new_current_qualified_centres']==0
    assert single_result['new_current_qualified_centres']==0,'Two temporal seeds must not split one ellipse into two motors'
    records=[];prior=out.parent/'aligned'
    for name in ['clear','partial','side']:
        count=cv2.imread(str(prior/name/'count20_mask.png'),0);period=cv2.imread(str(prior/name/'predictive_mask.png'),0)
        result=split_original_arcs(count,period);result['name']=name;records.append(result)
        (out/(name+'.json')).write_text(json.dumps(result,indent=2),'utf-8')
    summary=dict(controlled_two_children=controlled['new_current_qualified_centres'],
        records=[dict(name=r['name'],accepted_parents=r['accepted_split_parents'],new_centres=r['new_current_qualified_centres'],
            new_qualified_partial_arcs=r['new_qualified_partial_arcs'],
            attempted_parents=sum(2<=p['temporal_seed_count']<=4 for p in r['parents'])) for r in records],
        original_masks_not_modified=True,default_unchanged=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary))


if __name__=='__main__':main(sys.argv[1])
