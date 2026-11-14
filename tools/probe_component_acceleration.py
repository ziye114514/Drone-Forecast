"""Fixed contiguous-history repair across all cached domains, no retraining."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from probe_same_window_motion import replay
from compare_final_pipeline import reference_at,stats
from score_fred_adaptation import matched
from prepare_fred_transfer import parse
from score_fred_transfer import reference
OPTIONS=dict(curved=True,evidence_gain=True,memory=True,endpoint=True,memory_gaps=True,
    weak_support=True,position_history_rebase='stable-axis',contained_reanchor_motion=True)


def main(out):
    base=ROOT.parent/'output';out.mkdir(exist_ok=False)
    cases=[('train2','target25_continuity_20261031/RAW_other_center6s','dense_labels_train2_40_46'),
        ('side','reacquisition_motion_20261029/RAW_side_default8s','dense_labels_side_110_1195'),
        ('FRED0','target25_continuity_20261031/RAW_fred0_center4s','fred_transfer_20261029/input/0/coordinates.txt'),
        ('FRED1','fred_domain_adaptation_20261029/RAW_1_verified','fred_transfer_20261029/input/1/coordinates.txt'),
        ('FRED2','fred_transfer_20261029/RAW_2_candidate','fred_transfer_20261029/input/2/coordinates.txt'),
        ('FRED7','fred_domain_adaptation_20261029/RAW_7_baseline','fred_transfer_20261029/unseen7/input/coordinates.txt')]
    result={}
    for name,path,label in cases:
        if name=='FRED7':
            found=list((base/'fred_transfer_20261029/unseen7').rglob('coordinates.txt'))
            if not found:continue
            lp=found[0]
        else:lp=base/label
        rows=[json.loads(x) for x in (base/path/'predictions.jsonl').read_text().splitlines()]
        options=dict(OPTIONS)
        if name=='FRED2':
            # This earlier frozen run predates weak-semantic motion support.
            # Reproduce its original configuration, not a modern baseline.
            options.update(weak_support=False,position_history_rebase=False,contained_reanchor_motion=False)
        control,_=replay(rows,False,**options)
        difference=max((float(np.max(np.abs(np.asarray(a['physical'][h])-b['physical'][h]))) for a,b in zip(rows,control) for h in a['physical']),default=0.)
        assert difference<1e-7,(name,difference)
        candidate,_=replay(rows,False,component_acceleration=True,**options)
        assert all(a['valid']==b['valid'] and a['body']==b['body'] and a['force']['input_u']==b['force']['input_u'] for a,b in zip(rows,candidate))
        (out/(name+'.jsonl')).write_text('\n'.join(json.dumps(x,allow_nan=False) for x in candidate)+'\n','utf-8')
        if name.startswith('FRED'):
            labels=parse(lp.read_text('utf-8-sig'))
            def target(t):
                b=reference(labels,t);return None if b is None else (b[:2]+b[2:])/2
        else:
            labels=sorted(json.loads((lp/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
            def target(t):return reference_at(labels,t)
        result[name]=dict(replay_reproduced=difference,matched=matched(control,candidate,target),
            history_restarts=[dict(t=r['t'],diagnostic=r['image_motion']['acceleration'].get('component_recovery')) for r in candidate if r['image_motion']['acceleration'].get('reason')=='position_qualified_acceleration_components'])
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:dict(old=v['matched']['old_matched_error'],new=v['matched']['new_matched_error'],restarts=len(v['history_restarts'])) for k,v in result.items()},indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
