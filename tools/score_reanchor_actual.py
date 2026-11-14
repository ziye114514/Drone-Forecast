"""Control only the reanchor gate on an already completed RAW observation stream."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from probe_same_window_motion import replay
from probe_contained_reanchor import OPTIONS
from diagnose_maneuver_horizon import SPECS
from compare_final_pipeline import reference_at
from score_fred_adaptation import matched


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--name',choices=['side','other'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=[json.loads(x) for x in (args.run/'predictions.jsonl').read_text().splitlines()]
    new,_=replay(rows,False,contained_reanchor_motion=True,**OPTIONS);old,_=replay(rows,False,**OPTIONS)
    difference=max((float(np.max(np.abs(np.asarray(a['physical'][h])-b['physical'][h])))
        for a,b in zip(rows,new) for h in a['physical']),default=0.)
    assert difference<1e-7,difference
    _,original,label=next(s for s in SPECS if s[0]==args.name)
    labels=sorted(json.loads((ROOT.parent/'output'/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    result=dict(max_reproduced_difference=difference,comparison=matched(old,new,lambda t:reference_at(labels,t)),
        accepted_times=[r['t'] for r in new if r['image_motion']['observation'].get('contained_weak_patch_motion')],
        current_boxes_validity_and_input_u_unchanged=all(a['body']==b['body'] and a['valid']==b['valid'] and
            a['force']['input_u']==b['force']['input_u'] for a,b in zip(old,new)))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='comparison'}))
    print(json.dumps({k:v for k,v in result['comparison'].items() if k!='records'}))
