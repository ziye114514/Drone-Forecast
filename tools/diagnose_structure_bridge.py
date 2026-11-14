"""Same-input mechanism replay, or actual25Hz/5Hz source-clock bridge scoring."""
from pathlib import Path
from collections import Counter
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from structure_v3 import StructureV3
from summarize_live_pilot import reference_at,stats


def run(source,output,replay=False,labels=None):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    observer=StructureV3(3) if replay else None
    states=[];modes=Counter();changes=Counter();inputs=0;center_updates=0;recent=0;slots=0;tracking=0
    for line in (source/'predictions.jsonl').read_text('utf-8').splitlines():
        row=json.loads(line);measurement=row['rotor_branch'];old=measurement['five_circle_v3']['selected']
        if replay:
            measurement=observer.step(row['t'],row['body'],row['coefficients'].get('optimized_geometry'),
                row['predictions'].get('optimized_geometry',{}),measurement)
        m=measurement['five_circle_v3'];g=m['selected'];modes[g['mode']]+=1;recent+=int(g.get('recent_model_reused',False))
        if replay:changes[old['mode']+' -> '+g['mode']]+=1
        inputs+=m['assumed_thrust_motion']['diagnostic']['input_available']
        pair=m['live_position_pair']
        if pair['updated']:
            tracking+=1;center_updates+=int(pair['geometry_used']);slots+=int(pair['published_source_slot'])
            if not replay:assert pair['full_v7_image_features'] and pair['frame_sha256']==row['frame_sha256']
        states.append(dict(t=row['t'],body=row['body'],model=g,pair=pair,
            input_source=m['input_proxy_source'],activity=m['object_periodicity'],
            proxy_input_available=m['assumed_thrust_motion']['diagnostic']['input_available'],
            delayed=m['delayed_disturbance'],delayed_control=m['delayed_no_input']))
    #Freeze all states BEFORE any manual visibility or future-box scoring.
    (out/'states.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in states),'utf-8')
    review=WORK/'output/stage107_center_bias_filter_20261022/visibility_review'
    times=json.loads((review/'review_protocol107.json').read_text('utf-8'))['records']
    human=json.loads((review/'human_visibility107.json').read_text('utf-8'))['labels'];anchors=[]
    for anchor in times:
        found=[r for r in states if abs(r['t']-anchor['t'])<1e-7]
        if not found:continue
        r=found[0];anchors.append(dict(id=anchor['id'],t=r['t'],human=human[anchor['id']],
            mode=r['model']['mode'],accepted_hubs=r['model']['observed_count'],
            current_regions=r['model']['current_region_count'],input=r['proxy_input_available'],
            current_geometric_center=r['model']['current_geometric_center']))
    scores={};samples=[]
    if labels is not None and not replay:
        records=sorted(json.loads(Path(labels).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        errors={n:[] for n in ['v7_same_observations','v7_structure']}
        for row in states:
            pair=row['pair']
            if not pair['updated'] or not pair['published_source_slot'] or not all(pair['valid'].values()):continue
            truth=reference_at(records,row['t']+.4)
            if truth is None:continue
            e={n:float(np.linalg.norm(np.asarray(pair['predictions'][n]['0.4'])-truth)) for n in errors}
            for n in errors:errors[n].append(e[n])
            samples.append(dict(t=row['t'],target_t=row['t']+.4,truth=truth.tolist(),errors=e,
                points=pair['predictions'],center_applied=pair['geometry_used']))
        scores={n:stats(v) for n,v in errors.items()}
    result=dict(source=str(source),cached_observation_replay=replay,actual_new_RAW=not replay,
        ticks=len(states),modes=dict(modes),old_to_new_modes=dict(changes),recent_model_reuses=recent,
        input_available_ticks=inputs,position_updates=tracking,geometry_position_updates=center_updates,
        source_publication_slots=slots,anchors=anchors,scores04s=scores,samples=samples,
        labels_entered_inference=False,independent_rotor_or_force_truth=False,
        source_clock_only=True,not_a_wall_clock_realtime_benchmark=True,
        parameters_swept=False)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--replay',action='store_true');p.add_argument('--labels');a=p.parse_args()
    run(a.source,a.output,a.replay,a.labels)
