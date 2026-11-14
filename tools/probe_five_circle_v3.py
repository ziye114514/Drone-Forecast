"""Fixed current-observation replay and separated future-box scoring."""
from pathlib import Path
from collections import Counter
import copy,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from structure_v3 import StructureV3
from summarize_live_pilot import reference_at,stats


def process(run,out,labels=None,cached=False):
    rows=[];transform=StructureV3(3);counts={str(i):Counter() for i in (1,2,3)};cost=[]
    with (run/'predictions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);m=r.get('rotor_branch')
            if m is None:continue
            start=time.perf_counter()
            if 'five_circle_v3' not in m:
                before=copy.deepcopy(r['predictions']);m=transform.step(r['t'],r['body'],
                    r['coefficients'].get('optimized_geometry'),r['predictions'].get('optimized_geometry',{}),m)
                assert before==r['predictions']
                r['predictions'].update(m['extra_predictions'])
            cost.append((time.perf_counter()-start)*1000)
            for i,g in m['five_circle_v3']['models'].items():counts[i][g['mode']]+=1
            rows.append(dict(t=r['t'],valid=r['valid'],body=r['body'],
                model=m['five_circle_v3'],predictions=r['predictions']))
    out.mkdir(parents=True,exist_ok=False)
    (out/'model_states.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    # Truth is first read after every inference state/prediction is frozen.
    scores={};names=['optimized_geometry','structure1_disturbance','structure2_disturbance',
                     'structure3_disturbance','structure3_assumed_thrust_disturbance']
    if any('structure3_delayed_disturbance' in r['predictions'] for r in rows):
        names+=['structure3_delayed_disturbance','structure3_delayed_no_input']
    if labels is not None:
        records=sorted(json.loads(labels.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        errors={n:[] for n in names};samples=[]
        for r in rows:
            if not r['valid'] or not all(n in r['predictions'] for n in names):continue
            truth=reference_at(records,r['t']+.4)
            if truth is None:continue
            e={n:float(np.linalg.norm(np.asarray(r['predictions'][n]['0.4'])-truth)) for n in names}
            for n in names:errors[n].append(e[n])
            samples.append(dict(t=r['t'],target_t=r['t']+.4,errors=e,truth=truth.tolist()))
        scores=dict(common04s={n:stats(e) for n,e in errors.items()},samples=samples)
    summary=dict(run=str(run),ticks=len(rows),cached_observations=cached,
        actual_new_RAW=not cached,selected_mechanism=3,
        states={i:dict(v) for i,v in counts.items()},scores=scores,
        component_replay_median_ms=float(np.median(cost)),
        labels_entered_inference=False,baseline_predictions_preserved=True,
        scope='Offline source-time0.4s comparisons, not an actual low-frequency wall-deadline benchmark. No rotor centre/pose/force truth.',
        ready_counts={n:sum(n in r['predictions'] for r in rows) for n in names})
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='scores'}),flush=True)
    if scores:print(json.dumps(scores['common04s']),flush=True)
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    p.add_argument('--labels');p.add_argument('--cached',action='store_true');a=p.parse_args()
    process(Path(a.run),Path(a.out),None if a.labels is None else Path(a.labels),a.cached)
