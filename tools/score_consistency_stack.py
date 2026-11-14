"""Current cumulative stack versus a fixed paired-control training experiment."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_scale_rollout import ScaleRolloutMotion,scale_predictions
from summarize_live_pilot import reference_at,stats,read_rows
OUT=WORK/'output/flip_consistency_20261026';assert not (OUT/'comparison.json').exists()
NAMES=['mainline_v1','control','consistency'];results={}

def replay(run,case):
    inputs=read_rows(run/'tracking.jsonl');probes={round(r['t'],6):r for r in json.loads((OUT/'native_probe'/(case+'.json')).read_text('utf-8'))}
    models={n:ScaleRolloutMotion(25.) for n in NAMES};states={n:{} for n in NAMES}
    for row in inputs:
        key=round(row['t'],6);probe=probes[key];assert probe['frame_sha256']==row['detector_frame_sha256']
        for n,model in models.items():
            b=row['body'] if n=='mainline_v1' else probe[n]
            c,valid,_=model.step(row['t'],b)
            if n=='mainline_v1':assert c==row['coefficient'] and valid==row['valid']
            states[n][key]=dict(c=c,valid=valid,last_seen=model.seen)
    return states

def usable(state,pub):
    return bool(state['valid'] and state['last_seen'] is not None and pub['t']-state['last_seen']<=.16000001 and pub['observation_age_ms']<=160.00001)

for case in ['side','train2']:
    run=WORK/'output/single_model_optimized_RAW_20261026'/case
    states=replay(run,case);pubs=read_rows(run/'predictions.jsonl');tables={};samples=[]
    label=WORK/'output'/('dense_labels_side_110_1195' if case=='side' else 'dense_labels_train2_40_46')/'labels.json'
    records=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    validity={};lost={n:0 for n in NAMES};extra={n:0 for n in NAMES}
    for p in pubs:
        if p['observation_t'] is None:continue
        current={n:states[n][round(p['observation_t'],6)] for n in NAMES}
        v={n:usable(s,p) for n,s in current.items()};assert v['mainline_v1']==p['valid'];validity[p['t']]=v
        for n in NAMES:lost[n]+=v['mainline_v1'] and not v[n];extra[n]+=v[n] and not v['mainline_v1']
    for h in [.1,.2,.4]:
        errors={n:[] for n in NAMES}
        for p in pubs:
            if not p['valid'] or not all(validity.get(p['t'],{}).values()):continue
            truth=reference_at(records,p['t']+h)
            if truth is None:continue
            points={};e={}
            for n in NAMES:
                c=states[n][round(p['observation_t'],6)]['c'];point=scale_predictions(c,p['observation_age_ms']/1000)[0][str(h)]
                if n=='mainline_v1':assert point==p['predictions']['live_motion'][str(h)]
                e[n]=float(np.linalg.norm(np.asarray(point)-truth));errors[n].append(e[n]);points[n]=point
            samples.append(dict(publication_t=p['t'],horizon=h,truth=truth.tolist(),points=points,errors=e))
        tables[str(h)]={n:stats(e) for n,e in errors.items()}
    results[case]=dict(scores=tables,lost_valid_publications=lost,extra_valid_publications=extra,samples=samples)

run=WORK/'output/heldout_motion_RAW_20261026';states=replay(run,'test3')
pubs={round(p['t'],9):p for p in read_rows(run/'predictions.jsonl')}
old=json.loads((run/'scores.json').read_text('utf-8'));errors={n:[] for n in NAMES};lost={n:0 for n in NAMES};samples=[]
for sample in old['samples']:
    p=pubs[round(sample['publication_t'],9)];current={n:states[n][round(p['observation_t'],6)] for n in NAMES}
    valid={n:usable(s,p) for n,s in current.items()};assert valid['mainline_v1']
    for n in NAMES:lost[n]+=not valid[n]
    if not all(valid.values()):continue
    H=sample['target_t']-p['observation_t'];truth=np.asarray(sample['truth']);points={};e={}
    for n in NAMES:
        points[n]=scale_predictions(current[n]['c'],H-.1)[0]['0.1']
        if n=='mainline_v1':assert points[n]==sample['predictions']['merged_scale']
        e[n]=float(np.linalg.norm(np.asarray(points[n])-truth));errors[n].append(e[n])
    samples.append(dict(target_t=sample['target_t'],publication_t=p['t'],horizon=sample['horizon_s'],truth=truth.tolist(),points=points,errors=e))
results['test3']=dict(scores={'variable_0317_0451':{n:stats(e) for n,e in errors.items()}},lost_targets=lost,
    fixed04s=False,pooled_into_development=False,samples=samples)
macro={n:sum(results[c]['scores']['0.4'][n]['mean'] for c in ['side','train2'])/2 for n in NAMES}
report=dict(results=results,development_macro04s=macro,current_mainline_numerical_replay_exact=True,
    complete_stack=True,matched_control=True,globally_blind=False,parameters_swept=False,
    latency='Frozen actual v1 accepted times and publication times; offline perception replacement, not a new timed RAW production comparison.')
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(development_macro04s=macro,results={c:{k:v for k,v in r.items() if k!='samples'} for c,r in results.items()})),flush=True)
