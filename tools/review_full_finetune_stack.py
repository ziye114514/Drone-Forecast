"""Compare an existing complete fine-tune with the cumulative v1 stack."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_scale_rollout import ScaleRolloutMotion,scale_predictions
from summarize_live_pilot import reference_at,stats,read_rows
OUT=WORK/'output/accumulated_v2_review_20261026'
assert not (OUT/'comparison.json').exists()
policy=json.loads((OUT/'design.json').read_text('utf-8'))['policy']
results={};all_reasons=[]

def run_states(inputs,fine):
    models=dict(v1=ScaleRolloutMotion(25.),v2=ScaleRolloutMotion(25.));states={name:{} for name in models}
    for row in inputs:
        key=round(row['t'],6);proposed=fine[key]
        assert proposed['frame_sha256']==row['detector_frame_sha256']
        for name,model in models.items():
            body=row['body'] if name=='v1' else proposed['body']
            c,valid,d=model.step(row['t'],body)
            if name=='v1':assert c==row['coefficient'] and valid==row['valid']
            states[name][key]=dict(c=c,valid=valid,last_seen=model.seen)
    return states

def usable(state,pub):
    return bool(state['valid'] and state['last_seen'] is not None and pub['t']-state['last_seen']<=.16000001 and pub['observation_age_ms']<=160.00001)

def guards(case,tables,lost):
    reasons=[]
    if lost:reasons.append(case+':lost_valid_publications='+str(lost))
    for horizon,scores in tables.items():
        a,b=scores['v1'],scores['v2']
        if not a['n'] or a['n']!=b['n']:reasons.append(case+':unmatched_targets');continue
        for field,rule in [('mean','mean_regression_allowance'),('p95','p95_regression_allowance')]:
            limits=policy['accuracy'][rule];allow=max(limits['absolute_px'],limits['relative_fraction']*a[field])
            if b[field]>a[field]+allow:reasons.append(f'{case}:{horizon}:{field}_regression={b[field]-a[field]:.3f}>allowance{allow:.3f}')
    return reasons

for case,name in [('side','side_v2'),('train2','train2')]:
    run=WORK/'output/single_model_optimized_RAW_20261026'/case
    inputs=read_rows(run/'tracking.jsonl');pubs=read_rows(run/'predictions.jsonl')
    cached=json.loads((WORK/'output/roi_augmented_detector_20261026/forecast_probe'/(name+'.json')).read_text('utf-8'))['rows']
    fine={round(r['t'],6):dict(body=r['adapted_body'],frame_sha256=r['frame_sha256'],source='existing_fine_probe') for r in cached}
    missing=run/'fine_missing_reference/observations.json'
    if missing.exists():
        for r in json.loads(missing.read_text('utf-8')):fine[round(r['t'],6)]=r
    states=run_states(inputs,fine)
    label=WORK/'output'/('dense_labels_side_110_1195' if case=='side' else 'dense_labels_train2_40_46')/'labels.json'
    records=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    lost=extra=0;tables={};valid_at={}
    for p in pubs:
        if p['observation_t'] is None:continue
        current={n:states[n][round(p['observation_t'],6)] for n in states}
        validity={n:usable(s,p) for n,s in current.items()};assert validity['v1']==p['valid']
        lost+=validity['v1'] and not validity['v2'];extra+=validity['v2'] and not validity['v1']
        valid_at[p['t']]=validity
    for h in [.1,.2,.4]:
        errors={n:[] for n in states}
        for p in pubs:
            if not p['valid'] or not all(valid_at.get(p['t'],{}).values()):continue
            truth=reference_at(records,p['t']+h)
            if truth is None:continue
            for n in states:
                c=states[n][round(p['observation_t'],6)]['c']
                point=scale_predictions(c,p['observation_age_ms']/1000.)[0][str(h)]
                if n=='v1':assert point==p['predictions']['live_motion'][str(h)]
                errors[n].append(float(np.linalg.norm(np.asarray(point)-truth)))
        tables[str(h)]={n:stats(e) for n,e in errors.items()}
    reasons=guards(case,tables,lost);all_reasons.extend(reasons)
    results[case]=dict(scores=tables,lost_valid_publications=int(lost),extra_valid_publications=int(extra),guard_failures=reasons)

run=WORK/'output/heldout_motion_RAW_20261026'
inputs=read_rows(run/'tracking.jsonl');pubs={round(p['t'],9):p for p in read_rows(run/'predictions.jsonl')}
fine={round(r['t'],6):r for r in json.loads((run/'fine_reference/observations.json').read_text('utf-8'))}
states=run_states(inputs,fine)
old=json.loads((run/'scores.json').read_text('utf-8'));samples=[];errors={n:[] for n in states};lost=0
for sample in old['samples']:
    p=pubs[round(sample['publication_t'],9)];current={n:states[n][round(p['observation_t'],6)] for n in states}
    assert usable(current['v1'],p)
    if not usable(current['v2'],p):lost+=1;continue
    H=sample['target_t']-p['observation_t'];truth=np.asarray(sample['truth']);points={}
    for n in states:
        points[n]=scale_predictions(current[n]['c'],H-.1)[0]['0.1']
        if n=='v1':assert points[n]==sample['predictions']['merged_scale']
        errors[n].append(float(np.linalg.norm(np.asarray(points[n])-truth)))
    samples.append(dict(target_t=sample['target_t'],publication_t=p['t'],horizon_s=sample['horizon_s'],predictions=points))
tables={'variable_0317_0451':{n:stats(e) for n,e in errors.items()}}
reasons=guards('test3',tables,lost);all_reasons.extend(reasons)
results['test3']=dict(scores=tables,lost_targets=lost,guard_failures=reasons,samples=samples,
    fixed04s=False,pooled_into_development=False)
macro={n:sum(results[case]['scores']['0.4'][n]['mean'] for case in ['side','train2'])/2 for n in states}
accepted=bool(macro['v2']<macro['v1'] and not all_reasons)
if macro['v2']>=macro['v1']:all_reasons.append('development_macro04s_not_better')
report=dict(results=results,development_macro04s=macro,accepted_for_next_mainline=accepted,guard_failures=all_reasons,
    complete_stack_comparison=True,v1_numerical_replay_exact=True,no_retraining_or_weight_sweep=True,
    latency='Same accepted observations and v1 publication times; offline v2 detector reference. Not a new timed production comparison.',
    globally_blind=False,policy_after_user_preference=True)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
print(json.dumps({case:{k:v for k,v in r.items() if k!='samples'} for case,r in results.items()}),flush=True)
