"""Two causal acceleration mechanisms on the current fixed v5 observation set."""
from pathlib import Path
from collections import Counter
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_scale_rollout import ScaleRolloutMotion,scale_predictions
from live_local_flow import velocity_coefficient
from live_flow_acceleration import FlowAcceleration
from summarize_live_pilot import reference_at,stats,read_rows
OUT=WORK/'output/flow_acceleration_20261026';assert not (OUT/'comparison.json').exists()
NAMES=['v5','flow_derivative','position_consistent'];results={}
cases=dict(side=WORK/'output/local_displacement_tight_20261026/actual_side',train2=WORK/'output/local_displacement_tight_20261026/actual_train2',test3=WORK/'output/heldout_motion_RAW_20261026')

def states_at(case,run):
    samples={round(r['t'],6):r for r in json.loads((WORK/'output/spatial_flow_20261026'/(case+'.json')).read_text('utf-8'))}
    model=ScaleRolloutMotion(25.);accels={n:FlowAcceleration() for n in NAMES[1:]};states={};counts={n:Counter() for n in accels};details=[];costs=[]
    for row in read_rows(run/'tracking.jsonl'):
        t=row['t'];key=round(t,6);sample=samples[key];assert sample['frame_sha256']==row['detector_frame_sha256']
        c,valid,_=model.step(t,row['body']);expected=row['coefficient'] if case=='test3' else row['position_only_coefficient'];assert c==expected and valid==row['valid']
        observation=sample['observation'];v5,_=velocity_coefficient(c,model.kf.P[2:4,2:4] if c else None,observation,t)
        if case!='test3':assert v5==row['coefficient']
        states[key]=dict(v5=v5);audit={}
        for n,policy in [('flow_derivative','velocity-only'),('position_consistent','position-consistent')]:
            start=time.perf_counter();revised,diagnostic=accels[n].step(t,v5,row['body'],observation,policy);costs.append((time.perf_counter()-start)*1000)
            if v5 is not None:assert revised['position']==v5['position'] and revised['velocity']==v5['velocity']
            states[key][n]=revised;counts[n][diagnostic['reason']]+=1;audit[n]=diagnostic
        details.append(dict(t=t,audits=audit,baseline_acceleration=None if v5 is None else v5['acceleration']))
    (OUT/(case+'_diagnostics.json')).write_text(json.dumps(details,indent=2),encoding='utf-8')
    return states,{n:dict(c) for n,c in counts.items()},float(np.median(costs))

for case in ['side','train2']:
    run=cases[case];states,counts,cost=states_at(case,run);pubs=read_rows(run/'predictions.jsonl');samples=[];tables={}
    label=WORK/'output'/('dense_labels_side_110_1195' if case=='side' else 'dense_labels_train2_40_46')/'labels.json'
    records=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    for h in [.1,.2,.4]:
        errors={n:[] for n in NAMES}
        for p in pubs:
            if not p['valid']:continue
            truth=reference_at(records,p['t']+h)
            if truth is None:continue
            points={};e={}
            for n in NAMES:
                point=scale_predictions(states[round(p['observation_t'],6)][n],p['observation_age_ms']/1000)[0][str(h)]
                if n=='v5':assert point==p['predictions']['live_motion'][str(h)]
                points[n]=point;e[n]=float(np.linalg.norm(np.asarray(point)-truth));errors[n].append(e[n])
            samples.append(dict(publication_t=p['t'],target_t=p['t']+h,horizon=h,truth=truth.tolist(),points=points,errors=e))
        tables[str(h)]={n:stats(e) for n,e in errors.items()}
    results[case]=dict(scores=tables,updates=counts,median_observer_ms=cost,baseline_exact=True,current_p_v_validity_unchanged=True,samples=samples)
case='test3';run=cases[case];states,counts,cost=states_at(case,run)
pubs={round(p['t'],9):p for p in read_rows(run/'predictions.jsonl')};old=json.loads((run/'scores.json').read_text('utf-8'))
errors={n:[] for n in NAMES};samples=[]
for sample in old['samples']:
    p=pubs[round(sample['publication_t'],9)];assert p['valid'];truth=np.asarray(sample['truth']);H=sample['target_t']-p['observation_t'];points={};e={}
    for n in NAMES:
        point=scale_predictions(states[round(p['observation_t'],6)][n],H-.1)[0]['0.1']
        points[n]=point;e[n]=float(np.linalg.norm(np.asarray(point)-truth));errors[n].append(e[n])
    samples.append(dict(publication_t=p['t'],target_t=sample['target_t'],horizon=sample['horizon_s'],truth=truth.tolist(),points=points,errors=e))
results['test3']=dict(scores={'variable_0317_0451':{n:stats(e) for n,e in errors.items()}},updates=counts,median_observer_ms=cost,
    fixed04s=False,pooled_into_development=False,current_p_v_validity_unchanged=True,samples=samples)
macro={n:sum(results[c]['scores']['0.4'][n]['mean'] for c in ['side','train2'])/2 for n in NAMES}
result=dict(results=results,development_macro04s=macro,parameters_swept=False,RAW_or_GPU_started=False,globally_blind=False,
    scope='One fixed velocity-derivative fit and a position-consistency-qualified variant. Same current v5 observations/publications; no new timed RAW deployment.')
(OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(development_macro04s=macro,results={c:{k:v for k,v in r.items() if k!='samples'} for c,r in results.items()})),flush=True)
