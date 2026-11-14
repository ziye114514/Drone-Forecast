"""Two fixed spatial mechanisms against the current v4 publication cohort."""
from pathlib import Path
from collections import Counter
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_scale_rollout import ScaleRolloutMotion,scale_predictions
from live_local_flow import velocity_coefficient
from live_flow_spatial import observation
from summarize_live_pilot import reference_at,stats,read_rows
parser=argparse.ArgumentParser();parser.add_argument('--symmetric',action='store_true');args=parser.parse_args()
OBS=WORK/'output/spatial_flow_20261026'
OUT=WORK/'output/symmetric_velocity_20261026' if args.symmetric else OBS;assert not (OUT/'comparison.json').exists()
NAMES=['v4','symmetric','symmetric_spatial_cov'] if args.symmetric else ['v4','hull','spatial_cov'];results={}
cases=dict(side=WORK/'output/local_displacement_tight_20261026/actual_side',
    train2=WORK/'output/local_displacement_tight_20261026/actual_train2',test3=WORK/'output/heldout_motion_RAW_20261026')

def states_at(case,run):
    data={round(r['t'],6):r for r in json.loads((OBS/(case+'.json')).read_text('utf-8'))}
    model=ScaleRolloutMotion(25.);states={};counts={n:Counter() for n in NAMES};max_delta=0.;exact=0
    for row in read_rows(run/'tracking.jsonl'):
        t=row['t'];key=round(t,6);sample=data[key];assert sample['frame_sha256']==row['detector_frame_sha256']
        c,valid,_=model.step(t,row['body']);expected=row['coefficient'] if case=='test3' else row['position_only_coefficient']
        assert c==expected and valid==row['valid'];states[key]={}
        variants=[('v4',None,'minimum-volume'),('symmetric',None,'symmetric'),('symmetric_spatial_cov','spatial-covariance','symmetric')] if args.symmetric else [('v4',None,'minimum-volume'),('hull','center-hull','minimum-volume'),('spatial_cov','spatial-covariance','minimum-volume')]
        for name,policy,strategy in variants:
            obs=sample['observation'] if policy is None else observation(sample['observation'],policy)
            revised,audit=velocity_coefficient(c,model.kf.P[2:4,2:4] if c else None,obs,t,strategy=strategy)
            states[key][name]=revised;counts[name][audit['reason']]+=1
        if case!='test3' and c is not None:
            original=row['coefficient'];current=states[key]['v4']
            delta=float(np.linalg.norm(np.asarray(current['velocity'])-original['velocity']));max_delta=max(max_delta,delta)
            exact+=current==original
    return states,{n:dict(c) for n,c in counts.items()},dict(max_current_v4_velocity_delta_px_s=max_delta,exact_coefficient_ticks=exact)

for case in ['side','train2']:
    run=cases[case];states,counts,match=states_at(case,run);pubs=read_rows(run/'predictions.jsonl');samples=[];tables={};forecast_delta=0.
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
                point=scale_predictions(states[round(p['observation_t'],6)][n],p['observation_age_ms']/1000.)[0][str(h)]
                if n=='v4':forecast_delta=max(forecast_delta,float(np.linalg.norm(np.asarray(point)-p['predictions']['live_motion'][str(h)])))
                points[n]=point;e[n]=float(np.linalg.norm(np.asarray(point)-truth));errors[n].append(e[n])
            samples.append(dict(publication_t=p['t'],target_t=p['t']+h,horizon=h,truth=truth.tolist(),points=points,errors=e))
        tables[str(h)]={n:stats(e) for n,e in errors.items()}
    assert forecast_delta<.001,forecast_delta
    results[case]=dict(scores=tables,velocity_updates=counts,validity_unchanged=True,control_match=match,max_v4_forecast_delta_px=forecast_delta,samples=samples)

case='test3';run=cases[case];states,counts,match=states_at(case,run)
pubs={round(p['t'],9):p for p in read_rows(run/'predictions.jsonl')};old=json.loads((run/'scores.json').read_text('utf-8'))
errors={n:[] for n in NAMES};samples=[]
for sample in old['samples']:
    p=pubs[round(sample['publication_t'],9)];assert p['valid'];truth=np.asarray(sample['truth']);H=sample['target_t']-p['observation_t'];points={};e={}
    for n in NAMES:
        point=scale_predictions(states[round(p['observation_t'],6)][n],H-.1)[0]['0.1']
        points[n]=point;e[n]=float(np.linalg.norm(np.asarray(point)-truth));errors[n].append(e[n])
    samples.append(dict(publication_t=p['t'],target_t=sample['target_t'],horizon=sample['horizon_s'],truth=truth.tolist(),points=points,errors=e))
results['test3']=dict(scores={'variable_0317_0451':{n:stats(e) for n,e in errors.items()}},velocity_updates=counts,
    fixed04s=False,pooled_into_development=False,samples=samples,validity_unchanged=True)
macro={n:sum(results[c]['scores']['0.4'][n]['mean'] for c in ['side','train2'])/2 for n in NAMES}
report=dict(results=results,development_macro04s=macro,complete_stack=True,parameters_swept=False,
    scope='Same actual v4 accepted observations and publication times; two offline spatial qualification mechanisms. Not new production latency.',globally_blind=False)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(development_macro04s=macro,results={c:{k:v for k,v in r.items() if k!='samples'} for c,r in results.items()})),flush=True)
