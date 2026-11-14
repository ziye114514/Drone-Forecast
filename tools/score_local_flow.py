"""Full cumulative-v1 rollout with current qualified velocity observations."""
from pathlib import Path
from collections import Counter
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_scale_rollout import ScaleRolloutMotion,scale_predictions
from live_local_flow import velocity_coefficient
from summarize_live_pilot import reference_at,stats,read_rows
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
OUT=args.output_dir.resolve() if args.output_dir else WORK/'output/local_displacement_20261026';assert not (OUT/'comparison.json').exists()
NAMES=['mainline_v1','klt','sparse'];results={}

def states_at(run,case):
    inputs=read_rows(run/'tracking.jsonl');observations={round(r['t'],6):r for r in json.loads((OUT/(case+'.json')).read_text('utf-8'))}
    model=ScaleRolloutMotion(25.);states={};counts={n:Counter() for n in NAMES[1:]};details=[]
    for row in inputs:
        t=row['t'];key=round(t,6);observation=observations[key];assert observation['frame_sha256']==row['detector_frame_sha256']
        c,valid,_=model.step(t,row['body']);assert c==row['coefficient'] and valid==row['valid']
        coefficients=dict(mainline_v1=c);audit={}
        for n in NAMES[1:]:
            coefficients[n],audit[n]=velocity_coefficient(c,model.kf.P[2:4,2:4] if c else None,observation[n],t)
            if c is not None:
                assert coefficients[n]['position']==c['position'] and coefficients[n]['acceleration']==c['acceleration']
            counts[n][audit[n]['reason']]+=1
        states[key]=coefficients;details.append(dict(t=t,velocity_audits=audit))
    (OUT/(case+'_fusion.json')).write_text(json.dumps(details,indent=2),encoding='utf-8')
    return states,{n:dict(c) for n,c in counts.items()}

for case in ['side','train2']:
    run=WORK/'output/single_model_optimized_RAW_20261026'/case;states,counts=states_at(run,case)
    pubs=read_rows(run/'predictions.jsonl');label=WORK/'output'/('dense_labels_side_110_1195' if case=='side' else 'dense_labels_train2_40_46')/'labels.json'
    records=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda r:r['t']);tables={};samples=[]
    for h in [.1,.2,.4]:
        errors={n:[] for n in NAMES}
        for p in pubs:
            if not p['valid']:continue
            truth=reference_at(records,p['t']+h)
            if truth is None:continue
            points={};e={}
            for n in NAMES:
                c=states[round(p['observation_t'],6)][n];point=scale_predictions(c,p['observation_age_ms']/1000)[0][str(h)]
                if n=='mainline_v1':assert point==p['predictions']['live_motion'][str(h)]
                e[n]=float(np.linalg.norm(np.asarray(point)-truth));errors[n].append(e[n]);points[n]=point
            samples.append(dict(publication_t=p['t'],horizon=h,truth=truth.tolist(),points=points,errors=e))
        tables[str(h)]={n:stats(e) for n,e in errors.items()}
    results[case]=dict(scores=tables,velocity_updates=counts,validity_unchanged=True,samples=samples)
run=WORK/'output/heldout_motion_RAW_20261026';states,counts=states_at(run,'test3')
pubs={round(p['t'],9):p for p in read_rows(run/'predictions.jsonl')};old=json.loads((run/'scores.json').read_text('utf-8'))
errors={n:[] for n in NAMES};samples=[]
for s in old['samples']:
    p=pubs[round(s['publication_t'],9)];assert p['valid'];truth=np.asarray(s['truth']);H=s['target_t']-p['observation_t'];points={};e={}
    for n in NAMES:
        points[n]=scale_predictions(states[round(p['observation_t'],6)][n],H-.1)[0]['0.1']
        if n=='mainline_v1':assert points[n]==s['predictions']['merged_scale']
        e[n]=float(np.linalg.norm(np.asarray(points[n])-truth));errors[n].append(e[n])
    samples.append(dict(target_t=s['target_t'],publication_t=p['t'],horizon=s['horizon_s'],truth=truth.tolist(),points=points,errors=e))
results['test3']=dict(scores={'variable_0317_0451':{n:stats(e) for n,e in errors.items()}},velocity_updates=counts,
    fixed04s=False,pooled_into_development=False,samples=samples,validity_unchanged=True)
macro={n:sum(results[c]['scores']['0.4'][n]['mean'] for c in ['side','train2'])/2 for n in NAMES}
report=dict(results=results,development_macro04s=macro,complete_stack=True,baseline_replay_exact=True,
    current_state_and_validity_unchanged=True,velocity_only_rollout_correction=True,no_parameters_swept=True,
    globally_blind=False,scope='Frozen accepted observations/publications from actual v1. Offline optical-flow observer; not production RAW timing.')
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(development_macro04s=macro,results={c:{k:v for k,v in r.items() if k!='samples'} for c,r in results.items()})),flush=True)
