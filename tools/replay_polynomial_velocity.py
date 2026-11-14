"""Fixed local-polynomial derivative candidate on exact frozen pub targets."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from live_polynomial_velocity import PolynomialVelocityMotion
from summarize_live_pilot import reference_at,stats,read_rows

OUT=ROOT.parent/'output/live_polynomial_velocity_20261025';OUT.mkdir(exist_ok=False)
results={}
for cohort,folder in [('old_live','live_forecast_final_v2_20261025'),('independent_clock','live_body_association_v2_20261025')]:
    results[cohort]={}
    for name in ['clear','side','train2']:
        run=ROOT.parent/'output'/folder/name;rr=read_rows(run/'tracking.jsonl');pp=read_rows(run/'predictions.jsonl')
        models=dict(KF=LiveMotion(25.),polynomial=PolynomialVelocityMotion(25.));states={key:{} for key in models}
        counts={key:0 for key in models};changed=0;baseline_delta=0.
        for row in rr:
            saved={}
            for key,model in models.items():
                c,v,d=model.step(row['t'],row['body']);saved[key]=c
                states[key][round(row['t'],6)]=dict(c=c,valid=v,seen=model.seen)
                counts[key]+=d.get('polynomial_velocity_used',False)
                if key=='KF' and c is not None:
                    for field in ['position','velocity','acceleration']:
                        baseline_delta=max(baseline_delta,float(np.max(abs(np.asarray(c[field])-row['coefficient'][field]))))
            assert states['KF'][round(row['t'],6)]['valid']==states['polynomial'][round(row['t'],6)]['valid']
            if saved['KF'] is not None:
                for field in ['position','acceleration']:assert saved['KF'][field]==saved['polynomial'][field]
                changed+=saved['KF']['velocity']!=saved['polynomial']['velocity']
        assert baseline_delta<1e-10
        tables={}
        if name in ['side','train2']:
            label=ROOT.parent/'output'/('dense_labels_side_110_1195' if name=='side' else 'dense_labels_train2_40_46')/'labels.json'
            records=sorted(json.loads(label.read_text(encoding='utf-8'))['records'].values(),key=lambda q:q['t'])
            for h in [.1,.2,.4]:
                errors={key:[] for key in models};missing=0
                for pub in pp:
                    if not pub['valid']:continue
                    reference=reference_at(records,pub['t']+h)
                    if reference is None:missing+=1;continue
                    for key in models:
                        state=states[key][round(pub['observation_t'],6)]
                        point=predictions(state['c'],pub['observation_age_ms']/1000.)[0][str(h)]
                        errors[key].append(float(np.linalg.norm(np.asarray(point)-reference)))
                tables[str(h)]=dict(metrics={key:stats(v) for key,v in errors.items()},missing_reference=missing)
        results[cohort][name]=dict(observations=len(rr),changed_velocity=changed,baseline_replay_max_difference=baseline_delta,scores=tables)
report=dict(scope='Frozen actual observations and publications, no new RAW/NN. Adjacent present manual centers interpolated <=150ms; not new labels.',
    candidate='Existing23 robust quadratic mean velocity + same shrunk acceleration extrapolation to current time. Same fit eligibility and parameters.',
    position_acceleration_KF_state_and_validity_unchanged=True,default_promoted=False,results=results)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({cohort:{name:dict(changed=value['changed_velocity'],scores={h:q['metrics'] for h,q in value['scores'].items()}) for name,value in cases.items()} for cohort,cases in results.items()}),flush=True)
