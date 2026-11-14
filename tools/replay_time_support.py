"""One confidence mechanism; same frozen inputs, exact publication targets."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from live_time_support import TimeSupportMotion
from summarize_live_pilot import reference_at,stats,read_rows

OUT=ROOT.parent/'output/live_time_support_20261025';OUT.mkdir(exist_ok=False)
results={}
for cohort,folder in [('old_live','live_forecast_final_v2_20261025'),('independent_clock','live_body_association_v2_20261025')]:
    results[cohort]={}
    for name in ['clear','side','train2']:
        run=ROOT.parent/'output'/folder/name;rr=read_rows(run/'tracking.jsonl');pp=read_rows(run/'predictions.jsonl')
        models=dict(count=LiveMotion(25.),time=TimeSupportMotion(25.));states={key:{} for key in models}
        counts={key:0 for key in models};changed=0;max_baseline_delta=0.
        for row in rr:
            saved={}
            for key,model in models.items():
                c,v,d=model.step(row['t'],row['body']);saved[key]=c
                states[key][round(row['t'],6)]=dict(c=c,valid=v,seen=model.seen,diagnostic=d)
                counts[key]+=d['acceleration_eligible']
                if key=='count' and c is not None:
                    for field in ['position','velocity','acceleration']:
                        max_baseline_delta=max(max_baseline_delta,float(np.max(abs(np.asarray(c[field])-row['coefficient'][field]))))
            assert states['count'][round(row['t'],6)]['valid']==states['time'][round(row['t'],6)]['valid']
            if saved['count'] is not None:
                for field in ['position','velocity']:assert saved['count'][field]==saved['time'][field]
                changed+=saved['count']['acceleration']!=saved['time']['acceleration']
        assert max_baseline_delta<1e-10
        tables={};records=None
        if name in ['side','train2']:
            label=ROOT.parent/'output'/('dense_labels_side_110_1195' if name=='side' else 'dense_labels_train2_40_46')/'labels.json'
            records=sorted(json.loads(label.read_text(encoding='utf-8'))['records'].values(),key=lambda q:q['t'])
        if records is not None:
            for h in [.1,.2,.4]:
                errors={key:[] for key in models};detail=[];missing=0
                for pub in pp:
                    if not pub['valid']:continue
                    reference=reference_at(records,pub['t']+h)
                    if reference is None:missing+=1;continue
                    tick=round(pub['observation_t'],6);pair={}
                    for key in models:
                        state=states[key][tick]
                        predicted=predictions(state['c'],pub['observation_age_ms']/1000.)[0][str(h)]
                        error=float(np.linalg.norm(np.asarray(predicted)-reference));errors[key].append(error);pair[key]=error
                    detail.append(dict(publication_t=pub['t'],observation_t=pub['observation_t'],errors=pair))
                tables[str(h)]=dict(metrics={key:stats(v) for key,v in errors.items()},missing_reference=missing,detail=detail)
        results[cohort][name]=dict(observations=len(rr),acceleration_eligible=counts,changed_acceleration=changed,
            baseline_replay_max_difference=max_baseline_delta,scores=tables)
result=dict(scope='Frozen observation replay only, no new RAW/NN. Same actual publications and interpolated adjacent present manual boxes<=150ms.',
    mechanism='Confidence integrates observed support over past actual timestamps; same0.9/.35s/gap/fit/noise gates, no tuning.',
    causal=True,position_velocity_and_validity_unchanged=True,default_promoted=False,results=results)
(OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({cohort:{name:dict(eligible=value['acceleration_eligible'],changed=value['changed_acceleration'],scores={h:q['metrics'] for h,q in value['scores'].items()}) for name,value in cases.items()} for cohort,cases in results.items()}),flush=True)
