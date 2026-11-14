"""One fixed mechanism on frozen observations; labels used only after inference."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from live_innovation_motion import InnovationMotion,DESIGN
from summarize_live_pilot import reference_at,stats,read_rows

OUT=ROOT.parent/'output/live_innovation_covariance_20261025';OUT.mkdir(exist_ok=False)
design=dict(**DESIGN,scope='One fixed candidate, no RAW/NN rerun. Known development recordings only.',
    sources=['https://arxiv.org/abs/2112.12082','https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html'],
    not_a_reproduction_of_referenced_paper=True,
    provisional_pass_before_new_RAW='Both independent-clock side/train2 .4s means improve; neither .1s mean worsens by more than1px. No default promotion from this replay alone.',
    module_sha256=hashlib.sha256((ROOT/'live_innovation_motion.py').read_bytes()).hexdigest())
(OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
results={}
for cohort,folder in [('old_live','live_forecast_final_v2_20261025'),('independent_clock','live_body_association_v2_20261025')]:
    results[cohort]={}
    for name in ['clear','side','train2']:
        run=ROOT.parent/'output'/folder/name;rr=read_rows(run/'tracking.jsonl');pp=read_rows(run/'predictions.jsonl')
        models=dict(baseline=LiveMotion(25.),disabled=InnovationMotion(25.,enabled=False),adaptive=InnovationMotion(25.))
        states={key:{} for key in models};scales=[];resets=0;baseline_delta=0.;control_delta=0.
        traces=[]
        for row in rr:
            tick={}
            for key,model in models.items():
                c,v,d=model.step(row['t'],row['body']);tick[key]=dict(c=c,valid=v,seen=model.seen)
                states[key][round(row['t'],6)]=tick[key]
                if key=='baseline' and c is not None:
                    baseline_delta=max(baseline_delta,max(abs(np.asarray(c[f])-row['coefficient'][f]).max() for f in ['position','velocity','acceleration']))
                if key=='adaptive':
                    scales.append(d['Q_scale_used']);resets=max(resets,d['maneuver_Q_resets'])
                    traces.append(dict(t=row['t'],coefficient=c,valid=v,diagnostic=d))
            assert tick['baseline']['valid']==tick['adaptive']['valid']==tick['disabled']['valid']
            if tick['baseline']['c'] is not None:
                control_delta=max(control_delta,max(abs(np.asarray(tick['baseline']['c'][f])-tick['disabled']['c'][f]).max() for f in ['position','velocity','acceleration']))
        assert baseline_delta<1e-10 and control_delta<1e-10,(baseline_delta,control_delta)
        tables={}
        if name in ['side','train2']:
            label=ROOT.parent/'output'/('dense_labels_side_110_1195' if name=='side' else 'dense_labels_train2_40_46')/'labels.json'
            records=sorted(json.loads(label.read_text(encoding='utf-8'))['records'].values(),key=lambda q:q['t'])
            for h in [.1,.2,.4]:
                errors={key:[] for key in ['baseline','adaptive']};missing=0
                for pub in pp:
                    if not pub['valid']:continue
                    reference=reference_at(records,pub['t']+h)
                    if reference is None:missing+=1;continue
                    for key in errors:
                        state=states[key][round(pub['observation_t'],6)]
                        point=predictions(state['c'],pub['observation_age_ms']/1000.)[0][str(h)]
                        errors[key].append(float(np.linalg.norm(np.asarray(point)-reference)))
                tables[str(h)]=dict(metrics={key:stats(v) for key,v in errors.items()},missing_reference=missing)
        (OUT/(cohort+'_'+name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in traces)+'\n',encoding='utf-8')
        results[cohort][name]=dict(observations=len(rr),baseline_replay_max_difference=baseline_delta,
            disabled_control_max_difference=control_delta,scale_median=float(np.median(scales)),
            scale_min=min(scales),at_floor=sum(s<=.01000001 for s in scales),max_resets_within_one_identity=resets,scores=tables)
primary=results['independent_clock']
passes=all(primary[n]['scores']['0.4']['metrics']['adaptive']['mean']<primary[n]['scores']['0.4']['metrics']['baseline']['mean']
    and primary[n]['scores']['0.1']['metrics']['adaptive']['mean']<=primary[n]['scores']['0.1']['metrics']['baseline']['mean']+1. for n in ['side','train2'])
report=dict(design=design,results=results,provisional_replay_gate_passed=passes,default_promoted=False,
    RAW_or_GPU_replayed=False,labels_used_for_adaptation=False,validity_unchanged=True,
    reference_scope='Same publication-valid origins and targets, adjacent present manual centers interpolated <=150ms; no new labels or blind generalization.')
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
compact={cohort:{name:dict(Q_scale_median=r['scale_median'],scores={h:q['metrics'] for h,q in r['scores'].items()}) for name,r in cases.items()} for cohort,cases in results.items()}
print(json.dumps(dict(provisional_gate=passes,results=compact)),flush=True)
