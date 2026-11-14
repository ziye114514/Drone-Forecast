"""A fixed conditional scale mechanism, with matched frozen forecast targets."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from live_scale_rollout import ScaleRolloutMotion,scale_predictions,estimate_scale,DESIGN
from summarize_live_pilot import reference_at,stats,read_rows

OUT=ROOT.parent/'output/live_scale_rollout_20261026';OUT.mkdir(exist_ok=False)
design=dict(**DESIGN,kind='Own conditional projective correction, not a paper reproduction or novelty claim.',
    provisional_gate='Both primary independent-clock .4s means decrease; neither .1s mean worsens by more than1px.',
    source_sha256=hashlib.sha256((ROOT/'live_scale_rollout.py').read_bytes()).hexdigest(),
    related_source='https://openaccess.thecvf.com/content_cvpr_2018/html/Yagi_Future_Person_Localization_CVPR_2018_paper.html',
    estimator_source='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html')
(OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')

# Mechanism algebra with ideal scale, not photometric simulation or RAW truth.
tt=np.arange(0,.360001,.04);now=float(tt[-1]);z=lambda t:5.-t
sizes=[(float(t),np.array([500./z(t),250./z(t)])) for t in tt]
k,info=estimate_scale(sizes,now);assert k is not None and abs(k-1/z(now))<1e-10
constant=[(float(t),np.array([100.,50.])) for t in tt]
assert estimate_scale(constant,now)[0] is None
shape=[(float(t),np.array([100.*(1+t),50./(1+t)])) for t in tt]
assert estimate_scale(shape,now)[0] is None
p=lambda t:np.array([100*(1+3*t)/z(t),60*(2-t)/z(t)])
C=np.array([1600.,-180.]);v=C/z(now)**2;a=2*C/z(now)**3
c=dict(position=p(now).tolist(),velocity=v.tolist(),acceleration=a.tolist(),conditional_scale_rate=k)
for h in [.1,.2,.4]:
    target=p(now)+max(.5,1-.9*h*h)*(p(now+h)-p(now))
    result=scale_predictions(c)[0][str(h)]
    assert np.max(abs(np.asarray(result)-target))<1e-9
controls=dict(ideal_rational_projection_matches_damped_analytic_target=True,
    constant_scale_rejected=True,opposite_width_height_change_rejected=True,
    scope='Ideal analytic scale/trajectory only, not event or metric-depth truth')
(OUT/'controls.json').write_text(json.dumps(controls,indent=2),encoding='utf-8')

results={}
for cohort,folder in [('old_live','live_forecast_final_v2_20261025'),('independent_clock','live_body_association_v2_20261025')]:
    results[cohort]={}
    for name in ['clear','side','train2']:
        run=ROOT.parent/'output'/folder/name;rr=read_rows(run/'tracking.jsonl');pp=read_rows(run/'predictions.jsonl')
        models=dict(baseline=LiveMotion(25.),scale=ScaleRolloutMotion(25.));states={k:{} for k in models}
        traces=[];qualified=0;delta=0.
        for row in rr:
            current={}
            for key,model in models.items():
                c,valid,d=model.step(row['t'],row['body']);current[key]=(c,valid)
                states[key][round(row['t'],6)]=dict(c=c,valid=valid)
                if key=='scale':
                    qualified+=c is not None and c.get('conditional_scale_rate') is not None
                    traces.append(dict(t=row['t'],coefficient=c,valid=valid,diagnostic=d))
                elif c is not None:
                    delta=max(delta,max(abs(np.asarray(c[f])-row['coefficient'][f]).max() for f in ['position','velocity','acceleration']))
            a,b=current['baseline'],current['scale'];assert a[1]==b[1]
            if a[0] is not None:
                for f in ['position','velocity','acceleration']:assert a[0][f]==b[0][f]
        assert delta<1e-10
        tables={}
        if name in ['side','train2']:
            label=ROOT.parent/'output'/('dense_labels_side_110_1195' if name=='side' else 'dense_labels_train2_40_46')/'labels.json'
            records=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda q:q['t'])
            for h in [.1,.2,.4]:
                errors={k:[] for k in models};changed=0;missing=0
                for pub in pp:
                    if not pub['valid']:continue
                    truth=reference_at(records,pub['t']+h)
                    if truth is None:missing+=1;continue
                    points={}
                    for key in models:
                        c=states[key][round(pub['observation_t'],6)]['c']
                        points[key]=(predictions if key=='baseline' else scale_predictions)(c,pub['observation_age_ms']/1000.)[0][str(h)]
                        errors[key].append(float(np.linalg.norm(np.asarray(points[key])-truth)))
                    changed+=points['baseline']!=points['scale']
                tables[str(h)]=dict(metrics={k:stats(e) for k,e in errors.items()},changed_predictions=changed,missing_reference=missing)
        (OUT/(cohort+'_'+name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in traces)+'\n',encoding='utf-8')
        results[cohort][name]=dict(scale_qualified_ticks=qualified,baseline_max_difference=delta,scores=tables)
primary=results['independent_clock']
passed=all(primary[n]['scores']['0.4']['metrics']['scale']['mean']<primary[n]['scores']['0.4']['metrics']['baseline']['mean'] and
    primary[n]['scores']['0.1']['metrics']['scale']['mean']<=primary[n]['scores']['0.1']['metrics']['baseline']['mean']+1. for n in ['side','train2'])
report=dict(design=design,controls=controls,results=results,provisional_gate_passed=passed,default_promoted=False,
    RAW_or_GPU_replayed=False,reference='Same frozen valid publication targets; known manual-center interpolation<=150ms, no independent depth/pose truth or blind generalization.')
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(provisional_gate=passed,results=results)),flush=True)
