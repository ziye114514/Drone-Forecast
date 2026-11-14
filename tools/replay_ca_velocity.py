"""One CA-velocity configuration, frozen observations and identical pub targets."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from live_ca_velocity import CAVelocityMotion
from summarize_live_pilot import reference_at,stats,read_rows

OUT=ROOT.parent/'output/live_CA_velocity_20261026';OUT.mkdir(exist_ok=False)
design=dict(mechanism='Reuse existing six-state constant-acceleration filter; replace current velocity only.',
    position_acceleration_and_validity_unchanged=True,position_sigma_px=6.,nominal_tracker_hz=25.,
    jerk_sigma_px_s3=320000.,jerk_design='Match baseline discrete process velocity variance at nominal40ms',
    initial_pv_variance=10000.,initial_acceleration_variance=6400.**2,
    labels_used_for_filtering=False,parameter_sweep=False,
    provisional_gate='Both primary independent-clock .4s mean errors decrease; neither .1s mean worsens by more than1px. No promotion from frozen replay alone.',
    candidate_source_sha256=hashlib.sha256((ROOT/'live_ca_velocity.py').read_bytes()).hexdigest(),
    reused_filter_source_sha256=hashlib.sha256((ROOT/'src/event_drone_forecast/motion.py').read_bytes()).hexdigest())
(OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
results={}
for cohort,folder in [('old_live','live_forecast_final_v2_20261025'),('independent_clock','live_body_association_v2_20261025')]:
    results[cohort]={}
    for name in ['clear','side','train2']:
        run=ROOT.parent/'output'/folder/name;rr=read_rows(run/'tracking.jsonl');pp=read_rows(run/'predictions.jsonl')
        models=dict(baseline=LiveMotion(25.),CA_velocity=CAVelocityMotion(25.))
        states={k:{} for k in models};traces=[];baseline_delta=0.;changed=0
        for row in rr:
            current={}
            for key,model in models.items():
                c,v,d=model.step(row['t'],row['body']);current[key]=dict(c=c,valid=v,seen=model.seen)
                states[key][round(row['t'],6)]=current[key]
                if key=='CA_velocity':
                    traces.append(dict(t=row['t'],coefficient=c,valid=v,diagnostic=d))
                    if model.ca.P is not None:
                        assert np.isfinite(model.ca.x).all() and np.linalg.eigvalsh(model.ca.P).min()>-1e-7
                elif c is not None:
                    baseline_delta=max(baseline_delta,max(abs(np.asarray(c[f])-row['coefficient'][f]).max() for f in ['position','velocity','acceleration']))
            a,b=current['baseline'],current['CA_velocity'];assert a['valid']==b['valid']
            if a['c'] is not None:
                assert a['c']['position']==b['c']['position'] and a['c']['acceleration']==b['c']['acceleration']
                changed+=a['c']['velocity']!=b['c']['velocity']
        assert baseline_delta<1e-10
        tables={}
        if name in ['side','train2']:
            label=ROOT.parent/'output'/('dense_labels_side_110_1195' if name=='side' else 'dense_labels_train2_40_46')/'labels.json'
            records=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda q:q['t'])
            for h in [.1,.2,.4]:
                errors={k:[] for k in models};missing=0
                for pub in pp:
                    if not pub['valid']:continue
                    truth=reference_at(records,pub['t']+h)
                    if truth is None:missing+=1;continue
                    for key in models:
                        c=states[key][round(pub['observation_t'],6)]['c']
                        p=predictions(c,pub['observation_age_ms']/1000.)[0][str(h)]
                        errors[key].append(float(np.linalg.norm(np.asarray(p)-truth)))
                tables[str(h)]=dict(metrics={k:stats(e) for k,e in errors.items()},missing_reference=missing)
        (OUT/(cohort+'_'+name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in traces)+'\n',encoding='utf-8')
        results[cohort][name]=dict(observations=len(rr),changed_velocity=changed,baseline_max_difference=baseline_delta,scores=tables)
primary=results['independent_clock']
passed=all(primary[n]['scores']['0.4']['metrics']['CA_velocity']['mean']<primary[n]['scores']['0.4']['metrics']['baseline']['mean'] and
    primary[n]['scores']['0.1']['metrics']['CA_velocity']['mean']<=primary[n]['scores']['0.1']['metrics']['baseline']['mean']+1. for n in ['side','train2'])
report=dict(design=design,results=results,provisional_gate_passed=passed,default_promoted=False,
    RAW_or_GPU_replayed=False,reference='Known development recordings, same valid publications and actual targets; manual-center interpolation<=150ms. No blind efficacy or new independent velocity truth.')
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
compact={cohort:{n:r['scores'] for n,r in cases.items()} for cohort,cases in results.items()}
print(json.dumps(dict(provisional_gate=passed,results=compact)),flush=True)
