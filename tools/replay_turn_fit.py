"""Fixed direct-turn mechanism on identical prior valid publication targets."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from live_turn_fit import TurnFitMotion,turn_predictions,fit_turn,turn_design,DESIGN
from summarize_live_pilot import reference_at,stats,read_rows
OUT=ROOT.parent/'output/live_turn_fit_v3_20261026';OUT.mkdir(exist_ok=False)
(OUT/'source_snapshot').mkdir()
for name in ['live_turn_fit.py','tools/replay_turn_fit.py']:
    dest=OUT/'source_snapshot'/name;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes((ROOT/name).read_bytes())
design=dict(**DESIGN,source_sha256=hashlib.sha256((ROOT/'live_turn_fit.py').read_bytes()).hexdigest(),
    gate='Both primary independent-clock .4s means decrease; neither .1s mean worsens more than1px. No tuning after scoring.')
(OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
times=np.arange(0,.360001,.04);tau=times-times[-1];state=np.array([500.,300.,1000.,-100.])
points=(turn_design(tau,2.)@state).reshape(-1,2)
rate,_=fit_turn(list(zip(times,points)));assert abs(rate-2.)<1e-5
straight=(turn_design(tau,0.)@state).reshape(-1,2)
assert fit_turn(list(zip(times,straight)))[0] is None
c=dict(position=state[:2].tolist(),velocity=state[2:].tolist(),acceleration=[0.,0.],fitted_turn_rate=2.)
for h in [.1,.2,.4]:
    ideal=(turn_design(np.array([h]),2.)@state).reshape(-1,2)[0]
    target=state[:2]+max(.5,1-.9*h*h)*(ideal-state[:2])
    assert np.max(abs(np.asarray(turn_predictions(c)[0][str(h)])-target))<1e-9
results={}
for cohort,folder in [('old_live','live_forecast_final_v2_20261025'),('independent_clock','live_body_association_v2_20261025')]:
    results[cohort]={}
    for name in ['clear','side','train2']:
        run=ROOT.parent/'output'/folder/name;rr=read_rows(run/'tracking.jsonl');pp=read_rows(run/'predictions.jsonl')
        models=dict(baseline=LiveMotion(25.),turn=TurnFitMotion(25.));states={k:{} for k in models};qualified=0;traces=[]
        for row in rr:
            current={}
            for key,model in models.items():
                coeff,valid,diag=model.step(row['t'],row['body']);current[key]=(coeff,valid)
                states[key][round(row['t'],6)]=coeff
                if key=='turn':
                    qualified+=coeff is not None and coeff.get('fitted_turn_rate') is not None
                    traces.append(dict(t=row['t'],coefficient=coeff,valid=valid,diagnostic=diag))
                elif coeff is not None:
                    for field in ['position','velocity','acceleration']:assert coeff[field]==row['coefficient'][field]
            assert current['baseline'][1]==current['turn'][1]
            if current['baseline'][0] is not None:
                for field in ['position','velocity','acceleration']:assert current['baseline'][0][field]==current['turn'][0][field]
        tables={}
        if name in ['side','train2']:
            label=ROOT.parent/'output'/('dense_labels_side_110_1195' if name=='side' else 'dense_labels_train2_40_46')/'labels.json'
            records=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda q:q['t'])
            for h in [.1,.2,.4]:
                errors={k:[] for k in models};changed=0
                for pub in pp:
                    if not pub['valid']:continue
                    truth=reference_at(records,pub['t']+h)
                    if truth is None:continue
                    points={}
                    for key in models:
                        coeff=states[key][round(pub['observation_t'],6)]
                        points[key]=(predictions if key=='baseline' else turn_predictions)(coeff,pub['observation_age_ms']/1000.)[0][str(h)]
                        errors[key].append(float(np.linalg.norm(np.asarray(points[key])-truth)))
                    changed+=points['baseline']!=points['turn']
                tables[str(h)]=dict(metrics={k:stats(e) for k,e in errors.items()},changed_predictions=changed)
        (OUT/(cohort+'_'+name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in traces)+'\n',encoding='utf-8')
        results[cohort][name]=dict(turn_qualified_ticks=qualified,scores=tables)
primary=results['independent_clock']
passed=all(primary[n]['scores']['0.4']['metrics']['turn']['mean']<primary[n]['scores']['0.4']['metrics']['baseline']['mean'] and
    primary[n]['scores']['0.1']['metrics']['turn']['mean']<=primary[n]['scores']['0.1']['metrics']['baseline']['mean']+1 for n in ['side','train2'])
report=dict(design=design,ideal_turn_and_straight_controls_passed=True,results=results,provisional_gate_passed=passed,
    RAW_or_GPU_replayed=False,default_promoted=False,current_state_and_qualification_exact=True,
    scope='Known sparse manual-center interpolation<=150ms; same published valid targets; not independent generalization.')
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
