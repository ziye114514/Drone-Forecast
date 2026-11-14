"""One cadence-noise mechanism on frozen observations and exact same origins."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from live_covariance import process_noise,SPECTRAL_DENSITY
from summarize_live_pilot import reference_at,stats,read_rows

OUT=ROOT.parent/'output/live_noise_mechanism_20261025'
OUT.mkdir(exist_ok=False)
F=lambda dt:np.block([[np.eye(2),np.eye(2)*dt],[np.zeros((2,2)),np.eye(2)]])
for a,b in [(.02,.02),(.02,.08),(.04,.11)]:
    np.testing.assert_allclose(process_noise(a+b),F(b)@process_noise(a)@F(b).T+process_noise(b),rtol=1e-12,atol=1e-9)
    assert np.linalg.eigvalsh(process_noise(a)).min()>0
assert process_noise(.02)[2,2]==6400.**2*.02**2
results={};max_diff=0.
for name in ['clear','side','train2']:
    frozen=ROOT.parent/'output/live_forecast_final_v2_20261025'/name
    tracking=read_rows(frozen/'tracking.jsonl');pubs=read_rows(frozen/'predictions.jsonl')
    models={n:LiveMotion(25.,noise_model=n) for n in ['legacy','continuous']};states={n:{} for n in models}
    for row in tracking:
        for n,model in models.items():
            c,valid,diag=model.step(row['t'],row['body']);states[n][round(row['t'],6)]=c
            if n=='legacy' and c is not None:
                for field in ('position','velocity','acceleration'):
                    max_diff=max(max_diff,float(np.max(np.abs(np.asarray(c[field])-row['coefficient'][field]))))
    table={}
    if name!='clear':
        label=ROOT.parent/'output'/('dense_labels_side_110_1195' if name=='side' else 'dense_labels_train2_40_46')/'labels.json'
        records=sorted(json.loads(label.read_text(encoding='utf-8'))['records'].values(),key=lambda r:r['t'])
        for h in (.1,.2,.4):
            errors={n:[] for n in models};detail=[];missing=0
            for pub in pubs:
                if not pub['valid']:continue
                ref=reference_at(records,pub['t']+h)
                if ref is None:missing+=1;continue
                tick=round(pub['observation_t'],6);points={}
                for n in models:
                    p,_=predictions(states[n][tick],pub['observation_age_ms']/1000.)
                    points[n]=p[str(h)];errors[n].append(float(np.linalg.norm(np.asarray(points[n])-ref)))
                detail.append(dict(t=pub['t'],error={n:errors[n][-1] for n in errors}))
            table[str(h)]=dict(metrics={n:stats(v) for n,v in errors.items()},missing=missing,detail=detail)
    results[name]=table
assert max_diff<1e-10,max_diff
report=dict(mechanism='Pinned FilterPy continuous Q; spectral power matched to existing50Hz velocity diffusion, no fit/sweep',
            spectral_density=SPECTRAL_DENSITY,baseline_replay_max_difference=max_diff,semigroup_control_passed=True,
            data='Frozen actual live tracking/publications; observation replay, not new RAW/NN inference; interpolated reference only',results=results)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(max_difference=max_diff,results={n:{h:q['metrics'] for h,q in hs.items()} for n,hs in results.items()})),flush=True)
