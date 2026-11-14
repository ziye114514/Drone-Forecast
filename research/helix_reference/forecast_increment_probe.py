"""Frozen RAW-output replay: does verified rotation add to pure position bias?

No new RAW/NN run. Feedback is the current detector center, not independent
trajectory truth. Fit only already-due predictions; assess issued candidates
prequentially. This cannot establish absolute thrust or actual flight accuracy.
"""
from pathlib import Path
from collections import deque
import argparse,hashlib,json,shutil
import numpy as np
from scipy.optimize import lsq_linear

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT.parent/'output/helix_author_probe_20261025'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-name',default='forecast_increment_probe');args=parser.parse_args()
OUT=BASE/args.output_name;OUT.mkdir(exist_ok=False)
shutil.copy2(Path(__file__),OUT/'forecast_increment_probe.py')
PARAMS=dict(horizons=[.1,.2,.4],history_s=1.,minimum_matured=10,minimum_prequential_losses=10,
    prior_pixel_sigma=6.,coefficient_lower_bound=0.,maximum_feature_weight=.25,
    minimum_past_relative_gain=.05,normal_sign_assumption='Choose image-up projected normal; camera gravity not calibrated',
    physical_identifiability=False,feedback='same detector CURRENT center; evaluation self-consistency only')
(OUT/'protocol.json').write_text(json.dumps(PARAMS,indent=2),encoding='utf-8')

def feature(row):
    obs=row['auxiliary_observation'];proxy=obs.get('conditional_rotation_proxy') or {}
    rho=proxy.get('relative_to_first_supported')
    if not obs.get('valid') or rho is None:return None
    vectors=[]
    for v in obs['rotors'].values():
        p=v.get('current_projection')
        if p is None:return None
        n=np.asarray(p['normal_axis_unoriented'],float)
        if n[1]>0:n=-n
        vectors.append(n*np.sqrt(max(0.,1-p['minor_to_major']**2)))
    return rho*np.median(np.asarray(vectors),axis=0)

source=BASE/'RAW_shadow_recovery/predictions.jsonl'
rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
pending=[];samples={h:deque() for h in PARAMS['horizons']};losses={h:deque() for h in PARAMS['horizons']}
issued=[];scored=[];current_epoch=None;previous_body=None
for row in rows:
    t=row['t'];body=row['body'];obs=row['auxiliary_observation'];epoch=obs.get('rotation_proxy_epoch')
    point=np.asarray(body['center'],float) if body.get('detected') else None
    remaining=[]
    for item in pending:
        if item['target_t']>t+1e-8:remaining.append(item);continue
        if point is None or abs(item['target_t']-t)>1e-7:continue
        h=item['h'];base=item['baseline'];target=point-base
        errors={k:float(np.linalg.norm(item[k]-point)) for k in ['baseline','bias','feature','selected','selected_same_gate_bias']}
        scored.append(dict(origin_t=item['origin_t'],target_t=t,h=h,active=item['active'],errors=errors,epoch=item['epoch']))
        if item['epoch']==current_epoch:
            samples[h].append((t,item['u'],target))
            if item['fit_ready']:losses[h].append((t,{k:e*e for k,e in errors.items()}))
    pending=remaining
    if epoch!=current_epoch:
        current_epoch=epoch;samples={h:deque() for h in PARAMS['horizons']};losses={h:deque() for h in PARAMS['horizons']}
    u=feature(row)
    if u is None or not row['valid'] or row['preferred_forecast'] not in row['predictions']:continue
    for h in PARAMS['horizons']:
        for queue in (samples[h],losses[h]):
            while queue and t-queue[0][0]>PARAMS['history_s']+1e-8:queue.popleft()
        base=np.asarray(row['predictions'][row['preferred_forecast']][str(h)],float)
        bias=np.zeros(2);k=0.;mean_u=np.zeros(2);ready=len(samples[h])>=PARAMS['minimum_matured']
        if samples[h]:bias=np.mean([s[2] for s in samples[h]],axis=0)
        if ready:
            us=np.asarray([s[1] for s in samples[h]]);ys=np.asarray([s[2] for s in samples[h]])
            mean_u=us.mean(axis=0);centered=us-mean_u;residual=ys-bias
            # One nonnegative shared image response coefficient. Centering
            # gives the pure position-bias competitor exactly the same samples.
            A=centered.reshape(-1,1);b=residual.ravel()
            A=np.vstack([A,[[1/PARAMS['prior_pixel_sigma']]]]);b=np.r_[b,0.]
            fit=lsq_linear(A,b,bounds=(0.,np.inf),method='bvls')
            if fit.success:k=float(fit.x[0])
        pb=base+bias;pf=pb+k*(u-mean_u);active=False
        if ready and len(losses[h])>=PARAMS['minimum_prequential_losses']:
            past={key:np.mean([s[1][key] for s in losses[h]]) for key in ['baseline','bias','feature']}
            active=bool(past['feature']<(1-PARAMS['minimum_past_relative_gain'])*min(past['baseline'],past['bias']))
        selected=pb+PARAMS['maximum_feature_weight']*(pf-pb) if active else base
        item=dict(origin_t=t,target_t=round(t+h,8),epoch=epoch,h=h,u=u,baseline=base,bias=pb,
            feature=pf,selected=selected,active=active,fit_ready=ready,k=k)
        item['selected_same_gate_bias']=pb if active else base
        pending.append(item)
        issued.append(dict(origin_t=t,h=h,epoch=epoch,fit_ready=ready,active=active,k=k,
                           matured_samples=len(samples[h]),past_candidate_losses=len(losses[h])))
summary={}
for h in PARAMS['horizons']:
    seq=[r for r in scored if r['h']==h];pred=[r for r in issued if r['h']==h]
    summary[str(h)]=dict(scored=len(seq),issued=len(pred),active_issued=sum(r['active'] for r in pred),
        active_scored=sum(r['active'] for r in seq),fit_ready=sum(r['fit_ready'] for r in pred),
        mean_error_px={key:float(np.mean([r['errors'][key] for r in seq])) if seq else None for key in ['baseline','bias','feature','selected','selected_same_gate_bias']})
result=dict(scope='Causal replay of frozen RAW-output states, same-detector self-consistency; not independent trajectory truth',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),protocol=PARAMS,summary=summary,issued=issued,scored=scored,
    integrated_into_main=False)
(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
