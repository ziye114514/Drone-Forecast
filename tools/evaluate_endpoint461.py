"""Reuse the existing endpoint angular trend after the positive-force prior.

No position/force posterior, period, reference or learned coefficient changes.
All six inference streams are frozen before reading the old scoring references.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from endpoint_normal_trend import EndpointNormalTrend
from normal_turn_forecast import NormalTurnForecast
from physical_publication import evaluate
from extend_inverse321 import references
from score_rotor164 import score

def replace(packet,trend):
    if (packet.get('normal_turn_forecast') or {}).get('applied'):packet['normal_turn_forecast']['trend']=copy.deepcopy(trend)
    for part in packet['force'].get('response_model_mixture',[]):replace(part,trend)

if __name__=='__main__':
    out=ROOT.parent/'output/endpoint461_20261111';out.mkdir(exist_ok=False)
    files=['endpoint_normal_trend.py','tools/evaluate_endpoint461.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,reuses_existing272=True,physical_prior457_unchanged=True,
        coefficient_and_force_posterior_unchanged=True,all6_uniform=True,no_fit_or_threshold_search=True,
        target31=True,all_inference_before_references=True,new_video_unread=True,FPS_work=False),indent=2),'utf8')
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    # Existing static-centre protection must remain active under plane wobble.
    from axis_amplitude_bridge import current_axis_qualified
    ns=('13','15','16','11','3','4');counts={};parity={}
    for n in ns:
        rows=[json.loads(s) for s in (ROOT.parent/f'output/positive_normal457_20261110/{n}_positive.jsonl').read_text('utf8').splitlines()]
        memory=EndpointNormalTrend();old_memory=NormalTurnForecast();last_obs=None;outputs=[];changed=0;delta=0.;angular_delta=0.
        for old in rows:
            r=copy.deepcopy(old);obs=r['state'].get('observations')
            if obs is not None and last_obs is not None and obs<last_obs:memory=EndpointNormalTrend();old_memory=NormalTurnForecast()
            last_obs=obs;m=r['motion'];e=m.get('estimate') or {};dt=m.get('dt_s');flow=dict(valid=False)
            if dt and m.get('current_observation') and not e.get('peak_at_search_boundary') and e.get('score',0)>0:flow=dict(valid=True,velocity=[e['dx']/dt,e['dy']/dt])
            trend=memory.observe(r['t'],r['structure'],r['coefficient'],dict(observation=flow))
            prior=old_memory.observe(r['t'],r['structure'],r['coefficient'],dict(observation=flow))
            stored=(old.get('normal_turn_forecast') or {}).get('trend') or {}
            if (old.get('normal_turn_forecast') or {}).get('applied'):
                angular_delta=max(angular_delta,abs(prior['shrunk_angular_rate_rad_s']-stored['shrunk_angular_rate_rad_s']))
            if r['coefficient']:
                before=evaluate(r,[.1,.2,.4]);delta=max(delta,max(np.max(abs(np.array(v)-old['physical'][h])) for h,v in before.items()))
            if trend.get('qualified'):replace(r,trend);changed+=bool((r.get('normal_turn_forecast') or {}).get('applied'))
            r['endpoint_plane_trend']=trend;r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};outputs.append(r)
            assert r['coefficient']==old['coefficient'] and r['valid']==old['valid'] and r['body']==old['body']
        assert delta<1e-7 and angular_delta<1e-7,(n,delta,angular_delta);parity[n]=dict(prediction_px=float(delta),old_angular_trend_difference=float(angular_delta));counts[n]=changed
        (out/f'{n}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8');print(n,'frozen',changed,flush=True)
    original=json.loads((ROOT.parent/'output/reference429_20261110/results.json').read_text('utf8'));cases={};pool={a:[] for a in ('positive457','endpoint461')}
    for n in ns:
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n);ids={q['t'] for q in original['cases'][n]['control410']['records'] if q['valid']};cases[n]={}
        for a,p in [('positive457',ROOT.parent/f'output/positive_normal457_20261110/{n}_positive.jsonl'),('endpoint461',out/f'{n}.jsonl')]:
            s=score(p,ref,np.round(np.arange(10.8,15.6001,.2),1));v=[q['error_px'] for q in s['records'] if q['valid'] and q['t'] in ids];assert len(v)==len(ids);pool[a].extend(v);s['original_available']=dict(n=len(v),mean=float(np.mean(v)));cases[n][a]=s
        print(n,{a:s['original_available'] for a,s in cases[n].items()},flush=True)
    pooled={a:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),within31=sum(x<=31 for x in v)) for a,v in pool.items()}
    (out/'results.json').write_text(json.dumps(dict(cases=cases,pooled=pooled,control_parity=parity,changed_counts=counts,original102_preserved=True,new_video_validated=False,FPS_work=False),indent=2),'utf8');print('POOLED',pooled,flush=True)
