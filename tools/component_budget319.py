"""Noncausal diagnosis of normal-response vs external-residual continuation.

Future values are POSTERIOR image estimates, not thrust/force ground truth.
This module is deliberately tools-only and never called by inference.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from replay_joint236 import source
from physical_publication import evaluate
from compare_final_pipeline import reference_at,stats
from axis_amplitude_bridge import current_axis_qualified
X,W=np.polynomial.legendre.leggauss(24)

def normal_force(r):
    f=r['force'];known=np.asarray(f.get('unobserved_normal_response_px_s2') or [0,0],float)
    if f.get('active') and f.get('input_u') is not None:known=known+f['effective_gain_px_s2']*np.asarray(f['input_u'])
    return known

def interventions(row,rows,h):
    t=row['t'];s=(X+1)*h/2;weights=W*h/2*(h-s)
    future=[r for r in rows if t-1e-8<=r['t']<=t+h+.04 and r['coefficient']]
    if not future or future[-1]['t']<t+h-.04 or max(np.diff([r['t'] for r in future]),default=0)>.08:return {}
    tt=np.array([r['t']-t for r in future]);normal=np.array([normal_force(r) for r in future]);aa=np.array([r['force']['acceleration_px_s2'] for r in future]);ext=aa-normal
    interp=lambda vals:np.c_[np.interp(s,tt,vals[:,0]),np.interp(s,tt,vals[:,1])]
    def frozen(component):
        f=component['force']
        if f.get('response_model_mixture'):
            ps=[(p['weight'],frozen(p)) for p in f['response_model_mixture']];return {k:sum(w*p[k] for w,p in ps) for k in ps[0][1]}
        c=component['coefficient'];p=np.asarray(c['position']);v=np.asarray(c['velocity']);a=np.asarray(f['acceleration_px_s2']);n0=normal_force(component);e0=a-n0
        decay=np.exp(-s/f['acceleration_correlation_s']);base=copy.deepcopy(component);base['normal_turn_forecast']=dict(applied=False)
        original=np.asarray(evaluate(base,[h])[str(float(h))]);net=p+h*v+weights@(decay[:,None]*a)
        other=original-net
        return dict(future_estimated_normal_response_not_thrust=p+h*v+weights@(interp(normal)+decay[:,None]*e0)+other,
            future_estimated_external_residual_not_wind=p+h*v+weights@(interp(ext)+decay[:,None]*n0))
    result={a:p.tolist() for a,p in frozen(row).items()}
    axes=[r for r in future if current_axis_qualified(r['structure']) and r['structure'].get('axis_xy') is not None]
    if len(axes)<3 or axes[0]['t']>t+1e-8 or axes[-1]['t']<t+h-.04 or max(np.diff([r['t'] for r in axes]))>.12:return result
    ax=np.array([r['structure']['axis_xy'] for r in axes]);length=np.linalg.norm(ax,axis=1)
    if min(length)<1e-6:return result
    ts=np.array([r['t']-t for r in axes]);angles=np.unwrap(2*np.arctan2(ax[:,1],ax[:,0]))/2;delta=np.interp(s,ts,angles)-angles[0]
    ratio=np.interp(s,ts,length)/length[0]*np.exp(1j*delta)-1
    def plane(component):
        f=component['force']
        if f.get('response_model_mixture'):return sum(p['weight']*plane(p) for p in f['response_model_mixture'])
        base=copy.deepcopy(component);base['normal_turn_forecast']=dict(applied=False);p=np.asarray(evaluate(base,[h])[str(float(h))])
        z=complex(*normal_force(component));correction=np.sum(weights*np.exp(-s/f['acceleration_correlation_s'])*ratio)*z
        return p+np.array([correction.real,correction.imag])
    result['future_measured_projected_normal_NOT_truth']=plane(row).tolist();return result

if __name__=='__main__':
    out=ROOT.parent/'output/component_budget319_20261109';out.mkdir(exist_ok=False);(out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());cases={}
    for n in ('13','15','16'):
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()];folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        truth=sorted(json.loads((ROOT.parent/f'output/{folder}/references.json').read_text('utf8'))['records'].values(),key=lambda z:z['t']);records=[]
        for t in np.round(np.arange(10.8,15.6001,.2),1):
            r=[r for r in rows if r['t']<=t+1e-9][-1];target=reference_at(truth,t+.4)
            if not r['valid'] or target is None:records.append(dict(t=float(t),valid=False));continue
            h=t-r['t']+.4;arms=dict(baseline=evaluate(r,[h])[str(float(h))]);arms.update(interventions(r,rows,h))
            records.append(dict(t=float(t),valid=True,errors={a:float(np.linalg.norm(np.asarray(p)-target)) for a,p in arms.items()}))
        names=sorted(set(k for r in records if r['valid'] for k in r['errors'])-{'baseline'});summary={}
        for arm in names:
            common=[r['errors'] for r in records if r['valid'] and arm in r['errors']];summary[arm]=dict(baseline=stats([r['baseline'] for r in common]),intervention=stats([r[arm] for r in common]))
        cases[n]=dict(summary=summary,records=records);print(n,summary,flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=cases,diagnostic_uses_future=True,causal_predictor=False,forces_are_not_independent_truth=True),indent=2),'utf8')
