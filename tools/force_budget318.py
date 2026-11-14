"""Future observed plane/response interventions, DIAGNOSTIC ONLY.

Uses future native observations intentionally to ask which missing physical
information could change the error. Never exported as a causal predictor.
Neither future plane nor future filtered acceleration is independent truth.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from replay_joint236 import source
from physical_publication import evaluate
from compare_final_pipeline import reference_at,stats
from axis_amplitude_bridge import current_axis_qualified
X,W=np.polynomial.legendre.leggauss(24)

def intervened(row,rows,h):
    t=row['t'];s=(X+1)*h/2;weights=W*h/2*(h-s)
    future=[r for r in rows if t-1e-8<=r['t']<=t+h+.04 and r['coefficient']]
    if not future or future[-1]['t']<t+h-.04 or max(np.diff([r['t'] for r in future]),default=0)>.08:return {}
    times=np.array([r['t']-t for r in future]);acc=np.array([r['force']['acceleration_px_s2'] for r in future])
    c=row['coefficient'];p=np.asarray(c['position']);v=np.asarray(c['velocity'])
    out=dict(future_filtered_net_acceleration_NOT_truth=(p+h*v+weights@np.c_[np.interp(s,times,acc[:,0]),np.interp(s,times,acc[:,1])]).tolist())
    axes=[r for r in future if current_axis_qualified(r['structure']) and r['structure'].get('axis_xy') is not None]
    if len(axes)<3 or axes[0]['t']>t+1e-8 or axes[-1]['t']<t+h-.04 or max(np.diff([r['t'] for r in axes]))>.12:return out
    tt=np.array([r['t']-t for r in axes]);a=np.unwrap([2*np.arctan2(r['structure']['axis_xy'][1],r['structure']['axis_xy'][0]) for r in axes])/2
    delta=np.interp(s,tt,a)-a[0]
    future_rotation=np.exp(1j*delta)-1
    def plane_prediction(r):
        f=r['force']
        if f.get('response_model_mixture'):
            return sum(part['weight']*plane_prediction(part) for part in f['response_model_mixture'])
        # Remove only the previous future-normal rollout, preserving current
        # position, velocity, acceleration, unknown rate and memory exactly.
        plain=copy.deepcopy(r);plain['normal_turn_forecast']=dict(applied=False)
        base=np.asarray(evaluate(plain,[h])[str(float(h))])
        normal=r.get('normal_turn_forecast') or {};known=f.get('normal_turn_vector_px_s2')
        if known is None:
            known=np.asarray(f.get('unobserved_normal_response_px_s2') or [0,0],float)
            if f.get('active') and f.get('input_u') is not None:known=known+f['effective_gain_px_s2']*np.asarray(f['input_u'])
        z=complex(*known);memory=r.get('adaptive_force_memory') or {};long=memory.get('weight',0.) if memory.get('corrections_px') else 0.;short=memory.get('short_weight',0.) if memory.get('corrections_px') else 0.
        decay=(1-long-short)*np.exp(-s/f['acceleration_correlation_s'])+long+short*np.exp(-10*s)
        correction=np.sum(weights*decay*future_rotation)*z
        return base+np.array([correction.real,correction.imag])
    out['future_measured_plane_NOT_truth']=plane_prediction(row).tolist()
    return out

if __name__=='__main__':
    out=ROOT.parent/'output/force_budget318_20261109';out.mkdir(exist_ok=False)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());cases={}
    for n in ('13','15','16'):
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        truth=sorted(json.loads((ROOT.parent/f'output/{folder}/references.json').read_text('utf8'))['records'].values(),key=lambda z:z['t']);records=[]
        for t in np.round(np.arange(10.8,15.6001,.2),1):
            r=[r for r in rows if r['t']<=t+1e-9][-1];target=reference_at(truth,t+.4)
            if not r['valid'] or target is None:records.append(dict(t=float(t),valid=False));continue
            h=t-r['t']+.4;arms=dict(baseline=evaluate(r,[h])[str(float(h))]);arms.update(intervened(r,rows,h))
            records.append(dict(t=float(t),valid=True,errors={a:float(np.linalg.norm(np.asarray(p)-target)) for a,p in arms.items()},predictions=arms,target=target.tolist()))
        names=['future_measured_plane_NOT_truth','future_filtered_net_acceleration_NOT_truth'];summary={}
        for arm in names:
            common=[r['errors'] for r in records if r['valid'] and arm in r['errors']]
            summary[arm]=dict(baseline=stats([r['baseline'] for r in common]),intervention=stats([r[arm] for r in common]))
        cases[n]=dict(summary=summary,records=records);print(n,summary,flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=cases,diagnostic_uses_future=True,causal_predictor=False,independent_truth=False,not_a_performance_claim=True),indent=2),'utf8')
