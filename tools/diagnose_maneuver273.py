"""Separate current derivative error from future new motion.

Centred outline derivatives deliberately use +/-0.1s. They are an offline
counterfactual, NEVER a causal predictor or claimed accuracy improvement.
"""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from replay_joint236 import source
from physical_publication import evaluate
from compare_final_pipeline import reference_at,stats
from adaptive_force_memory import kernel

if __name__=='__main__':
    OUT=ROOT.parent/'output/maneuver273_20261108';OUT.mkdir(exist_ok=False)
    dense=json.loads((ROOT.parent/'output/motion271_20261108/results.json').read_text('utf8'));result={}
    for n in ['13','15','16']:
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()];points=dense['cases'][n]['dense_proxy'];records=[]
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        refs=sorted(json.loads((ROOT.parent/f'output/{folder}/references.json').read_text('utf8'))['records'].values(),key=lambda r:r['t'])
        for t in np.round(np.arange(10.8,15.6001,.2),1):
            r=[r for r in rows if r['t']<=t+1e-9][-1];t0=r['t'];target=reference_at(refs,t+.4)
            h=t-t0+.4;local=[p for p in points if abs(p['t']-t0)<=.100001 and p['center'] is not None]
            if not r['valid'] or target is None or len(local)<6:continue
            tt=np.array([p['t']-t0 for p in local]);X=np.c_[np.ones(len(tt)),tt,.5*tt*tt]
            Y=np.array([p['center'] for p in local]);beta=np.linalg.lstsq(X,Y,rcond=None)[0]
            c=r['coefficient'];p=np.array(c['position']);v=np.array(c['velocity']);a=np.array(c['acceleration'])
            pred=np.asarray(evaluate(r,[h])[str(float(h))]);kp=kernel(-1/.35,h)
            variants=dict(baseline=pred,oracle_current_velocity=pred+h*(beta[1]-v),
                oracle_current_acceleration=pred+kp*(beta[2]-a),
                oracle_current_both=pred+h*(beta[1]-v)+kp*(beta[2]-a),
                oracle_current_both_CA=p+h*beta[1]+.5*h*h*beta[2])
            records.append(dict(t=float(t),current_velocity=v.tolist(),centred_outline_velocity=beta[1].tolist(),
                current_acceleration=a.tolist(),centred_outline_acceleration=beta[2].tolist(),
                fit_residual_px=np.sqrt(np.mean((Y-X@beta)**2,axis=0)).tolist(),future_lookahead_s=float(tt[-1]),
                errors={k:float(np.linalg.norm(val-target)) for k,val in variants.items()}))
        summary={arm:stats([r['errors'][arm] for r in records]) for arm in records[0]['errors']}
        result[n]=dict(summary=summary,records=records)
        print(n,{k:round(v['mean'],3) for k,v in summary.items()},flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(cases=result,not_a_deployable_prediction=True,
        future_lookahead_s=.1,weak_outline_not_independent_COM_truth=True),indent=2),'utf8')
