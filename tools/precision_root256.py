"""One fixed precision error budget; diagnostic interventions, not tuning.

Past-native quadratic fits use only observations available at the origin.
Endpoint-optimal curvature uses future reference ONLY as an unattainable
diagnostic bound and must never become a runtime prediction or performance.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from replay_joint236 import source
from compare_final_pipeline import reference_at,stats
from physical_publication import evaluate
from adaptive_force_memory import kernel


def native_fits(rows):
    history=[];result=[]
    for r in rows:
        t=r['t'];body=r['body'];motion=r['motion'];shift=body.get('measurement_reference_shift_xy')
        if shift is not None:
            history=[(s,p+np.asarray(shift)) for s,p in history]
        history=[(s,p) for s,p in history if t-s<=.200001]
        current=bool(body.get('detected') and not body.get('center_provisional') and motion.get('current_observation'))
        if current:history.append((t,np.asarray(body['center'],float)))
        fit=None
        if current and len(history)>=6 and t-history[0][0]>=.16-1e-8:
            tt=np.array([s-t for s,p in history]);xy=np.array([p for s,p in history])
            if np.max(np.diff(tt))<=.08+1e-8:
                X=np.c_[np.ones(len(tt)),tt,.5*tt**2];beta=np.linalg.lstsq(X,xy,rcond=None)[0]
                residual=xy-X@beta;sigma=np.maximum(2.,np.sqrt(np.mean(residual**2,axis=0)))
                # Explicitly correlated window: retain only3 effective samples
                # for this diagnostic uncertainty, not N independent boxes.
                covariance=np.linalg.inv(X.T@X)*len(tt)/3
                fit=dict(t=t,position=beta[0].tolist(),velocity=beta[1].tolist(),acceleration=beta[2].tolist(),
                    velocity_sigma_proxy=(np.sqrt(covariance[1,1])*sigma).tolist(),
                    acceleration_sigma_proxy=(np.sqrt(covariance[2,2])*sigma).tolist(),
                    residual_rms_px=np.sqrt(np.mean(residual**2,axis=0)).tolist(),span_s=float(-tt[0]),samples=len(tt),
                    independent_truth=False)
        result.append(fit)
    return result


def persistent(row):
    r=copy.deepcopy(row)
    def update(f):
        f['acceleration_correlation_s']=1e10
        for a in f.get('response_model_mixture',[]):update(a['force'])
    update(r['force']);return r


if __name__=='__main__':
    OUT=ROOT.parent/'output/precision_root256_20261108';OUT.mkdir(exist_ok=False)
    protocol=dict(cases=['13','15','16'],origins='10.8..15.6 step0.2, fixed25',horizon_s=.4,
        native_past_fit_window_s=.2,min_samples=6,min_span_s=.16,largest_gap_s=.08,
        candidate_algorithms_implemented=False,NN_used=False,GPU_used=False,
        future_reference_used_only_for_scoring_or_diagnostic_lower_bound=True,
        no_parameter_sweep=True,no_recording_specific_rules=True,
        references_are_weak_native_outline_not_true_COM=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf-8')
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());cases={}
    for n in protocol['cases']:
        rows=[json.loads(s) for s in source(n).read_text('utf-8').splitlines()];fits=native_fits(rows)
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        truth=sorted(json.loads((ROOT.parent/f'output/{folder}/references.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t']);records=[]
        for t in np.round(np.arange(10.8,15.6001,.2),1):
            indices=[i for i,r in enumerate(rows) if r['t']<=t+1e-9];i=indices[-1] if indices else None
            r=rows[i] if i is not None else None;now=reference_at(truth,t);target=reference_at(truth,t+.4)
            item=dict(t=float(t),valid=False)
            if r is None or not r['valid'] or target is None or now is None or t-r['t']>.04:
                records.append(item);continue
            h=t-r['t']+.4;c=r['coefficient'];p=np.asarray(c['position']);v=np.asarray(c['velocity']);a=np.asarray(c['acceleration'])
            pred=np.asarray(evaluate(r,[h])[str(float(h))]);curve=pred-p-h*v
            interventions=dict(baseline=pred,reference_current_position_only=pred+now-p,
                remove_future_acceleration=p+h*v,
                hold_current_net_acceleration=p+h*v+.5*h*h*a,
                same_model_no_future_acceleration_decay=np.asarray(evaluate(persistent(r),[h])[str(float(h))]))
            # Endpoint-optimal POSITIVE scalar changes only the curvature
            # contribution, after correcting current position. It uses future
            # truth, so measures an upper bound, never a deployable forecast.
            desired=np.asarray(target)-now-h*v
            oracle_scale=max(0.,float(desired@curve/max(float(curve@curve),1e-12)))
            interventions['hindsight_best_curvature_amplitude']=now+h*v+oracle_scale*curve
            fit=fits[i]
            if fit is not None:
                vf=np.asarray(fit['velocity']);af=np.asarray(fit['acceleration']);k=kernel(-1/.35,h)
                interventions['past_native_velocity_only']=pred+h*(vf-v)
                interventions['past_native_acceleration_only']=pred+k*(af-a)
                interventions['past_native_velocity_and_acceleration']=pred+h*(vf-v)+k*(af-a)
            err={k:float(np.linalg.norm(np.asarray(val)-target)) for k,val in interventions.items()}
            q=pred-target;unit=curve/max(float(np.linalg.norm(curve)),1e-12)
            future_i=[j for j,s in enumerate(rows) if s['t']<=t+.4+1e-9];futurefit=fits[future_i[-1]] if future_i else None
            item.update(valid=True,source_t=r['t'],errors=err,prediction=pred.tolist(),target=np.asarray(target).tolist(),
                current_position_error_px=float(np.linalg.norm(p-now)),current_velocity=v.tolist(),current_net_acceleration=a.tolist(),
                future_curve_displacement_px=curve.tolist(),hindsight_amplitude_scale=oracle_scale,
                current_native_fit=fit,future_native_fit_for_diagnosis_only=futurefit,
                error_parallel_to_predicted_curve=float(q@unit),error_orthogonal_to_predicted_curve=float(np.linalg.norm(q-(q@unit)*unit)),
                input_u=r['force'].get('input_u'),structure_mode=r['structure']['mode'])
            records.append(item)
        common=[r for r in records if r['valid'] and r['current_native_fit'] is not None]
        arms=list(common[0]['errors']) if common else []
        summary={a:stats([r['errors'][a] for r in common]) for a in arms}
        large=[r for r in common if r['errors']['baseline']>30.]
        cases[n]=dict(eligible=25,scored=sum(r['valid'] for r in records),native_fit_common=len(common),
            common_summary=summary,large_error_common=len(large),
            large_error_summary={a:stats([r['errors'][a] for r in large]) for a in arms},
            position_error=stats([r['current_position_error_px'] for r in common]),records=records,
            root_causes_not_assigned_by_future_oracle_alone=True)
        print(n,'common',len(common),{k:round(v['mean'],2) if v['mean'] is not None else None for k,v in summary.items()},flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(protocol=protocol,cases=cases),indent=2),'utf-8')
