"""Retrospective diagnosis only: labels NEVER enter the runtime forecaster.

Cannot prove a pilot's future command was unpredictable. Tests whether a
past-only local quadratic position reference explains the future, and gives
an exact signed-vector error accounting conditional on that noisy reference.
"""
from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'tools')]
from compare_final_pipeline import reference_at,stats
OUT=ROOT.parent/'output/maneuver_response_20261029'
SPECS=[('side','motion_consistency_30px_20261028/RAW_side_release_candidate8s/predictions.jsonl','dense_labels_side_110_1195'),
       ('other','thrust_rate_review_20261029/experiment/other_baseline.jsonl','dense_labels_train2_40_46')]


def fit_reference(labels,t,left=.5,right=0.):
    data=[r for r in labels if t-left-1e-8<=r['t']<=t+right+1e-8 and r.get('state')=='present' and r.get('box')]
    if len(data)<5:return None
    stamps=np.array([r['t'] for r in data]);span=float(np.ptp(stamps))
    if span<.35-1e-8 or np.max(np.diff(stamps))>.1500001:return None
    x=(stamps-t)/.5;A=np.column_stack((np.ones(len(x)),x,.5*x*x))
    y=np.array([(np.array(r['box'][:2])+r['box'][2:])/2 for r in data])
    beta=np.linalg.lstsq(A,y,rcond=None)[0];residual=y-A@beta
    variance=max(36.,float(np.sum(residual*residual)/max(1,2*(len(data)-3))))
    C=np.linalg.inv(A.T@A)*variance
    return dict(position=beta[0],velocity=beta[1]/.5,acceleration=beta[2]/.25,
        beta=beta,covariance=C,noise_px=float(np.sqrt(variance)),n=len(data),span=span,
        maximum_label_t=float(stamps[-1]),median_residual_px=float(np.median(np.linalg.norm(residual,axis=1))))


def reference_prediction(fit,h):
    x=h/.5;row=np.array([1.,x,.5*x*x])
    return row@fit['beta'],float(row@fit['covariance']@row+36.)


def main():
    OUT.mkdir(exist_ok=True);result={}
    for name,source,label in SPECS:
        rows=[json.loads(x) for x in (ROOT.parent/'output'/source).read_text().splitlines()]
        labels=sorted(json.loads((ROOT.parent/'output'/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        records=[];last=None
        for r in rows:
            slot=int(np.floor(r['t']*5+1e-8))
            if not r['valid'] or slot==last:continue
            last=slot;t=r['t'];future=reference_at(labels,t+.4)
            if future is None:continue
            current=reference_at(labels,t);error=np.array(r['physical']['0.4'])-future
            entry=dict(t=t,error_px=float(np.linalg.norm(error)),current_force=bool(r['force']['active']),
                current_motion_qualified=bool((r.get('image_motion') or {}).get('observation',{}).get('valid')),
                current_acceleration_qualified=bool((r.get('image_motion') or {}).get('acceleration',{}).get('applied')),
                mode=r['structure']['mode'],forecast=r['physical']['0.4'],future_reference=future.tolist())
            past=fit_reference(labels,t);centered=fit_reference(labels,t,.25,.25)
            if current is not None:entry['current_position_error_px']=float(np.linalg.norm(np.asarray(r['coefficient']['position'])-current))
            if past is not None:
                assert past['maximum_label_t']<=t+1e-8
                pred,var=reference_prediction(past,.4);innovation=pred-future;chi=float(innovation@innovation/var)
                state=r['coefficient'];p=np.array(state['position']);v=np.array(state['velocity']);a=np.array(past['acceleration'])
                # Exact vector identity; magnitudes themselves are NOT additive.
                origin=p-past['position'];velocity=.4*(v-past['velocity'])
                dynamics=np.array(r['physical']['0.4'])-p-.4*v-.5*.4**2*a
                later=pred-future
                assert np.linalg.norm(origin+velocity+dynamics+later-error)<1e-7
                unit=error/max(np.linalg.norm(error),1e-9)
                entry['past_reference']=dict(n=past['n'],span_s=past['span'],maximum_label_t=past['maximum_label_t'],
                    velocity=past['velocity'].tolist(),acceleration=a.tolist(),prediction=pred.tolist(),
                    future_residual_px=float(np.linalg.norm(innovation)),future_standard_error_proxy_px=float(np.sqrt(var)),
                    future_innovation_chi2_proxy=chi,noise_floor_px=6.,uncertainty_calibrated=False,
                    future_outside_past_quadratic_proxy=chi>5.991464547107979,
                    current_velocity_difference_px_s=(v-past['velocity']).tolist(),
                    signed_error_projection_px={k:float(value@unit) for k,value in
                        [('position_reference',origin),('velocity_reference',velocity),('force_rollout_reference',dynamics),('later_reference_innovation',later)]},
                    interpretation='Conditional on a noisy past-only reference; not a ground-truth acceleration or pilot-command label')
            if centered is not None:
                pred,var=reference_prediction(centered,.4)
                entry['centered_reference_DIAGNOSTIC_ONLY']=dict(future_labels_used_until=centered['maximum_label_t'],
                    error_px=float(np.linalg.norm(pred-future)),prediction=pred.tolist(),
                    purpose='Retrospective state estimate; NOT a causal baseline, lower bound, or deployable score')
            records.append(entry)
        eligible=[r for r in records if 'past_reference' in r]
        changes=[r for r in eligible if r['past_reference']['future_outside_past_quadratic_proxy']]
        stable=[r for r in eligible if not r['past_reference']['future_outside_past_quadratic_proxy']]
        result[name]=dict(origins=len(records),baseline_error=stats([r['error_px'] for r in records]),
            past_reference_eligible=len(eligible),future_reference_inconsistent=len(changes),
            not_detectably_inconsistent=len(stable),insufficient_reference=len(records)-len(eligible),
            inconsistent_baseline_error=stats([r['error_px'] for r in changes]),
            not_detectably_inconsistent_baseline_error=stats([r['error_px'] for r in stable]),
            signed_projection_mean_px={key:float(np.mean([r['past_reference']['signed_error_projection_px'][key] for r in eligible]))
                for key in ('position_reference','velocity_reference','force_rollout_reference','later_reference_innovation')} if eligible else {},
            records=records)
    payload=dict(clips=result,retrospective_diagnosis_only=True,annotations_not_sent_to_runtime=True,
        does_not_identify_unpredictable_commands=True,source_model_or_DEFAULT_not_modified=True)
    (OUT/'diagnosis.json').write_text(json.dumps(payload,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='records'} for k,v in result.items()},indent=2))


if __name__=='__main__':main()
