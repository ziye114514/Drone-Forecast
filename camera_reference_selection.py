"""Native background evidence selects a coordinate hypothesis, never a drone.

H0: no resolved common background motion. H1: coherent v/a with the EXISTING
1000px/s and1000px/s² priors. Equal prior odds, original.12s support, and shared
image endpoint noise determine a Bayes factor. No forecast labels select it.
The two rotor predictions share data and are not independent measurements.
"""
from collections import deque
import copy
import numpy as np

class ReferenceEvidenceWeight:
    def __init__(self):self.rows=deque();self.last=None
    def step(self,evidence):
        t=evidence['t'];pair=evidence.get('pair')
        if self.last is not None and t<=self.last:raise ValueError('Increasing cutoffs required')
        self.last=t
        if pair is None:
            self.rows.clear();return dict(weight=0.,reason='no_current_common_background_observation',rows=0)
        if self.rows and abs(pair['source_t']-self.rows[-1]['t'])>1e-6:self.rows.clear()
        self.rows.append(pair)
        while self.rows and t-self.rows[0]['source_t']>.12+1e-8:self.rows.popleft()
        if len(self.rows)<3:return dict(weight=0.,reason='need_existing_three_observed_intervals',rows=len(self.rows))
        rr=list(self.rows);n=len(rr);X=np.zeros((2*n,4));C=np.zeros((2*n,2*n));y=[]
        for i,r in enumerate(rr):
            s,e=r['source_t'],r['t'];X[2*i:2*i+2,:2]=(e-s)*np.eye(2)
            X[2*i:2*i+2,2:]=.5*((e-t)**2-(s-t)**2)*np.eye(2)
            C[2*i:2*i+2,2*i:2*i+2]=r['covariance_px2'];y.extend(r['displacement_xy'])
            if i and abs(s-rr[i-1]['t'])<1e-6:
                C[2*i:2*i+2,2*i-2:2*i]=-2*np.eye(2);C[2*i-2:2*i,2*i:2*i+2]=-2*np.eye(2)
        y=np.asarray(y);slab=C+1e6*(X@X.T)
        sign0,ld0=np.linalg.slogdet(C);sign1,ld1=np.linalg.slogdet(slab)
        assert sign0>0 and sign1>0
        log_bf=.5*(float(y@np.linalg.solve(C,y)-y@np.linalg.solve(slab,y))-(ld1-ld0))
        weight=float(1/(1+np.exp(-np.clip(log_bf,-700,700))))
        return dict(weight=weight,reason='native_background_static_vs_coherent_motion_evidence',
            rows=n,span_s=t-rr[0]['source_t'],log_bayes_factor=log_bf,
            equal_prior_odds=True,old_velocity_acceleration_prior_variance=1e6,
            shared_background_image_errors_in_likelihood=True,current_forecast_error_or_labels_used=False,
            posterior_is_engineering_model_not_calibrated_camera_truth=True)


class ReferenceMemoryWeight:
    """Qualification memory, not independent repeated evidence or a posterior.

    A quiet current background cannot erase prior contamination of the static
    coordinate observer. Its existing Singer.35s state time scale supplies the
    decay; max retains support without multiplying overlapping likelihoods.
    """
    def __init__(self):
        self.current=ReferenceEvidenceWeight();self.last=None;self.weight=0.
    def step(self,evidence):
        t=evidence['t'];info=self.current.step(evidence)
        carried=0. if self.last is None else self.weight*np.exp(-(t-self.last)/.35)
        instantaneous=info['weight'];self.weight=max(instantaneous,float(carried));self.last=t
        return dict(info,weight=self.weight,instantaneous_weight=instantaneous,
            carried_weight=float(carried),reason='current_background_or_existing_force_memory_support',
            qualification_memory_s=.35,qualification_memory_reuses_force_correlation=True,
            max_support_does_not_multiply_overlapping_likelihoods=True,
            weight_is_qualification_not_calibrated_model_posterior=True)

def mix(baseline,camera,info):
    from physical_publication import evaluate,freeze
    assert baseline['t']==camera['t'] and baseline['valid']==camera['valid']
    weight=float(info['weight']);r=copy.deepcopy(baseline)
    if r['coefficient'] is None:return r
    a=freeze(baseline,baseline['t']);b=freeze(camera,camera['t'])
    r['background_reference_model_average']=dict(info,baseline=a,background=b,
        same_native_rotor_front=True,branches_are_coordinate_hypotheses_not_independent_tracks=True,
        known_and_hidden_rotor_measurements_unchanged=True)
    for key in ('position','velocity','acceleration'):
        r['coefficient'][key]=((1-weight)*np.asarray(baseline['coefficient'][key])+weight*np.asarray(camera['coefficient'][key])).tolist()
    r['state']['reference_mixture_covariance_not_calibrated']=True
    r['force']['reference_hypothesis_force_packets_preserved']=True
    r['physical']=evaluate(r,[.1,.2,.4]);return r

def predict(row,horizons):
    from physical_publication import evaluate
    info=row['background_reference_model_average'];a=evaluate(info['baseline'],horizons);b=evaluate(info['background'],horizons);w=info['weight']
    return {h:((1-w)*np.asarray(a[h])+w*np.asarray(b[h])).tolist() for h in a}
