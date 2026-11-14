"""Discovery-only linear frequency drift, independent late phase validation.

Original implementation of the continuous phase/velocity/acceleration model,
motivated by HelixTrack arXiv2603.09235 section3.3. No author code copied.
The same angular-order search and .5/9-sector gates remain. This is a CPU
observation prototype, not mechanical RPM truth or a production force input.
"""
import numpy as np
from scipy.optimize import minimize,LinearConstraint,Bounds
from angular_phase_candidates import ring_phase_candidates,ring_phase_samples
from predictive_harmonics import BASE_FREQUENCIES

def accelerating_ring_phase(events,ellipse,t,velocity=None):
    seed=ring_phase_candidates(events,ellipse,t,velocity)
    empty=dict(qualified=False,phase_hz=None,mechanical_RPM_measured=False,
        force_input_enabled=False,parameters_use_discovery_only=True,
        geometry_motion_error_can_mimic_rate_change=True,constant_rate_control=seed)
    data,reason=ring_phase_samples(events,ellipse,t,velocity)
    if data is None:return dict(empty,reason=reason)
    e,theta,w,train,late=[data[k] for k in ('events','theta','weight','train','late')]
    angular=seed['angular_harmonic'];fseed=seed['candidate_signed_phase_hz'];direction=1 if fseed>0 else -1
    duration=(data['mid']-data['begin'])/1e6
    s=(e['t']-data['begin'])/(duration*1e6)-.5
    basis=np.column_stack((s,.5*s*s));phase=angular*theta
    wt=w[train];den=max(float(wt.sum()),1e-12);bt=basis[train];pt=phase[train]
    def objective(c):
        z=wt*np.exp(1j*(pt-direction*2*np.pi*(bt@c)));mean=z.sum()/den
        derivative=(z[:,None]*(-direction*2j*np.pi*bt)).sum(0)/den
        return -float(abs(mean)**2),-2*np.real(np.conj(mean)*derivative)
    low=float(BASE_FREQUENCIES.min());high=float(BASE_FREQUENCIES.max()*4)
    c0=np.array([abs(fseed)*duration,0.])
    # Refine the existing frequency basin by half the original15Hz bin. Rate
    # drift is bounded only by keeping both window endpoints in the existing
    # temporal passband. No recording/annotation-dependent acceleration cap.
    bounds=Bounds([max(low,abs(fseed)-7.5)*duration,-np.inf],
                  [min(high,abs(fseed)+7.5)*duration,np.inf])
    last=(data['end']-data['begin'])/(duration*1e6)-.5
    constraint=LinearConstraint(np.array([[1.,-.5],[1.,last]]),low*duration,high*duration)
    result=minimize(objective,c0,jac=True,bounds=bounds,constraints=[constraint],method='SLSQP',
        options=dict(maxiter=60,ftol=1e-10))
    if not result.success or not np.isfinite(result.x).all():
        return dict(empty,reason='discovery_fit_failed',fit_message=str(result.message))
    c=result.x
    def projection(mask,sign):
        return np.sum(w[mask]*np.exp(1j*(phase[mask]-sign*2*np.pi*(basis[mask]@c))))/max(float(w[mask].sum()),1e-12)
    early=projection(train,direction);current=projection(late,direction);opposite=projection(late,-direction)
    power=abs(current)**2;other=abs(opposite)**2
    contrast=float((power-other)/max(power+other,1e-12))
    agreement=float(np.real(current*np.conj(early))/max(abs(early),1e-12))
    sectors=len(np.unique(np.floor((theta[late&(w>=.1)]+np.pi)*12/(2*np.pi)).astype(int)%12))
    qualified=bool(abs(early)>=.5 and abs(current)>=.5 and contrast>=.5 and agreement>=.5 and sectors>=9)
    signed_rate=direction*(c[0]+c[1]*last)/duration
    signed_slope=direction*c[1]/duration**2
    return dict(empty,qualified=qualified,phase_hz=abs(float(signed_rate)) if qualified else None,
        angular_harmonic=angular,phase_frequency_at_cutoff_hz=float(signed_rate),
        phase_frequency_rate_hz_s=float(signed_slope),
        phase_f2_rate_hz2_s=float(2*signed_rate*signed_slope),
        phase_reference_t_s=(data['begin']+duration*.5*1e6)/1e6,
        discovery_frequency_at_reference_hz=float(direction*c[0]/duration),
        discovery_concentration=float(abs(early)),validation_concentration=float(abs(current)),
        phase_agreement=agreement,validation_direction_power_contrast=contrast,validation_sectors=sectors,
        discovery_fit_iterations=int(result.nit),frequency_order_direction_frozen_before_validation=True,
        reason='accelerating_phase_validated' if qualified else 'accelerating_phase_not_validated')
