"""Conditional common image-force gain with local velocity nuisance removal.

Fit v = v0 + k*integral(u) + d*t on the SAME qualified past interval.
Unknown initial velocity and constant lumped force are removed before the
scalar gain solve. This is ordinary whitened least squares, not a new physical
sensor; confidence remains a proxy under correlated image observations.
"""
import numpy as np
from flow_force_balance import FlowBalancedForce,input_integrals


def fit_gain(velocities,inputs,maximum_gain=5000.):
    integral,info=input_integrals(velocities,inputs)
    if integral is None:return None,info
    stamps=np.array([r[0] for r in velocities]);span=float(np.ptp(stamps))
    if span<.16-1e-8 or np.max(np.diff(stamps))>.08+1e-8:return None,dict(info,reason='short_or_gapped_flow_velocity_window')
    times=(stamps-stamps[-1])/span;design=np.column_stack((np.ones(len(times)),times))
    A=np.kron(design,np.eye(2)).reshape(len(times),2,4)
    W=np.array([r[3] if len(r)>3 else np.linalg.inv(np.linalg.cholesky(r[2])) for r in velocities])
    A=(W@A).reshape(-1,4);U=(W@integral[:,:,None]).ravel()
    V=(W@np.array([r[1] for r in velocities])[:,:,None]).ravel()
    Q=np.linalg.qr(A,mode='reduced')[0]
    independent_u=U-Q@(Q.T@U);independent_v=V-Q@(Q.T@V)
    energy=float(independent_u@independent_u)
    if energy<1e-12:return None,dict(info,reason='gain_indistinguishable_from_constant_disturbance')
    gain=float(independent_u@independent_v/energy)
    residual=independent_v-gain*independent_u
    variance=max(1.,float(residual@residual)/max(1,len(V)-5));sigma=float(np.sqrt(variance/energy))
    median=float(np.median(abs(residual)));inliers=float(np.mean(abs(residual)<=8.07/6.))
    accepted=bool(0.<gain<maximum_gain and gain>=2*sigma and median<=1. and inliers>=.9)
    return gain if accepted else None,dict(info,reason='qualified_local_gain' if accepted else 'gain_uncertain_or_velocity_residual_inconsistent',
        candidate_gain_px_s2=gain,standard_error_proxy=sigma,gain_signal_to_noise=gain/max(sigma,1e-9),
        independent_input_energy=energy,median_standardized_residual=median,robust_inlier_fraction=inliers,
        constant_input_cannot_identify_gain=True,uncertainty_is_not_independent_truth=True)


class FlowIdentifiableForce(FlowBalancedForce):
    def __init__(self,default_gain=200.,early_disturbance=True,hold_acceleration_gaps=True,gain_signal_to_noise=0.):
        super().__init__(default_gain,early_disturbance,hold_acceleration_gaps,gain_signal_to_noise)
        self.gain_learning_enabled=False

    def step(self,t,c,observed_position,u,quality=0.):
        previous=self.last_t
        predictions,info=super().step(t,c,observed_position,u,quality)
        diagnostic=dict(updated=False,reason='no_current_qualified_force_and_flow_balance')
        if info['active'] and info['flow_force_balance'].get('updated'):
            candidate,diagnostic=fit_gain(self.flow_velocities,self.balance_inputs)
            if candidate is not None:
                dt=0. if previous is None else t-previous;new=self.gain+(1-np.exp(-dt/1.))*(candidate-self.gain)
                self.d-=(new-self.gain)*np.asarray(u);self.gain=new;self.learned=True
                self.gain_sigma=diagnostic['standard_error_proxy'];self.frozen_reason=None
                info.update(effective_gain_px_s2=self.gain,gain_learned=True,gain_ever_learned=True,
                    gain_sigma=self.gain_sigma,disturbance_px_s2=self.d.tolist(),frozen_reason=None)
                diagnostic.update(updated=True,known_gain_and_disturbance_reparameterized_for_current_acceleration_continuity=True)
        info['local_flow_gain']=diagnostic
        return predictions,info
