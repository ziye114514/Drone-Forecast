"""Observable normal-response scale from past tangential event motion.

a = A(t)n(t) + b - beta*v. Tangential projection removes every rotor's
thrust, including hidden rotors, before estimating b and beta. Only the
observable part of this fit may change the FUTURE normal-drive amplitude.
Current net acceleration and measured/unmeasured force ownership stay fixed.
The parameters are weak-perspective image responses, not Newton forces.
"""
import copy
import numpy as np
from identifiable_tangent_memory import IdentifiableTangentMemory,fit_samples
from axis_amplitude_bridge import current_axis_qualified


def fit_bias(samples):
    fit=fit_samples(samples);A=np.array([s[1] for s in samples]);y=np.array([s[2] for s in samples]);variance=np.array([s[3] for s in samples])
    B=A[:,:2]/np.sqrt(variance[:,None]);x=A[:,2]/np.sqrt(variance);z=y/np.sqrt(variance)
    if np.linalg.matrix_rank(B)<2:return dict(qualified=False,reason='normal_external_partition_unobservable')
    inverse=np.linalg.pinv(B);beta=fit['positive_response_mean_s_inv'];sigma=fit['response_sigma_proxy_s_inv']
    bias=inverse@(z-x*beta);residual=z-B@bias-x*beta
    # Keep beta/bias cross uncertainty. A finite zero-mean external-force
    # prior would incorrectly assign hidden thrust to the observable term.
    noise_scale=max(1.,float(residual@residual)/max(len(z)-3,1))
    return dict(qualified=True,bias=bias,covariance=inverse@inverse.T*noise_scale,
        bias_beta_sensitivity=inverse@x,response=beta,response_variance=sigma*sigma*noise_scale,
        residual_noise_scale=noise_scale,external_bias_data_rank=2,
        external_bias_zero_mean_prior_used=False,fit=fit)


class InverseNormalResponse:
    def __init__(self,*,calibrate=True):self.memory=IdentifiableTangentMemory();self.calibrate=calibrate;self.last_tangent_info=None;self.observations=None

    def step(self,row):
        observations=row.get('state',{}).get('observations')
        if observations is not None and self.observations is not None and observations<self.observations:
            self.memory=IdentifiableTangentMemory()
        self.observations=observations
        fit_info=self.memory.observe(row);self.last_tangent_info=fit_info;result=copy.deepcopy(row)
        info=dict(qualified=False,current_position_velocity_net_acceleration_unchanged=True,
            hidden_thrust_stays_unknown=True,mechanical_force_or_mass_identified=False,
            calibration_enabled=self.calibrate,future_force_model='drag_filtered_change_of_normal')
        if not row['coefficient'] or not current_axis_qualified(row['structure']):
            result['inverse_normal_response']=info;return result
        n=np.asarray(row['structure']['axis_xy'],float)
        # A fitted disk axis is headless. Match the existing force observer's
        # image-up sign prior before defining a nonnegative normal amplitude.
        # Flipping an ellipse normal must not enable/disable calibration.
        if n[1]>0:n=-n
        energy=float(n@n)
        if energy<1e-8:result['inverse_normal_response']=info;return result
        fit=fit_bias(self.memory.samples) if fit_info.get('qualified') else dict(qualified=False)
        details=[]
        def apply(packet):
            f=packet['force'];normal=packet.get('normal_turn_forecast') or {}
            if normal.get('applied') and f.get('normal_turn_vector_px_s2') is not None:
                old=np.asarray(f['normal_turn_vector_px_s2']);drive=old.copy();item=dict(applied=False)
                if self.calibrate and fit['qualified']:
                    c=packet['coefficient'];a=np.asarray(c['acceleration']);v=np.asarray(c['velocity']);direction=n/energy
                    # a and v are the existing present state, not new truth.
                    amplitude=float(direction@(a-fit['bias']+fit['response']*v))
                    sensitivity=fit['bias_beta_sensitivity']+v
                    var=float(direction@fit['covariance']@direction+fit['response_variance']*(direction@sensitivity)**2)
                    reliability=amplitude*amplitude/(amplitude*amplitude+max(var,1.)) if amplitude>0 else 0.
                    drive=(1-reliability)*old+reliability*amplitude*n
                    item=dict(applied=bool(reliability>0),normal_response_estimate_px_s2=amplitude,
                        normal_response_sigma_proxy_px_s2=float(np.sqrt(max(var,0.))),reliability=float(reliability),
                        current_thrust_measured=False,external_bias=fit['bias'].tolist(),
                        external_bias_covariance=fit['covariance'].tolist(),response_s_inv=fit['response'],
                        current_motion_not_independent_of_fit=True)
                # Same old omega and force decay. The derivative of a rotating
                # force drives acceleration through drag; force itself is not
                # exponentially rotated as though it were a decaying impulse.
                f['normal_turn_vector_px_s2']=drive.tolist();f['future_normal_response_model']='drag_filtered_normal_derivative'
                f['future_normal_response_calibration']=item;details.append(item)
            for part in f.get('response_model_mixture',[]):apply(part)
        apply(result);info.update(qualified=bool(details),components=details,
            external_partition_observable=fit['qualified'],no_external_bias_injected_into_current_state=True)
        result['inverse_normal_response']=info;return result


def normal_drive_kernel(lam,h,omega,alpha=0.,tau=None):
    """For da/dt=lam*a+A*d(n)/dt and n(s)=exp(i*omega*s)n0."""
    from adaptive_force_memory import kernel
    if alpha:
        nodes,weights=_NODES,_WEIGHTS;s=(nodes+1)*h/2
        if tau:
            angle=omega*s+alpha*tau*(s-tau*(-np.expm1(-s/tau)));rate=omega+alpha*tau*(-np.expm1(-s/tau))
        else:angle=omega*s+.5*alpha*s*s;rate=omega+alpha*s
        values=np.array([kernel(lam,h-u) for u in s]);drive=np.sum(weights*values*1j*rate*np.exp(1j*angle))*h/2
        return kernel(lam,h)+drive
    if abs(omega)<1e-12:return kernel(lam,h)
    return kernel(lam,h)+1j*omega/(1j*omega-lam)*(kernel(1j*omega,h)-kernel(lam,h))

_NODES,_WEIGHTS=np.polynomial.legendre.leggauss(24)
