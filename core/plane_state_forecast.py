"""Endpoint angular state of the observed five-circle/line plane.

The old .35s linear slope describes a window average. A native-timestamp
Singer angular state estimates the present rate and acceleration instead.
Only future normal evolution changes; stationary-body and visibility gates,
current translation state, and the existing30deg future-sweep bound remain.
"""
import copy
import numpy as np
from axis_amplitude_bridge import current_axis_qualified
from conic_direction_forecast import current_uncertainty
from rotor_torque_innovation import dynamics,update,coefficients,TAU


def original_sigma(structure):
    side=structure.get('axis_support') or {}
    if structure.get('metric_axis_sigma_proxy_deg') is not None:return np.deg2rad(max(5.,float(structure['metric_axis_sigma_proxy_deg'])))
    if side.get('valid'):
        a=side.get('multiscale_spread_deg');b=side.get('line_pca_disagreement_deg')
        return np.deg2rad(max(1.,5. if a is None else float(a),5. if b is None else float(b)))
    return np.deg2rad(5./max(.45,float(structure.get('geometric_weight',.45))))


class PlaneStateForecast:
    def __init__(self,*,conic_noise=True):
        self.x=None;self.P=None;self.last_t=None;self.seen=None;self.conic_noise=conic_noise;self.epoch=None

    def step(self,row):
        t=row['t'];st=row['structure'];n=np.asarray(st.get('axis_xy') if st.get('axis_xy') is not None else [],float)
        good=current_axis_qualified(st) and n.shape==(2,) and np.isfinite(n).all() and np.linalg.norm(n)>1e-8
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing cutoffs required')
        reset=self.seen is None or t-self.seen>.12 or st.get('geometry_epoch')!=self.epoch
        noise=current_uncertainty(row);sigma=noise['sigma_rad'] if self.conic_noise and noise['qualified'] else original_sigma(st)
        if self.x is not None and not reset:
            F,Q=dynamics(t-self.last_t,3);self.x=F@self.x;self.P=F@self.P@F.T+Q
        if good:
            angle=float(np.arctan2(n[1],n[0]))
            if self.x is None or reset:
                self.x=np.array([angle,0.,0.]);bound=np.deg2rad(30);self.P=np.diag([sigma*sigma,(bound/.4)**2,(2*bound/.4**2)**2])
            else:
                x,P=update(self.x,self.P,angle,sigma)
                if x is None:
                    self.x=np.array([angle,0.,0.]);self.P=np.diag([sigma*sigma,(np.deg2rad(30)/.4)**2,(2*np.deg2rad(30)/.4**2)**2])
                else:self.x,self.P=x,P
            self.seen=t;self.epoch=st.get('geometry_epoch')
        self.last_t=t;result=copy.deepcopy(row)
        info=dict(qualified=False,current_conic_uncertainty=noise,use_current_conic_uncertainty=self.conic_noise,
            current_translation_and_force_state_unchanged=True,period_or_metric_thrust_invented=False)
        if self.x is None or not good:return dict(result,plane_state_forecast=info)
        omega=float(self.x[1]);alpha=float(self.x[2]);times=[0.,.4]
        if abs(alpha)>1e-12:
            root=1+omega/(alpha*TAU)
            if 0<root<1:
                u=-TAU*np.log(root)
                if 0<u<.4:times.append(float(u))
        def sweep(s):return omega*s+alpha*TAU*(s-TAU*(-np.expm1(-s/TAU)))
        largest=max(abs(sweep(s)) for s in times);scale=min(1.,np.deg2rad(30)/max(largest,1e-12));omega*=scale;alpha*=scale
        info.update(qualified=True,angular_rate_rad_s=omega,angular_acceleration_rad_s2=alpha,
            angular_rate_sigma_proxy=float(np.sqrt(max(0.,self.P[1,1]))),angular_acceleration_sigma_proxy=float(np.sqrt(max(0.,self.P[2,2]))),
            angular_state_time_s=t,uncertainty_calibrated=False)
        def apply(r):
            normal=r.get('normal_turn_forecast') or {}
            if normal.get('applied'):
                trend=normal.setdefault('trend',{})
                trend.update(shrunk_angular_rate_rad_s=omega,angular_acceleration_rad_s2=alpha,
                    angular_acceleration_correlation_s=TAU,reason='current_causal_plane_state',
                    angular_state_is_window_average=False,angular_state_uses_no_future_plane=True)
            for part in r.get('force',{}).get('response_model_mixture',[]):apply(part)
        apply(result);result['plane_state_forecast']=info;return result
