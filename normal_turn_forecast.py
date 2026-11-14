"""Headless disk-normal trend, applied only to the known thrust contribution.

No rotor IDs, yaw/heading or second position filter. Existing net-acceleration
Singer forecast is retained; only the change in the known input's direction
is integrated. Unknown-force/gravity terms are not rotated with the body.
"""
from collections import deque
import numpy as np
from axis_amplitude_bridge import current_axis_qualified


class NormalTurnForecast:
    def __init__(self):
        self.rows=deque();self.last_t=None
        self.axis_sigma_override_rad=None
    def observe(self,t,structure,coefficient,motion=None):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing time required')
        if self.last_t is not None and t-self.last_t>.12:self.rows.clear()
        self.last_t=t
        info=dict(qualified=False,reason='no_current_plane_axis',heading_used=False,rotor_ids_used=False)
        n=np.asarray(structure.get('axis_xy') if structure.get('axis_xy') is not None else [],float)
        if not current_axis_qualified(structure) or n.shape!=(2,) or not np.isfinite(n).all() or np.linalg.norm(n)<1e-8:return info
        angle=float(np.arctan2(n[1],n[0]))
        if self.rows:
            # A plane normal and its negative describe the same disk.
            angle=self.rows[-1][1]+.5*float(np.angle(np.exp(2j*(angle-self.rows[-1][1]))))
        side=structure.get('axis_support') or {}
        if structure.get('metric_axis_sigma_proxy_deg') is not None:
            sigma=np.deg2rad(max(5.,float(structure['metric_axis_sigma_proxy_deg'])))
        elif side.get('valid'):
            spread=side.get('multiscale_spread_deg');disagreement=side.get('line_pca_disagreement_deg')
            sigma=np.deg2rad(max(1.,5. if spread is None else float(spread),
                5. if disagreement is None else float(disagreement)))
        else:sigma=np.deg2rad(5./max(.45,float(structure.get('geometric_weight',.45))))
        if self.axis_sigma_override_rad is not None:
            if not np.isfinite(self.axis_sigma_override_rad) or self.axis_sigma_override_rad<=0:raise ValueError('Positive current angle uncertainty required')
            sigma=float(self.axis_sigma_override_rad)
        if self.rows and abs(angle-self.rows[-1][1])>np.deg2rad(30):
            self.rows.clear();info['reason']='axis_jump_reinitializes_trend'
        self.rows.append((t,angle,sigma))
        while self.rows and t-self.rows[0][0]>.35+1e-8:self.rows.popleft()
        if len(self.rows)<6 or t-self.rows[0][0]<.16-1e-8:return dict(info,reason='angle_trend_warmup')
        a=np.asarray(self.rows);tt=a[:,0]-t
        if np.max(np.diff(tt))>.08+1e-8:return dict(info,reason='axis_gap')
        weights=1/np.square(a[:,2]);weights/=weights.sum()
        mt=float(weights@tt);ma=float(weights@a[:,1]);energy=float(weights@np.square(tt-mt))
        rate=float(weights@((tt-mt)*(a[:,1]-ma))/max(energy,1e-12))
        residual=a[:,1]-(ma+rate*(tt-mt))
        noise=max(float(np.sqrt(weights@np.square(residual))),float(np.median(a[:,2])))
        span=float(np.ptp(tt));rate_sigma=noise/max(span,.08)
        # Correlated axis errors: do not divide scatter by a fictitious sample count.
        reliability=rate*rate/(rate*rate+rate_sigma*rate_sigma) if rate else 0.
        shrunk=rate*reliability
        maximum=np.deg2rad(30)/.4
        info.update(reason='current_headless_plane_trend',angular_rate_rad_s=rate,
            angular_rate_sigma_proxy=rate_sigma,rate_reliability=reliability,
            shrunk_angular_rate_rad_s=float(np.clip(shrunk,-maximum,maximum)),
            maximum_future_sweep_deg=30.,rows=len(a),span_s=span,
            angle_sigma_proxy_rad=noise,uncertainty_calibrated=False)
        if coefficient is None:return dict(info,reason='no_position_state')
        velocity=np.asarray(coefficient['velocity'],float)
        observation=(motion or {}).get('observation') or {}
        acceleration=(motion or {}).get('acceleration') or {}
        measured_velocity=np.asarray(observation.get('velocity',[0.,0.]),float)
        moving=bool(observation.get('valid') and np.linalg.norm(measured_velocity)>=6./.1)
        if acceleration.get('applied'):
            value=np.asarray(acceleration['shrunk_acceleration']);se=np.asarray(acceleration['standard_error_proxy'])
            moving|=bool(np.linalg.norm(value)>=2*max(np.linalg.norm(se),1.))
        if not moving:return dict(info,reason='stationary_or_unconfirmed_translation_protects_wobble')
        info['qualified']=True
        return info
    @staticmethod
    def correct(predictions,force,trend):
        result={k:list(v) for k,v in predictions.items()}
        info=dict(applied=False,reason=trend.get('reason'),trend=trend,
            only_known_input_rotated=True,unknown_force_rotated=False)
        u=force.get('input_u')
        if not predictions or not trend.get('qualified') or not force.get('active') or u is None:return result,info
        rate=trend['shrunk_angular_rate_rad_s'];tau=float(force.get('acceleration_correlation_s',.35))
        known=float(force['effective_gain_px_s2'])*np.asarray(u,float);z=complex(*known)
        lam=-1/tau+1j*rate;real=-1/tau
        changes={}
        for key,p in predictions.items():
            h=float(key)
            kernel=(np.expm1(lam*h)-lam*h)/(lam*lam)
            base=(np.expm1(real*h)-real*h)/(real*real)
            delta=(kernel-base)*z;d=np.array([delta.real,delta.imag])
            result[key]=(np.asarray(p)+d).tolist();changes[key]=d.tolist()
        return result,dict(info,applied=True,reason='known_thrust_normal_evolution',
            corrections_px=changes,correlation_s=tau)
