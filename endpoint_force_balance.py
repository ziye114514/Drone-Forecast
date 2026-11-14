"""Current endpoint acceleration constrains an otherwise unidentified force.

The curved velocity fit is a qualified estimate at the current cutoff. It
can set d = a_current - k*u_current when a missing input history prevents an
integral force balance. A newly recovered period must not replace a current
total-acceleration observation with an arbitrary disturbance initialization.
"""
import numpy as np
from projected_unknown_force import ProjectedResidualForce
from axis_amplitude_bridge import current_axis_qualified

class EndpointForceBalance(ProjectedResidualForce):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.endpoint_motion=None
    def observe_axis_motion(self,structure,motion):
        self.endpoint_motion=(structure,motion);super().observe_axis_motion(structure,motion)
    def step(self,t,c,observed_position,u,quality=0.):
        incoming=self.endpoint_motion;self.endpoint_motion=None
        predictions,info=super().step(t,c,observed_position,u,quality)
        diagnostic=dict(updated=False,reason='no_current_qualified_endpoint',gain_conditional=True,
            metric_force_measured=False,extra_position_filters=0)
        if incoming is None or not info['active'] or self.d is None or c is None:
            info['endpoint_force_balance']=diagnostic;return predictions,info
        structure,motion=incoming;a=(motion or {}).get('acceleration') or {}
        if not current_axis_qualified(structure) or a.get('reason')!='qualified_current_curve_acceleration':
            info['endpoint_force_balance']=diagnostic;return predictions,info
        if info['flow_force_balance'].get('updated'):
            info['endpoint_force_balance']=dict(diagnostic,reason='same_flow_already_used_by_integral_balance');return predictions,info
        observed=np.asarray(a['shrunk_acceleration']);se=np.asarray(a['standard_error_proxy'])
        current=np.asarray(info['acceleration_px_s2']);error=observed-current
        floor=(np.sqrt(2.)*6./(.5*.2*.2))**2
        prior=np.full(2,floor if self.rows else 3000.**2)
        if self.rows:
            residuals=np.array([r[2]-self.gain*r[1] for r in self.rows]);prior+=np.mean((residuals-self.d)**2,axis=0)
        tangent_used=(info.get('projected_unknown_force') or {}).get('updated',False)
        if tangent_used:
            n=np.asarray(structure['axis_xy']);n=n/np.linalg.norm(n)
            noise=float(np.sum(n*n*se*se));reference=float(np.sum(n*n*prior))
            correction=n*float(np.clip(n@error,-3000.,3000.)) if noise<reference else np.zeros(2)
            diagnostic.update(correction_space='remaining_normal_only',flow_variance_proxy=noise,prior_variance_proxy=reference)
        else:
            selected=np.isfinite(se)&(se*se<prior)
            correction=np.where(selected,np.clip(error,-3000.,3000.),0.)
            diagnostic.update(correction_space='current_endpoint_components',component_selected=selected.tolist(),
                flow_variance_proxy=(se*se).tolist(),prior_variance_proxy=prior.tolist())
        if np.linalg.norm(correction)>1e-10:
            self.d+=correction;a0=self.gain*np.asarray(u)+self.d
            length=np.linalg.norm(a0)
            if length>3000.:a0*=3000./length
            self.last_acceleration=a0.copy();info.update(acceleration_px_s2=a0.tolist(),disturbance_px_s2=self.d.tolist(),
                acceleration_source='qualified_current_endpoint_force_balance')
            p,v=np.asarray(c['position']),np.asarray(c['velocity']);tau=self.correlation_s
            predictions={str(h):(p+h*v+tau*(h+tau*np.expm1(-h/tau))*a0).tolist() for h in (.1,.2,.4)}
            diagnostic.update(updated=True,reason='current_endpoint_closes_unidentified_force_balance',
                correction_px_s2=correction.tolist(),unknown_force_initialized_without_matured_rows=not bool(self.rows),
                current_gain_not_reidentified_from_same_motion=True,uncertainty_is_proxy=True)
        else:diagnostic['reason']='prior_force_component_more_precise'
        info['endpoint_force_balance']=diagnostic;return predictions,info
