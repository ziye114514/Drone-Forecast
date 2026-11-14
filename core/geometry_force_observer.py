"""Conditional rank-two image force inference without measured rotor rates.

For a_i = T_i*n_i + d, a tangent q_i eliminates arbitrary T_i. Several
nonparallel, current plane normals can identify a locally constant image d.
This is conditional inverse dynamics, not a true-thrust/wind/RPM sensor.
Uses endpoint motion acceleration, not a past-window slope as today's value.
"""
from collections import deque
import numpy as np
from adaptive_force_memory import kernel
from axis_amplitude_bridge import current_axis_qualified


def projected_fit(rows,age_weight=False):
    n=np.array([r['n'] for r in rows]);q=np.column_stack((-n[:,1],n[:,0]))
    a=np.array([r['a'] for r in rows]);se=np.array([r['se'] for r in rows])
    angles=np.array([r['angle_sigma'] for r in rows]);y=np.sum(q*a,axis=1)
    d=np.zeros(2);info={}
    for _ in range(4):
        amplitude=np.sum(n*(a-d),axis=1)
        variance=np.sum(q*q*se*se,axis=1)+angles*angles*amplitude*amplitude
        variance=np.maximum(variance,1.)
        residual=y-q@d;robust=np.minimum(1.,(8.07/6.)/np.maximum(abs(residual)/np.sqrt(variance),1e-12))
        w=robust/variance
        if age_weight:w*=np.exp(-(rows[-1]['t']-np.array([r['t'] for r in rows]))/.35)
        w/=w.sum()
        information=(q.T*w)@q;values=np.linalg.eigvalsh(information)
        rank=int(np.sum(values>max(values[-1]*1e-4,1e-12)))
        info=dict(data_information_rank=rank,information_eigenvalues=values.tolist(),
            information_condition=float(values[-1]/max(values[0],1e-12)))
        if rank<2 or info['information_condition']>1500.:
            return None,None,dict(info,reason='plane_changes_do_not_identify_two_unknown_components')
        d=np.linalg.solve(information,q.T@(w*y))
    residual=y-q@d
    inliers=float(np.mean(abs(residual)<=2.*np.sqrt(variance)))
    if inliers<.9:
        return None,None,dict(info,reason='locally_constant_unknown_force_not_supported',inlier_fraction=inliers)
    # Overlapping endpoint fits are correlated: use a common variance and
    # normalized information, never an invented independent sample-count gain.
    scale=max(float(w@variance),float(w@(residual*residual)))
    covariance=scale*np.linalg.inv(information)
    return d,covariance,dict(info,reason='conditional_geometry_force_fit',inlier_fraction=inliers,
        disturbance_px_s2=d.tolist(),disturbance_covariance_proxy=covariance.tolist(),
        axis_uncertainty_included=True,no_independence_gain_from_overlapping_flow_fits=True)


class GeometryForceObserver:
    def __init__(self,irregular=False):self.rows=deque();self.last_t=None;self.irregular=irregular

    def step(self,t,c,predictions,force,structure,motion,normal=None,memory=None):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing cutoffs required')
        self.last_t=t
        while self.rows and t-self.rows[0]['t']>.5+1e-8:self.rows.popleft()
        result={k:list(v) for k,v in predictions.items()}
        info=dict(qualified=False,applied=False,reason='no_current_endpoint_and_plane',
            metric_thrust_measured=False,current_RPM_invented=False,heading_used=False,
            current_position_velocity_unchanged=True)
        endpoint=(motion or {}).get('acceleration') or {};trend=(normal or {}).get('trend') or {}
        if (c is None or not current_axis_qualified(structure) or
            endpoint.get('reason')!='qualified_current_curve_acceleration' or not trend.get('qualified')):
            return result,info
        n=np.asarray(structure['axis_xy'],float);n/=np.linalg.norm(n)
        if n[1]>0:n=-n
        # Body-line -> force-plane is approximate, even with a tiny line fit
        # residual. Keep a5degree floor for that systematic approximation.
        minimum=5. if (structure.get('axis_support') or {}).get('valid') else 1.
        angle_sigma=max(np.radians(minimum),float(trend.get('angle_sigma_proxy_rad',np.radians(5.))))
        a=np.asarray(endpoint['shrunk_acceleration']);se=np.asarray(endpoint['standard_error_proxy'])
        self.rows.append(dict(t=t,n=n.copy(),a=a.copy(),se=se.copy(),angle_sigma=angle_sigma))
        minimum=4 if self.irregular else 8;span=.16 if self.irregular else .23
        if len(self.rows)<minimum or self.rows[-1]['t']-self.rows[0]['t']<span-1e-8:
            return result,dict(info,reason='force_observer_warmup',rows=len(self.rows))
        stamps=np.array([r['t'] for r in self.rows])
        if np.max(np.diff(stamps))>(.2 if self.irregular else .08)+1e-8:
            return result,dict(info,reason='endpoint_history_gap',rows=len(self.rows))
        d,C,fit=projected_fit(list(self.rows),age_weight=self.irregular);info.update(fit,rows=len(self.rows),
            irregular_endpoint_update=self.irregular)
        if d is None:return result,info
        amplitude=float(n@(a-d));q=np.array([-n[1],n[0]])
        variance=float(n@(C+np.diag(se*se))@n+(angle_sigma*float(q@(a-d)))**2)
        sigma=float(np.sqrt(max(variance,0.)))
        info.update(inferred_image_thrust_px_s2=amplitude,thrust_standard_error_proxy=sigma,
            inference_source='current_motion_and_temporal_plane_excitation_not_rotor_period')
        if amplitude<=2.*max(sigma,1.):
            return result,dict(info,reason='inferred_amplitude_not_resolved_above_uncertainty')
        info['qualified']=True
        if force.get('active'):
            return result,dict(info,reason='current_event_input_preserved')
        omega=float(trend['shrunk_angular_rate_rad_s']);tau=float(force.get('acceleration_correlation_s',.35))
        long=float((memory or {}).get('weight',0.));short=float((memory or {}).get('short_weight',0.))
        known=complex(*(amplitude*n));changes={}
        for key,p in predictions.items():
            h=float(key)
            rotation=lambda real:kernel(real+1j*omega,h)-kernel(real,h)
            delta=((1-long-short)*rotation(-1/tau)+long*rotation(0.)+short*rotation(-10.))*known
            change=np.array([delta.real,delta.imag]);result[key]=(np.asarray(p)+change).tolist();changes[key]=change.tolist()
        return result,dict(info,applied=True,reason='conditional_inferred_thrust_plane_trend',
            unknown_force_not_rotated=True,corrections_px=changes,current_event_frequency_still_missing=True)
