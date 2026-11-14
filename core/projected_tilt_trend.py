"""Forecast inclination as well as azimuth of the observed rotor plane.

Under the existing weak-perspective circle model ||n_xy|| = sin(tilt).
Azimuth-only rollout discards its time variation. This module only learns
tilt rate from CURRENT qualified conics, never from a whole-body axis or
unseen rotors. Sign of camera-depth normal remains ambiguous/unreported.
"""
from collections import deque
import numpy as np
from axis_amplitude_bridge import current_axis_qualified


class ProjectedTiltTrend:
    def __init__(self):self.rows=deque();self.epoch=None;self.last_t=None
    def observe(self,t,structure,geometry):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing source time required')
        epoch=structure.get('geometry_epoch')
        if (self.last_t is not None and t-self.last_t>.12+1e-8) or (self.epoch is not None and epoch!=self.epoch):self.rows.clear()
        self.last_t=t;self.epoch=epoch
        info=dict(qualified=False,reason='no_current_inclination_measurement',heading_used=False,
            hidden_frequencies_created=0,weak_perspective=True,depth_sign_resolved=False)
        rotors=[r for r in geometry.get('rotors',[]) if r.get('current_observation',True) and not r.get('fov_clipped') and
            (r.get('mode')=='ellipse' or r.get('curvature_validated') or r.get('arc_center_qualified'))]
        # An unresolved body line is an azimuth, not a measured90degree plane.
        if len(rotors)<2 or not current_axis_qualified(structure):return info
        if (structure.get('axis_support') or {}).get('valid'):
            return dict(info,reason='body_axis_does_not_measure_plane_inclination')
        n=structure.get('axis_xy')
        if n is None:return info
        n=np.asarray(n,float);s=float(np.linalg.norm(n))
        if n.shape!=(2,) or not np.isfinite(n).all() or not 1e-6<s<=1+1e-8:return info
        theta=float(np.arcsin(np.clip(s,0,1)))
        # Reuse the original azimuth noise floor and observation confidence.
        sigma=np.deg2rad(5./max(.45,float(structure.get('geometric_weight',.45))))
        if self.rows and (t-self.rows[-1][0]>.12+1e-8 or abs(theta-self.rows[-1][1])>np.deg2rad(30)):self.rows.clear()
        self.rows.append((t,theta,sigma))
        while self.rows and t-self.rows[0][0]>.35+1e-8:self.rows.popleft()
        a=np.asarray(self.rows)
        if len(a)<6 or t-a[0,0]<.16-1e-8:return dict(info,reason='inclination_rate_warmup')
        tt=a[:,0]-t
        if max(np.diff(tt))>.08+1e-8:return dict(info,reason='inclination_observation_gap')
        w=1/a[:,2]**2;w/=w.sum();mt=float(w@tt);ma=float(w@a[:,1]);energy=float(w@(tt-mt)**2)
        rate=float(w@((tt-mt)*(a[:,1]-ma))/max(energy,1e-12))
        residual=a[:,1]-(ma+rate*(tt-mt));noise=max(float(np.sqrt(w@residual**2)),float(np.median(a[:,2])))
        rate_sigma=noise/max(float(np.ptp(tt)),.08)
        trust=rate*rate/(rate*rate+rate_sigma*rate_sigma) if rate else 0.
        rate=float(np.clip(rate*trust,-np.deg2rad(30)/.4,np.deg2rad(30)/.4))
        if theta+min(0.,rate)*.4<0:
            return dict(info,reason='front_normal_sign_ambiguous',inclination_rad=theta)
        return dict(info,qualified=True,reason='current_conic_inclination_rate',inclination_rad=theta,
            inclination_rate_rad_s=rate,inclination_rate_sigma_proxy=rate_sigma,
            inclination_rate_reliability=trust,observations=len(a),span_s=float(np.ptp(tt)),
            maximum_future_sweep_deg=30.,uncertainty_calibrated=False)


NODES,WEIGHTS=np.polynomial.legendre.leggauss(16)


def projected_normal_kernel(lam,h,omega,theta,rate):
    """Complex kernel relative to the CURRENT projected normal vector.

sin(theta+rate*s) keeps the projection within a unit3D normal, including
continuation through an edge-on view. It never multiplies the external force.
"""
    if abs(np.sin(theta))<1e-6:raise ValueError('Projected normal is degenerate')
    s=(NODES+1)*h/2
    relative=np.sin(theta+rate*s)/np.sin(theta)
    return complex(np.sum(WEIGHTS*(h-s)*np.exp(lam*s+1j*omega*s)*relative)*h/2)
