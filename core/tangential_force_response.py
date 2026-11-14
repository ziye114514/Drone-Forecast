"""Conditional image-plane drag response from a qualified headless plane.

For a=T*n+d, q perpendicular to n eliminates the unknown thrust amplitude.
Over a locally stable plane, fit q*a=b-beta*q*v with beta>=0. The constant
wind/gravity contribution is lumped into b. This is an effective 2D response,
not a measured aerodynamic coefficient. Reuses native image motion only.
Physics motivation: Faessler et al., RAL 2018, arXiv:1712.02402, Eq. 2.
"""
from collections import deque
import numpy as np
from adaptive_force_memory import kernel
from axis_amplitude_bridge import current_axis_qualified


def fit_response(rows,q):
    x=np.array([q@r['v'] for r in rows]);y=np.array([q@r['a'] for r in rows])
    noise=np.array([np.sum(q*q*r['se']**2) for r in rows])
    w=1/np.maximum(noise,1.);w/=w.sum()
    meanx=float(w@x);meany=float(w@y);excitation=float(w@((x-meanx)**2))
    beta=-float(w@((x-meanx)*(y-meany)))/max(excitation,1e-12)
    b=meany+beta*meanx;residual=y-b+beta*x
    # Do not gain fictitious precision by averaging overlapping flow fits.
    scatter=max(float(w@noise),float(w@(residual*residual)))
    se=np.sqrt(scatter/max(excitation,1e-12))
    return beta,b,se,residual,noise


class TangentialForceResponse:
    def __init__(self):self.rows=deque();self.last_t=None

    def step(self,t,c,predictions,force,structure,motion,normal=None,memory=None):
        if self.last_t is not None and (t<=self.last_t or t-self.last_t>.5):
            if t<=self.last_t:raise ValueError('Increasing cutoffs required')
            self.rows.clear()
        self.last_t=t
        result={k:list(v) for k,v in predictions.items()}
        info=dict(applied=False,reason='no_current_qualified_plane_motion',information_rank=1,
            normal_force_unchanged=True,metric_drag_identified=False,no_new_motion_filter=True)
        while self.rows and t-self.rows[0]['t']>.5+1e-8:self.rows.popleft()
        a=(motion or {}).get('acceleration') or {}
        observation=(motion or {}).get('observation') or {}
        if not current_axis_qualified(structure) or c is None or not a.get('applied') or not observation.get('valid'):
            return result,info
        n=np.asarray(structure['axis_xy']);n=n/np.linalg.norm(n);q=np.array([-n[1],n[0]])
        self.rows.append(dict(t=t,n=n.copy(),v=np.asarray(c['velocity']),
            a=np.asarray(a['shrunk_acceleration']),se=np.asarray(a['standard_error_proxy'])))
        if len(self.rows)<8 or self.rows[-1]['t']-self.rows[0]['t']<.23-1e-8:
            return result,dict(info,reason='tangent_response_warmup',rows=len(self.rows))
        stamps=np.array([r['t'] for r in self.rows]);normals=np.array([r['n'] for r in self.rows])
        spread=float(np.degrees(np.arccos(np.clip(abs(normals@n),0,1))).max())
        confidence=float((structure.get('axis_support') or {}).get('confidence',structure.get('geometric_weight',.45)))
        limit=5./max(.45,min(1.,confidence))
        if np.max(np.diff(stamps))>.08+1e-8 or spread>limit:
            return result,dict(info,reason='plane_or_motion_not_stable',axis_spread_deg=spread)
        rows=list(self.rows);beta,b,se,residual,noise=fit_response(rows,q)
        info.update(beta_per_s=beta,beta_standard_error_proxy=float(se),
            lumped_tangent_offset_px_s2=b,axis_spread_deg=spread,rows=len(rows))
        if not 0.<beta<=30. or beta<2.*se:
            return result,dict(info,reason='positive_relaxation_not_identifiable')
        if np.mean(abs(residual)<=2*np.sqrt(np.maximum(noise,1.)))<.9:
            return result,dict(info,reason='constant_tangent_disturbance_not_supported')
        oldbeta,oldb,_,_,_=fit_response(rows[:-2],q)
        vx=np.array([q@r['v'] for r in rows[-2:]]);ay=np.array([q@r['a'] for r in rows[-2:]])
        prediction=oldb-oldbeta*vx
        constant=np.mean([q@r['a'] for r in rows[:-2]])
        if oldbeta<=0. or np.sum((ay-prediction)**2)>=np.sum((ay-constant)**2):
            return result,dict(info,reason='relaxation_does_not_predict_recent_forces_better')
        trend=(normal or {}).get('trend') or {}
        omega=float(trend.get('shrunk_angular_rate_rad_s',0.)) if trend.get('qualified') else 0.
        if abs(omega)*.4>np.radians(limit):
            return result,dict(info,reason='future_plane_sweep_exceeds_tangent_model')
        # Shrink the identified rate towards the existing response prior.
        prior=1/float(force.get('acceleration_correlation_s',.35))
        reliability=beta*beta/(beta*beta+se*se)
        effective=prior+reliability*(beta-prior)
        long=float((memory or {}).get('weight',0.));short=float((memory or {}).get('short_weight',0.))
        total=np.asarray(force.get('acceleration_px_s2',[0.,0.]));tangent=float(q@total)
        changes={}
        for key,p in predictions.items():
            h=float(key)
            old=(1-long-short)*kernel(-prior,h)+long*kernel(0.,h)+short*kernel(-10.,h)
            correction=q*tangent*(kernel(-effective,h)-old)
            result[key]=(np.asarray(p)+correction).tolist();changes[key]=correction.tolist()
        return result,dict(info,applied=True,reason='qualified_plane_tangent_force_response',
            effective_beta_per_s=float(effective),reliability=float(reliability),corrections_px=changes,
            current_RPM_invented=False,body_heading_used=False)
