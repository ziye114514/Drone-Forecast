"""Qualified native-circle expansion conditions a future projection prior.

No body-box scale, metric depth, hidden RPM, extra current acceleration, or
motor identity. Future depth controls remain unobserved. The engineering
circle-size uncertainty is correlated scatter, not a calibrated covariance.
"""
from collections import deque
import copy
import numpy as np
from conic_observability import usable

NODES,WEIGHTS=np.polynomial.legendre.leggauss(16)


class RotorScaleTrend:
    def __init__(self):self.rows=deque();self.observations=None
    def observe(self,row):
        t=float(row['t']);obs=row.get('state',{}).get('observations')
        if obs is not None and self.observations is not None and obs<self.observations:self.rows.clear()
        self.observations=obs
        rotors=[r for r in row['geometry'].get('rotors',[]) if usable(r)] if row['body'].get('detected') else []
        info=dict(qualified=False,current_native_conics=len(rotors),body_bbox_size_used=False,
            metric_depth_measured=False,rotor_identity_used=False)
        if len(rotors)<2:return dict(info,reason='fewer_than_two_current_size_qualified_conics')
        sizes=np.array([r['major'] for r in rotors]);logs=np.log(sizes);value=float(np.median(logs))
        # An engineering size-error proxy from fit localisation, with the
        # observed between-conic disagreement retained instead of divided byN.
        sigma=max(float(np.median([2*r.get('center_sigma_px',3.)/r['major'] for r in rotors])),
            float(1.4826*np.median(abs(logs-value))),1e-6)
        if self.rows and t-self.rows[-1][0]>.12+1e-8:self.rows.clear()
        self.rows.append((t,value,sigma))
        while self.rows and t-self.rows[0][0]>.35+1e-8:self.rows.popleft()
        a=np.asarray(self.rows)
        if len(a)<6 or t-a[0,0]<.16-1e-8:return dict(info,reason='scale_rate_warmup')
        tt=a[:,0]-t
        if np.max(np.diff(tt))>.08+1e-8:return dict(info,reason='scale_measurement_gap')
        w=1/a[:,2]**2;w/=w.sum();mt=float(w@tt);ma=float(w@a[:,1]);energy=float(w@(tt-mt)**2)
        rate=float(w@((tt-mt)*(a[:,1]-ma))/max(energy,1e-12));residual=a[:,1]-(ma+rate*(tt-mt))
        noise=max(float(np.sqrt(w@residual**2)),float(np.median(a[:,2])))
        se=noise/max(float(np.ptp(tt)),.08);trust=rate*rate/(rate*rate+se*se) if rate else 0.;used=rate*trust
        denominator=1-used*.4
        if denominator<=0 or not .6<=1/denominator<=1.6:
            return dict(info,reason='future_scale_outside_existing_teacher_domain',raw_log_scale_rate_s_inv=rate)
        return dict(info,qualified=True,reason='current_circle_major_scale_rate',log_scale_rate_s_inv=used,
            raw_log_scale_rate_s_inv=rate,rate_sigma_proxy_s_inv=se,reliability=trust,
            predicted_relative_scale04=1/denominator,observations=len(a),span_s=float(np.ptp(tt)),
            size_uncertainty_is_correlated_proxy=True)


def increment(acceleration,rate,h):
    h=float(h);a=np.asarray(acceleration,float)
    if h==0 or rate==0:return np.zeros(2)
    s=(NODES+1)*h/2
    if min(1-rate*s)<=0:return np.zeros(2)
    gain=np.sum(WEIGHTS*(h-s)*np.exp(-s/.35)*((1-rate*s)**-3-1))*h/2
    return gain*a


def apply(row,trend,coefficient):
    r=copy.deepcopy(row)
    if not r.get('coefficient') or not trend.get('qualified') or coefficient==0:return r
    r['rotor_scale_response']=dict(trend,coefficient=float(coefficient),current_p_v_a_unchanged=True,
        present_measured_hidden_force_ownership_unchanged=True,future_projection_prior=True)
    from physical_publication import evaluate
    r['physical']=evaluate(r,[.1,.2,.4]);return r


def predict(row,horizons):
    from physical_publication import evaluate
    r=copy.deepcopy(row);info=r.pop('rotor_scale_response');values=evaluate(r,horizons)
    return {str(float(h)):(np.asarray(values[str(float(h))])+float(info['coefficient'])*increment(r['coefficient']['acceleration'],info['log_scale_rate_s_inv'],h)).tolist() for h in horizons}
