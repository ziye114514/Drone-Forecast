"""Optional critically damped continuation of a qualified image-net-force trend.

Only the fallback with no learned rotor gain/current rotor input is eligible.
Reuse the existing qualified quadratic-flow jerk and its reliability. This is
not RPM/thrust measurement or future pilot-command knowledge.
"""
import numpy as np


def jerk_kernel(lam,h):
    x=lam*h
    if abs(x)<1e-4:
        return h**3*(1/6+x/12+x*x/40+x*x*x/180)
    return ((x-2)*np.expm1(x)+2*x)/(lam**3)


def correct(predictions,force,motion,memory=None):
    info=dict(applied=False,reason='no_qualified_unknown_force_trend',metric_force_measured=False,
        future_controls_observed=False,new_position_filters=0)
    if force.get('active') or force.get('short_gap_acceleration_held') or force.get('gain_ever_learned'):
        return predictions,dict(info,reason='rotor_or_calibrated_force_path_preserved')
    a=(motion or {}).get('acceleration') or {};curve=a.get('curved_motion') or {}
    if not a.get('applied') or not curve.get('qualified') or (memory or {}).get('applied'):
        return predictions,info
    tau=float(force.get('acceleration_correlation_s',.35));acc=np.asarray(force['acceleration_px_s2'])
    reliability=float(curve.get('curvature_reliability',0.))
    # curve jerk is ALREADY reliability-shrunk. At full reliability this gives
    # a(s)=exp(-s/tau)*(a+(j+a/tau)*s), matching the observed initial derivative.
    trend=np.asarray(curve['jerk_px_s3'])+reliability*acc/tau
    norm=np.linalg.norm(trend);limit=3000./tau
    if norm>limit:trend*=limit/norm
    result={k:(np.asarray(p)+jerk_kernel(-1/tau,float(k))*trend).tolist() for k,p in predictions.items()}
    return result,dict(info,applied=True,reason='qualified_critically_damped_unknown_net_force',
        image_jerk_px_s3=curve['jerk_px_s3'],correction_trend_px_s3=trend.tolist(),
        curvature_reliability=reliability,correlation_s=tau)
