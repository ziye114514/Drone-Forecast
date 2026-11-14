"""Qualified current acceleration when the recent velocity curve is bending.

The linear flow derivative estimates a window-average acceleration. This
optional quadratic fit estimates its current endpoint with continuous
uncertainty shrinkage and an improvement in held-out recent velocities.
It estimates total image acceleration, not thrust.
"""
import copy
import numpy as np
from scipy.optimize import least_squares
from fast_huber import fit as huber_fit
from live_flow_acceleration import FlowAcceleration
from live_motion import cap

def _fit(rows,t,span,degree,covariance=True):
    times=np.array([r[0] for r in rows]);x=(times-t)/span
    design=np.column_stack((np.ones(len(x)),x,.5*x*x))[:,:degree+1]
    jac=np.kron(design,np.eye(2)).reshape(len(rows),2,2*(degree+1))
    whiten=np.array([r[3] for r in rows]);J=(whiten@jac).reshape(-1,2*(degree+1))
    target=(whiten@np.array([r[1] for r in rows])[:,:,None]).ravel()
    initial=np.linalg.lstsq(J,target,rcond=None)[0]
    residual=J@initial-target
    if np.all(abs(residual)<=8.07/6.) and np.max(abs(J.T@residual))<1e-8:
        parameters=initial;success=True
    else:
        parameters,success,_=huber_fit(J,target,initial)
        residual=J@parameters-target
    if not covariance:return parameters.reshape(-1,2),None,residual,success
    weights=np.minimum(1.,(8.07/6.)/np.maximum(abs(residual),1e-12))
    scatter=max(1.,1.4826*float(np.median(abs(residual-np.median(residual)))))
    C=scatter*scatter*np.linalg.pinv(J.T@(weights[:,None]*J))
    return parameters.reshape(-1,2),C,residual,success

def curved_derivative(rows,t,linear_info=None):
    info=dict(qualified=False,reason='curved_flow_warmup',degree=2)
    if len(rows)<8:return None,info
    stamps=np.array([r[0] for r in rows]);span=float(np.ptp(stamps))
    if span<.23-1e-8 or np.max(np.diff(stamps))>.08+1e-8:return None,info
    beta,C,residual,success=_fit(rows,t,span,2)
    if not success or np.mean(abs(residual)<=8.07/6.)<.9 or np.median(abs(residual))>1.:
        return None,dict(info,reason='curved_flow_residual_not_supported')
    jerk=beta[2]/span**2;sej=np.sqrt(np.maximum(np.diag(C)[4:],0.))/span**2
    snr=float(np.linalg.norm(jerk)/max(np.linalg.norm(sej),1.))
    info.update(jerk_px_s3=jerk.tolist(),jerk_standard_error_proxy=sej.tolist(),jerk_signal_to_noise=snr)
    # A nested continuous update retains the linear model as the zero-weight
    # endpoint. Weak curvature contributes proportionally to its uncertainty;
    # it is not promoted to a fully observed jerk by lowering a hard gate.
    reliability=snr*snr/(1.+snr*snr)
    info['curvature_reliability']=reliability
    # Both models see the same earlier velocities and predict the same final
    # two observations. Do not accept a quadratic merely for in-sample fit.
    train=list(rows)[:-2];test=list(rows)[-2:];loss=[]
    for degree in (1,2):
        b,_,_,ok=_fit(train,t,span,degree,covariance=False)
        if not ok:return None,dict(info,reason='curvature_training_failed')
        x=(np.array([r[0] for r in test])-t)/span
        A=np.column_stack((np.ones(len(x)),x,.5*x*x))[:,:degree+1]
        e=A@b-np.array([r[1] for r in test]);W=np.array([r[3] for r in test])
        loss.append(float(np.sum(np.square(W@e[:,:,None]))))
    info.update(held_out_linear_loss=loss[0],held_out_curved_loss=loss[1],held_out_velocity_rows=2)
    if loss[1]>=loss[0]:return None,dict(info,reason='curvature_does_not_predict_recent_velocities_better')
    reuse=bool(linear_info is not None and all(k in linear_info for k in
        ('acceleration','standard_error_proxy','fitted_current_velocity')))
    if reuse:
        linear=np.array([linear_info['fitted_current_velocity'],np.asarray(linear_info['acceleration'])*span])
        linear_se=np.asarray(linear_info['standard_error_proxy'])
    else:
        linear,LC,_,linear_ok=_fit(rows,t,span,1)
        if not linear_ok:return None,dict(info,reason='linear_control_fit_failed')
        linear_se=np.sqrt(np.maximum(np.diag(LC)[2:4],0.))/span
    beta[:2]=linear+reliability*(beta[:2]-linear);beta[2]*=reliability
    a=beta[1]/span
    # The two fits share every input; do not treat their errors as independent.
    sea=np.sqrt(np.maximum(np.diag(C)[2:4]/span**2,linear_se**2))
    linear_a=linear[1]/span
    linear_shrunk=linear_a*np.maximum(0.,1.-linear_se**2/(linear_a**2+1e-12))
    value=cap(linear_shrunk+(a-linear_a))
    info.update(qualified=True,reason='qualified_current_curve_acceleration',
        acceleration=a.tolist(),shrunk_acceleration=value.tolist(),standard_error_proxy=sea.tolist(),
        fitted_current_velocity=beta[0].tolist(),n=len(rows),span_s=span,
        jerk_px_s3=(beta[2]/span**2).tolist(),
        covariance_is_uncalibrated_proxy=True,jerk_not_extrapolated_into_future=True)
    return value,info

class CurvedFlowAcceleration(FlowAcceleration):
    def __init__(self,position_velocity='state'):
        super().__init__(position_velocity)
    def step(self,t,coefficient,body,flow,policy='position-consistent'):
        original=coefficient
        baseline,old=super().step(t,coefficient,body,flow,policy)
        if original is None or not flow.get('valid') or not flow.get('derivative_eligible',True):
            return baseline,dict(old,curved_motion=dict(qualified=False,reason='partial_reference_velocity_only' if flow.get('valid') and not flow.get('derivative_eligible',True) else 'no_current_flow'))
        a,info=curved_derivative(self.velocities,t,old)
        if a is None:return baseline,dict(old,curved_motion=info)
        if len(self.positions)<6:return baseline,dict(old,curved_motion=dict(info,qualified=False,reason='few_curve_position_rows'))
        times=np.array([r[0]-t for r in self.positions]);p=np.array([r[1] for r in self.positions])
        v=np.asarray(info['fitted_current_velocity']);j=np.asarray(info['jerk_px_s3'])
        dynamics=times[:,None]*v+.5*times[:,None]**2*a+times[:,None]**3*j/6.
        origin=np.median(p-dynamics,axis=0);residual=p-origin-dynamics
        median=float(np.median(abs(residual)));inliers=float(np.mean(abs(residual)<=8.07))
        info.update(position_median_residual_px=median,position_inlier_fraction=inliers)
        if median>6. or inliers<.9:
            return baseline,dict(old,curved_motion=dict(info,qualified=False,reason='curve_disagrees_with_position_history'))
        value=copy.deepcopy(original);value['acceleration']=a.tolist()
        return value,dict(info,applied=True,policy=policy,curved_motion=info,linear_control=old,
            reason='qualified_current_curve_acceleration')
