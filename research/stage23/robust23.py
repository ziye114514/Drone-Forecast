"""Robust acceleration variants; same frozen CV state/validity as stage22."""
from pathlib import Path
import copy,importlib.util
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reused_predict22',ROOT/'research/stage22/predict22.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
PARAMS=dict(base.PARAMS,acceleration_cap_px_s2=3000.,minimum_fit_span_s=.16,minimum_fit_points=6,
    huber_delta_px=1.345*base.motion.SIGMA_MEAS,noise_floor_px=base.motion.SIGMA_MEAS,max_nfev=30)
MODELS=['hold','cv','cv_hist_acc','bounded_acc','huber_acc','trusted_acc']


def cap(acceleration):
    a=np.array(acceleration,float);norm=np.linalg.norm(a)
    return a*min(1.,PARAMS['acceleration_cap_px_s2']/max(norm,1e-9))


def robust_acceleration(history):
    if len(history)<PARAMS['minimum_fit_points']:return np.zeros(2),np.full(2,np.inf),dict(valid=False,reason='few_points')
    times=np.array([t for t,p in history]);points=np.array([p for t,p in history]);span=np.ptp(times)
    if span<PARAMS['minimum_fit_span_s']-1e-8:return np.zeros(2),np.full(2,np.inf),dict(valid=False,reason='short_time_support')
    tau=(times-times.mean())/span;design=np.column_stack([np.ones(len(times)),tau,.5*tau*tau])
    initial=np.linalg.lstsq(design,points,rcond=None)[0];jacobian=np.kron(design,np.eye(2))
    fit=least_squares(lambda x:(design@x.reshape(3,2)-points).ravel(),initial.ravel(),
        jac=lambda x:jacobian.copy(),loss='huber',f_scale=PARAMS['huber_delta_px'],max_nfev=PARAMS['max_nfev'])
    if not fit.success:return np.zeros(2),np.full(2,np.inf),dict(valid=False,reason='solver_not_converged',nfev=fit.nfev)
    coefficients=fit.x.reshape(3,2);residual=points-design@coefficients;variance=[]
    for dim in range(2):
        r=residual[:,dim];weights=np.minimum(1.,PARAMS['huber_delta_px']/np.maximum(abs(r),1e-12))
        sigma=max(PARAMS['noise_floor_px'],1.4826*float(np.median(abs(r-np.median(r)))))
        covariance=sigma*sigma*np.linalg.pinv(design.T@(weights[:,None]*design))
        variance.append(max(0.,float(covariance[2,2]))/span**4)
    acceleration=coefficients[2]/span**2;standard_error=np.sqrt(variance)
    return acceleration,standard_error,dict(valid=True,reason='huber_fit',span_s=float(span),n=len(times),nfev=fit.nfev,
        acceleration=acceleration.tolist(),standard_error=standard_error.tolist(),median_abs_residual_px=float(np.median(abs(residual))),
        robust_inlier_fraction=float(np.mean(abs(residual)<=PARAMS['huber_delta_px'])))


class Predictor:
    def __init__(self):self.base=base.Predictor()

    def step(self,t,center):
        original=self.base.step(t,center);q=copy.deepcopy(original);q['robust_diagnostic']=dict(valid=False,reason='forecast_not_published')
        if q['valid']:
            cv=q['coefficients']['cv'];old=q['coefficients']['cv_hist_acc']['acceleration']
            acceleration,error,diagnostic=robust_acceleration(self.base.models['cv_hist_acc'].history)
            shrink=np.maximum(0.,1.-np.divide(error*error,acceleration*acceleration+1e-12)) if diagnostic['valid'] else np.zeros(2)
            variants={'bounded_acc':cap(old),'huber_acc':acceleration,'trusted_acc':cap(acceleration*shrink)}
            for name,a in variants.items():q['coefficients'][name]=dict(copy.deepcopy(cv),acceleration=a.tolist())
            q['predictions']={name:{str(h):base.evaluate(c,h) for h in PARAMS['horizons']} for name,c in q['coefficients'].items()}
            q['robust_diagnostic']=dict(diagnostic,shrink=shrink.tolist())
        return q,original
