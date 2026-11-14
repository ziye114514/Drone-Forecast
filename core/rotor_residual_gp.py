"""Small source-trained GP prior for image-domain unknown-force continuation.

Conceptual reference: Torrente et al., Data-Driven MPC for Quadrotors, RAL2021.
This is an independent NumPy/SciPy implementation of ordinary GP regression,
NOT their control stack or a reproduction of their metric flight results.

Inputs: rotor-derived current motion, headless plane and plane-turn rate,
normalized by current native rotor extent. No position, time, source ID,
whole-body NN, future command or manually annotated training target.
Outputs: future displacement residual expressed as an unknown-force-rate prior.
The current p/v/a and visible/hidden force partition are preserved.
"""
import copy
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from calibrated_force_rollout import teacher_samples,envelope
from conditional_unknown_force import base_prediction,HORIZON,TAU
from separated_force_response import rate_kernel
from adaptive_force_memory import kernel
from axis_amplitude_bridge import current_axis_qualified


def coordinates(row,mode='disk'):
    st=row['structure'];n=np.asarray(st.get('axis_xy') or [],float)
    good=bool(n.shape==(2,) and np.linalg.norm(n)>1e-6 and current_axis_qualified(st))
    norm=float(np.linalg.norm(n)) if good else 1.
    if mode=='camera' or not good:n=np.array([0.,-1.]);norm=1.
    else:
        n=n/norm
        if n[1]>0 or n[1]==0 and n[0]<0:n=-n
    B=np.column_stack((n,np.array([-n[1],n[0]])))
    body=row['body'];extent=float(body.get('observed_support_extent_px',body.get('recent_reference_extent_px',100.)))
    if extent<=0 or not np.isfinite(extent):extent=100.
    c=row['coefficient'];v=B.T@np.asarray(c['velocity'])/extent;a=B.T@np.asarray(c['acceleration'])/extent
    z=np.r_[HORIZON*v,kernel(-1/TAU,HORIZON)*a]
    if mode!='camera':
        normal=row.get('normal_turn_forecast') or {};trend=normal.get('trend') or {}
        omega=float(trend.get('shrunk_angular_rate_rad_s',0.)) if normal.get('applied') else 0.
        z=np.r_[z,n,norm,HORIZON*omega]
    return z,B,extent


def teachers(rows,mode):
    lookup={r['t']:r for r in rows};out=[]
    for old in teacher_samples(rows):
        r=lookup[old['t']];h=old['h'];z,B,size=coordinates(r,mode)
        base=np.asarray(base_prediction(r,[h],mode)[str(float(h))])-np.asarray(r['coefficient']['position'])
        factor=rate_kernel(-1/TAU,0.,h)/rate_kernel(-1/TAU,0.,HORIZON)
        y=B.T@(np.asarray(old['y'])-base)/(size*factor)
        out.append(dict(t=r['t'],z=z.tolist(),y=y.tolist(),measurement_variance=72/(size*size*factor*factor)))
    return out


def fit(groups,mode):
    groups=[g for g in groups if g];samples=[s for g in groups for s in g]
    Z=np.array([s['z'] for s in samples]);Y=np.array([s['y'] for s in samples]);mean=Z.mean(0);std=np.std(Z,axis=0)
    std=np.where(std<1e-6,1.,std);Z=(Z-mean)/std
    distance=np.maximum(0.,np.sum(Z*Z,axis=1)[:,None]+np.sum(Z*Z,axis=1)[None,:]-2*Z@Z.T)
    measurement=np.array([s['measurement_variance'] for s in samples]);overlap=np.zeros((len(Z),len(Z)));at=0
    for g in groups:
        stamps=np.array([s['t'] for s in g]);end=at+len(g)
        # Overlapping .4s teaching intervals carry correlated response error.
        # This is a PSD triangular overlap kernel, not a fictitious extra N.
        overlap[at:end,at:end]=np.maximum(0.,1.-abs(stamps[:,None]-stamps[None,:])/HORIZON);at=end
    def solve(theta,gradient=False):
        amplitude,length,noise=np.exp(theta);K=amplitude**2*np.exp(-.5*distance/(length*length))
        C=K+noise*noise*overlap+np.diag(measurement+1e-8)
        L=cho_factor(C,lower=True,check_finite=False);alpha=cho_solve(L,Y,check_finite=False)
        value=.5*float(np.sum(Y*alpha))+Y.shape[1]*np.log(np.diag(L[0])).sum()+.5*Y.size*np.log(2*np.pi)
        if not gradient:return value,L,alpha,K
        inv=cho_solve(L,np.eye(len(Z)),check_finite=False);W=Y.shape[1]*inv-alpha@alpha.T
        grad=np.array([.5*np.sum(W*(2*K)),.5*np.sum(W*(K*distance/(length*length))),.5*np.sum(W*(2*noise*noise*overlap))])
        return value,grad
    initial=np.log([max(.01,float(np.std(Y))),np.sqrt(Z.shape[1]),max(.01,float(np.median(np.sqrt(measurement))))])
    result=minimize(lambda x:solve(x,True),initial,jac=True,method='L-BFGS-B',bounds=[(-8.,4.),(-4.,6.),(-8.,4.)],options=dict(maxiter=400,ftol=1e-10,gtol=1e-6))
    value,L,alpha,K=solve(result.x)
    if not np.isfinite(value):raise RuntimeError('Invalid GP fit')
    return dict(mode=mode,Z=Z.tolist(),Y=Y.tolist(),input_mean=mean.tolist(),input_scale=std.tolist(),
        log_parameters=result.x.tolist(),alpha=alpha.tolist(),cholesky=np.tril(L[0]).tolist(),
        training_samples=len(Z),training_sources=len(groups),negative_log_marginal_likelihood=float(value),
        optimizer_success=bool(result.success),hyperparameters_learned_only_from_native_teachers=True,
        annotations_used_for_training=False,source_ID_is_not_runtime_feature=True,
        correlated_overlapping_teachers=True,posterior_is_not_calibrated_flight_force_truth=True)


class RotorResidualGP:
    def __init__(self,model):
        self.model=model;self.Z=np.asarray(model['Z']);self.alpha=np.asarray(model['alpha']);self.L=np.asarray(model['cholesky']);self.offset=np.zeros(2)
        self.amplitude,self.length,self.noise=np.exp(model['log_parameters'])
    def step(self,row):
        r=copy.deepcopy(row)
        if not r['coefficient']:return r
        current=envelope(r)
        if current is not None:self.offset=current[0]-np.asarray(r['coefficient']['position'])
        z,B,size=coordinates(r,self.model['mode']);z=(z-np.asarray(self.model['input_mean']))/np.asarray(self.model['input_scale'])
        distance=np.sum((self.Z-z)**2,axis=1);k=self.amplitude**2*np.exp(-.5*distance/self.length**2)
        from scipy.linalg import solve_triangular
        v=solve_triangular(self.L,k,lower=True,check_finite=False);variance=max(0.,self.amplitude**2-float(v@v))
        d=B@(k@self.alpha)*size;j=d/rate_kernel(-1/TAU,0.,HORIZON)
        r['rotor_residual_gp']=dict(mode=self.model['mode'],future_unknown_rate_px_s3=j.tolist(),position_offset=self.offset.tolist(),
            residual_displacement_sigma_proxy_px=float(size*np.sqrt(variance)),closest_standardized_input_distance=float(np.sqrt(np.min(distance))),
            unknown_rate_is_conditional_prior_not_measured_thrust=True,current_p_v_a_unchanged=True,
            current_visible_force_ownership_unchanged=True,output_reference='visible_rotor_envelope_center',COM_truth=False)
        from physical_publication import evaluate
        r['physical']=evaluate(r,[.1,.2,.4]);return r


def predict(row,horizons):
    info=row['rotor_residual_gp'];base=copy.deepcopy(row);base.pop('rotor_residual_gp');values=base_prediction(base,horizons,info['mode'])
    offset=np.asarray(info['position_offset']);j=np.asarray(info['future_unknown_rate_px_s3'])
    return {str(float(h)):(np.asarray(values[str(float(h))])+offset+rate_kernel(-1/TAU,0.,float(h))*j).tolist() for h in horizons}
