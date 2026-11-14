"""Source-trained conditional prior for FUTURE lumped-unknown force rate.

The present p/v/a and visible/hidden ownership are unchanged. Native rotor
displacements that have already matured supervise a small linear image-response
model. Disk-normal/tangent projections are headless; no coordinates, video ID,
metric thrust or hidden motor speed are features. This is a learned control /
disturbance continuation PRIOR, not a measured external force.
"""
import copy
import numpy as np
from scipy.optimize import minimize
from adaptive_force_memory import kernel
from separated_force_response import rate_kernel
from calibrated_force_rollout import teacher_samples, envelope
from axis_amplitude_bridge import current_axis_qualified

HORIZON=.4
TAU=.35


def plain(row):
    r=copy.deepcopy(row)
    def clear(p):
        p['normal_turn_forecast']=dict(applied=False)
        for part in p['force'].get('response_model_mixture',[]):clear(part)
    clear(r)
    return r


def basis(row,camera=False,history=False):
    c=row['coefficient'];v=np.asarray(c['velocity']);a=np.asarray(c['acceleration'])
    n=np.asarray(row['structure'].get('axis_xy') or [],float)
    if camera:P=np.diag([0.,1.])
    elif n.shape==(2,) and np.linalg.norm(n)>1e-6 and current_axis_qualified(row['structure']):P=np.outer(n,n)/float(n@n)
    else:P=np.eye(2)*.5
    tangent=np.eye(2)-P
    # Five response coefficients, rather than one blanket speed shrinkage.
    # Braking is a dimensionless observed motion state, not a video selector.
    braking=max(0.,-float(v@a)/max(float(np.linalg.norm(v)*np.linalg.norm(a)),1e-9))
    ka=kernel(-1/TAU,HORIZON)
    result=np.column_stack((HORIZON*P@v,HORIZON*tangent@v,ka*P@a,ka*tangent@a,HORIZON*braking*v))
    if history:
        info=row.get('causal_force_history') or {};j=np.asarray(info['net_jerk_px_s3']) if info.get('qualified') else np.zeros(2)
        kj=rate_kernel(-1/TAU,0.,HORIZON)
        result=np.column_stack((result,kj*P@j,kj*tangent@j))
    return result


def base_prediction(row,horizons,mode):
    from physical_publication import evaluate
    return evaluate(plain(row) if mode=='no_future_plane' else row,horizons)


def teachers(rows,mode='disk',history=False):
    lookup={r['t']:r for r in rows};out=[]
    for old in teacher_samples(rows):
        r=lookup[old['t']];h=old['h']
        baseline=np.asarray(base_prediction(r,[h],mode)[str(float(h))])-np.asarray(r['coefficient']['position'])
        factor=rate_kernel(-1/TAU,0.,h)/rate_kernel(-1/TAU,0.,HORIZON)
        out.append(dict(t=r['t'],h=h,X=(basis(r,mode=='camera',history)*factor).tolist(),y=(np.asarray(old['y'])-baseline).tolist()))
    return out


def fit(groups,mode='disk',prior=None,group_normalize=True,history=False,bounds=None):
    groups=[g for g in groups if g];prior=np.zeros(7 if history else 5) if prior is None else np.asarray(prior,float)
    def objective(z):
        difference=z-prior;value=.5*float(difference@difference);grad=difference.copy()
        for g in groups:
            for sample in g:
                X=np.asarray(sample['X']);e=X@z-np.asarray(sample['y']);norm=np.sqrt(float(e@e)+1e-8)
                scale=6*np.sqrt(2)*(len(g) if group_normalize else 1);value+=norm/scale;grad+=X.T@e/(norm*scale)
        return value,grad
    solved=minimize(objective,prior,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-12,gtol=1e-8,maxiter=1000))
    if not solved.success and np.linalg.norm(solved.jac)>1e-4:raise RuntimeError(solved.message)
    return dict(weights=solved.x.tolist(),mode=mode,training_samples=sum(map(len,groups)),training_sources=len(groups),
        objective=float(solved.fun),unit_Gaussian_prior=True,manual_labels_used=False,video_ID_or_coordinates_used=False,
        hyperparameter_search=False,future_rate_is_learned_prior_not_observed_force=True,
        prior_weights=prior.tolist(),group_normalized=group_normalize,net_force_history_used=history)


class ConditionalUnknownForce:
    def __init__(self,model):self.model=model;self.offset=np.zeros(2);self.observations=None
    def step(self,row):
        r=copy.deepcopy(row)
        observations=row.get('state',{}).get('observations')
        if observations is not None and self.observations is not None and observations<self.observations:self.offset=np.zeros(2)
        self.observations=observations
        if r['coefficient'] is None:return r
        current=envelope(r)
        if current is not None:self.offset=current[0]-np.asarray(r['coefficient']['position'])
        correction=basis(r,self.model['mode']=='camera',self.model.get('net_force_history_used',False))@np.asarray(self.model['weights'])
        j=correction/rate_kernel(-1/TAU,0.,HORIZON)
        r['conditional_unknown_force']=dict(self.model,position_offset=self.offset.tolist(),
            future_unknown_rate_px_s3=j.tolist(),observed_force_unchanged=True,current_p_v_a_unchanged=True,
            output_reference='visible_rotor_envelope_center',COM_truth=False)
        from physical_publication import evaluate
        r['physical']=evaluate(r,[.1,.2,.4]);return r


def predict(row,horizons):
    info=row['conditional_unknown_force'];r=copy.deepcopy(row);r.pop('conditional_unknown_force')
    values=base_prediction(r,horizons,info['mode']);offset=np.asarray(info['position_offset']);j=np.asarray(info['future_unknown_rate_px_s3'])
    return {str(float(h)):(np.asarray(values[str(float(h))])+offset+rate_kernel(-1/TAU,0.,float(h))*j).tolist() for h in horizons}
