"""Calibrate causal native-velocity innovations, not blanket speed shrinkage.

Use the currently observed rotor normal to split the lag-compensated native
displacement innovation into normal and tangential parts. Same noisy events
are NOT treated as independent KF observations. This is a learned mean readout
from other source recordings, with explicit published p/v/a calibration.
"""
import numpy as np
from scipy.optimize import minimize
from calibrated_force_rollout import components,teacher_samples
from axis_amplitude_bridge import current_axis_qualified


def innovation(row,camera_axis=False):
    mo=row['motion'];s=mo.get('estimate') or {};dt=mo.get('dt_s');zero=np.zeros(2)
    if not dt or not mo.get('current_observation') or s.get('score',0)<=0 or s.get('peak_at_search_boundary') or not s.get('current_displacement_qualified',True):return zero,zero
    c=row['coefficient'];native=np.array([s['dx'],s['dy']])/dt+dt/2*np.asarray(c['acceleration']);delta=native-np.asarray(c['velocity'])
    R=np.asarray(s.get('displacement_covariance_px2',np.eye(2)*4.))/dt**2
    R+=np.diag(np.square(row.get('state',{}).get('velocity_sigma_px_s',[100.,100.])))
    n=np.asarray(row['structure'].get('axis_xy') or [],float)
    if camera_axis:n=np.array([0.,-1.])
    if n.shape!=(2,) or np.linalg.norm(n)<1e-6 or not camera_axis and not current_axis_qualified(row['structure']):
        power=float(delta@delta);return zero,delta*power/(power+max(float(np.trace(R)),1e-12))
    n=n/np.linalg.norm(n);tangent=np.array([-n[1],n[0]])
    def part(u):
        z=float(u@delta);variance=float(u@R@u);return u*z*(z*z/(z*z+max(variance,1e-12)))
    return part(n),part(tangent)


def basis(row,h,camera_axis=False):
    old=components(row,h);normal,tangent=innovation(row,camera_axis)
    return old[:,0],np.column_stack((h*normal,h*tangent,old[:,1],old[:,2]))


def teachers(rows,camera_axis=False):
    existing=teacher_samples(rows);lookup={r['t']:r for r in rows};out=[]
    for old in existing:
        constant,X=basis(lookup[old['t']],old['h'],camera_axis)
        out.append(dict(old,X=X.tolist(),y=(np.asarray(old['y'])-constant).tolist()))
    return out


def fit(groups,mode='full'):
    active=[0,1,2] if mode=='no_future_plane' else [0,1,2,3]
    prior=np.array([0.,0.,1.,1.]);fixed=prior.copy()
    if mode=='no_future_plane':fixed[3]=0
    clips=[g for g in groups if g]
    def objective(z):
        weights=fixed.copy();weights[active]=z;diff=z-prior[active];value=.5*float(diff@diff);gradient=diff.copy()
        for group in clips:
            for sample in group:
                X=np.asarray(sample['X']);e=X@weights-np.asarray(sample['y']);norm=np.sqrt(float(e@e)+1e-12);scale=6*np.sqrt(2)*len(group)
                value+=norm/scale;gradient+=X[:,active].T@e/(norm*scale)
        return value,gradient
    result=minimize(objective,prior[active],jac=True,method='L-BFGS-B',bounds=[(0.,None)]*len(active),options=dict(ftol=1e-12,gtol=1e-9,maxiter=500))
    if not result.success and np.linalg.norm(result.jac)>1e-5:raise RuntimeError(result.message)
    weights=fixed.copy();weights[active]=result.x
    return dict(coefficients=weights.tolist(),parameterization='native_velocity_innovation',mode=mode,
        training_samples=sum(map(len,clips)),training_clips=len(clips),camera_axis_control=mode=='camera_axis',
        manual_reference_used_for_fit=False,hyperparameter_search=False,objective=float(result.fun))


def predict(row,base,horizons,info):
    position=np.asarray(base['coefficient']['position'])+np.asarray(info['position_offset']);weights=np.asarray(info['coefficients'])
    return {str(float(h)):(position+basis(base,float(h),info.get('camera_axis_control',False))[0]+basis(base,float(h),info.get('camera_axis_control',False))[1]@weights).tolist() for h in horizons}
