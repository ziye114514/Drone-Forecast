"""Small cross-recording calibration of the SAME rotor-force rollout.

Features are inertial displacement, nonrotating net-force displacement, and
the own rotor-plane-turn increment. No coordinates, recording ID, body NN or
future input enters runtime. Training targets are matured native rotor-box
displacements, not manual evaluation references. Source-held-out evaluation
must precede any claim of a deployable precision improvement.
"""
import copy
import numpy as np
from scipy.optimize import minimize
from physical_publication import evaluate


def components(row,h):
    actual=np.asarray(evaluate(row,[h])[str(float(h))]);plain=copy.deepcopy(row)
    def clear(packet):
        packet['normal_turn_forecast']=dict(applied=False)
        for part in packet['force'].get('response_model_mixture',[]):clear(part)
    clear(plain);steady=np.asarray(evaluate(plain,[h])[str(float(h))]);c=row['coefficient']
    inertial=float(h)*np.asarray(c['velocity'])
    return np.column_stack((inertial,steady-np.asarray(c['position'])-inertial,actual-steady))


def envelope(row):
    body=row['body']
    if body.get('native_reference_calibration_allowed') is False:return None
    if not body.get('detected') or body.get('center_provisional') or not body.get('bbox'):return None
    b=np.asarray(body['bbox'],float)
    if min(b[:2])<=0 or b[2]>=1279 or b[3]>=719:return None
    extent=float(max(b[2:]-b[:2]));recent=float(body.get('recent_reference_extent_px',extent))
    if extent<.5*recent:return None
    return (b[:2]+b[2:])/2,extent


def teacher_samples(rows):
    stamps=np.array([r['t'] for r in rows]);out=[];last=None
    for r in rows:
        if not r['valid'] or not r['coefficient'] or last is not None and r['t']-last<.199:continue
        a=envelope(r)
        if a is None:continue
        j=int(np.argmin(abs(stamps-r['t']-.4)));future=rows[j];h=future['t']-r['t'];b=envelope(future)
        if abs(h-.4)>.04 or b is None or not .6<=b[1]/a[1]<=1.6:continue
        # Native image-space feature teachers, no trajectory annotation.
        out.append(dict(t=r['t'],h=h,X=components(r,h).tolist(),y=(b[0]-a[0]).tolist(),
            source_extent=a[1],target_extent=b[1]))
        last=r['t']
    return out


def fit(groups,mode='full'):
    active=[0,1,2] if mode=='full' else [1,2] if mode=='force_only' else [0,1]
    fixed=np.array([1.,1.,0. if mode=='no_future_plane' else 1.]);prior=np.ones(len(active));clips=[g for g in groups if g]
    def objective(z):
        weights=fixed.copy();weights[active]=z;value=.5*float((z-prior)@(z-prior));gradient=z-prior
        for group in clips:
            for sample in group:
                X=np.asarray(sample['X']);y=np.asarray(sample['y']);e=X@weights-y
                norm=np.sqrt(float(e@e)+1e-12);scale=6*np.sqrt(2)*len(group)
                value+=norm/scale;gradient+=X[:,active].T@e/(norm*scale)
        return value,gradient
    result=minimize(objective,prior,jac=True,method='L-BFGS-B',bounds=[(0.,None)]*len(active),options=dict(ftol=1e-12,gtol=1e-9,maxiter=500))
    if not result.success and np.linalg.norm(result.jac)>1e-5:raise RuntimeError(result.message)
    weights=fixed.copy();weights[active]=result.x
    return dict(coefficients=weights.tolist(),mode=mode,training_clips=len(clips),training_samples=sum(map(len,clips)),
        objective=float(result.fun),recording_ID_or_coordinates_as_features=False,
        learned_from_matured_native_observations=True,manual_reference_used_for_fit=False,
        hyperparameter_search=False,own_future_plane_feature_enabled=mode!='no_future_plane')


class CalibratedRollout:
    def __init__(self,calibration):self.calibration=calibration;self.offset=np.zeros(2)
    def step(self,row):
        r=copy.deepcopy(row)
        if not r['coefficient']:return r
        current=envelope(r)
        if current is not None:self.offset=current[0]-np.asarray(r['coefficient']['position'])
        r['calibrated_force_rollout']=dict(self.calibration,position_offset=self.offset.tolist(),
            output_is_visible_rotor_envelope_center_not_COM_truth=True,
            no_new_detection_or_future_geometry=True)
        r['physical']=evaluate(r,[.1,.2,.4]);return r


def predict(row,horizons):
    info=row['calibrated_force_rollout'];base=copy.deepcopy(row);base.pop('calibrated_force_rollout')
    if info.get('parameterization')=='native_velocity_innovation':
        from velocity_force_calibration import predict as velocity_predict
        return velocity_predict(row,base,horizons,info)
    position=np.asarray(base['coefficient']['position'])+np.asarray(info['position_offset']);weights=np.asarray(info['coefficients'])
    return {str(float(h)):(position+components(base,float(h))@weights).tolist() for h in horizons}
