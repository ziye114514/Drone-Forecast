"""Causal dominant-background cue for the lumped UNKNOWN image-motion term.

Rotors remain the only target detector. Native ON/OFF outside that target is
used only to estimate a possible camera/background-motion nuisance. It is not
a calibrated camera pose, wind, metric force or rotor measurement. A single
bounded continuation weight learns ONLY after native observations mature.
"""
import copy
import numpy as np
from native_plane_motion import estimate_plane_motion
from native_history_buffer import NativeHistoryBuffer
from calibrated_force_rollout import envelope
from adaptive_force_memory import kernel
from physical_publication import freeze


def correction_feature(acceleration,h):
    # Existing short-gap(.1s) vs ordinary(.35s) correlation hypotheses.
    # Their h² terms cancel: no extra current p/v/a or force impulse.
    return (kernel(-1/.1,h)-kernel(-1/.35,h))*np.asarray(acceleration)


class BackgroundMotion:
    def __init__(self,reference_support=False,translation_only=False):
        self.buffer=NativeHistoryBuffer();self.previous=None;self.mask=None;self.last=None;self.transforms=[]
        self.reference_support=bool(reference_support)
        self.translation_only=bool(translation_only)
    def step(self,row,counts):
        t=row['t'];on,off=counts;mask=(on+off)>0;body=row['body'];box=body.get('bbox');c=row.get('coefficient')
        if box:cx,cy=(np.asarray(box[:2])+box[2:])/2;w,h=np.asarray(box[2:])-box[:2]
        elif c:cx,cy=c['position'];w,h=200.,120.
        else:cx,cy=640.,360.;w,h=200.,120.
        lo=np.maximum(0,np.floor([cx-w-112,cy-h-112])).astype(int);hi=np.minimum([1280,720],np.ceil([cx+w+112,cy+h+112])).astype(int);mask[lo[1]:hi[1],lo[0]:hi[0]]=False
        info=dict(qualified=False,acceleration_xy=[0.,0.],rotor_detection_replaced=False,
            input_is_native_background_counts=True,calibrated_camera_pose=False,future_images_used=False)
        if self.previous is not None and self.last is not None and 0<t-self.last<=.12:
            point=np.asarray(c['position']) if c else np.array([640.,360.])
            shift,proof=estimate_plane_motion(self.previous,self.mask,counts,mask,point,
                require_center_support=self.reference_support,translation_only=self.translation_only)
            info.update(background_pair_qualified=proof['qualified'],reason=proof['reason'],tracks=proof.get('bidirectional_tracks'),inliers=proof.get('inliers'))
            if self.reference_support:
                info['background_reference_support']=dict(required=True,inside_track_hull=proof.get('center_in_current_track_hull'),mode=proof.get('mode'),
                    no_extrapolation_from_remote_background_features=True)
            if self.translation_only:
                info['background_motion_model']='shared_translation_no_spatial_extrapolation'
            if shift is not None:
                M=np.asarray(proof['affine']);Q=np.asarray(proof['displacement_covariance_px2']);eig,U=np.linalg.eigh((Q+Q.T)/2);Q=(U*np.maximum(eig,4.))@U.T
                self.transforms.append(dict(source=self.last,t=t,M=M,Q=Q));self.transforms=[q for q in self.transforms if t-q['source']<=.12+1e-8]
                # Track the SAME background point backwards through current/
                # past affine transforms, not the moving drone's own path.
                q=point.copy();samples=[];end=t
                for z in reversed(self.transforms):
                    if abs(z['t']-end)>1e-6:break
                    old=np.linalg.solve(z['M'][:,:2],q-z['M'][:,2]);samples.append((z['source'],z['t'],q-old,z['Q']));q=old;end=z['source']
                if len(samples)>=3:
                    prior=np.eye(4)/1e6;rhs=np.zeros(4);normal=prior.copy()
                    for s,e,d,Q in samples:
                        X=np.column_stack(((e-s)*np.eye(2),.5*((e-t)**2-(s-t)**2)*np.eye(2)))
                        # Shared-frame flow errors are correlated; do not
                        # divide a noise proxy by feature/pixel count.
                        W=np.linalg.inv(Q*len(samples));normal+=X.T@W@X;rhs+=X.T@W@d
                    fit=np.linalg.solve(normal,rhs);a=fit[2:];norm=float(np.linalg.norm(a))
                    if norm>3000:a*=3000/norm
                    info.update(qualified=True,reason='causal_background_motion_acceleration_proxy',velocity_xy=fit[:2].tolist(),
                        acceleration_xy=a.tolist(),pair_count=len(samples),span_s=t-min(z[0] for z in samples),
                        covariance_is_uncalibrated_proxy=True,possible_camera_motion_not_world_drone_acceleration=True)
            else:self.transforms=[]
        else:self.transforms=[]
        self.previous,self.mask=self.buffer.update(counts,mask,(0,0,1280,720));self.last=t
        return info


class OnlineBackgroundResponse:
    def __init__(self):self.pending=[];self.last_teacher=None;self.xx=0.;self.xy=0.;self.observations=None;self.matured=0
    def step(self,row,background):
        from physical_publication import evaluate
        r=copy.deepcopy(row);t=row['t'];obs=row.get('state',{}).get('observations')
        if obs is not None and self.observations is not None and obs<self.observations:self.__init__()
        self.observations=obs;current=envelope(row) if row.get('coefficient') else None;keep=[]
        for old in self.pending:
            h=t-old['t']
            if h<.4-.00001:keep.append(old);continue
            if abs(h-.4)>.04 or current is None or not .6<=current[1]/old['extent']<=1.6:continue
            base=copy.deepcopy(old['base']);base.pop('native_fov_observation',None)
            predicted=np.asarray(evaluate(base,[h])[str(float(h))]);x=correction_feature(old['a'],h);y=current[0]-predicted
            #6px native-centre proxy at both ends; teachers overlap by2x.
            variance=2*6.**2*2;self.xx+=float(x@x)/variance;self.xy+=float(x@y)/variance;self.matured+=1
        self.pending=keep;beta=float(np.clip(self.xy/(1+self.xx),0.,1.))
        if row.get('valid') and current is not None and background.get('qualified') and (self.last_teacher is None or t-self.last_teacher>=.199):
            self.pending.append(dict(t=t,base=freeze(row,t),a=background['acceleration_xy'],extent=current[1]));self.last_teacher=t
        r['background_motion_proxy']=background
        if row.get('coefficient'):
            a=background['acceleration_xy'] if background.get('qualified') else [0.,0.]
            r['background_unknown_response']=dict(weight=beta,acceleration_xy=a,matured_native_teachers=self.matured,
                pending_teachers=len(self.pending),posterior_precision=1+self.xx,
                current_p_v_a_and_force_ledger_unchanged=True,current_target_detector_is_rotor_only=True,
                unseen_future_camera_commands_predicted=False,manual_future_labels_used=False)
            r['physical']=evaluate(r,[.1,.2,.4])
        return r


def predict(row,horizons):
    from physical_publication import evaluate
    r=copy.deepcopy(row);info=r.pop('background_unknown_response');values=evaluate(r,horizons)
    return {str(float(h)):(np.asarray(values[str(float(h))])+float(info['weight'])*correction_feature(info['acceleration_xy'],float(h))).tolist() for h in horizons}
