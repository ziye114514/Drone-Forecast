"""Current native evidence transports a previously observed disk/line plane.

Reuse LK/RANSAC of filtered ON/OFF support, then transform the prior circle
tensor. This measures a current transform, not new individual rotor centres.
Uncertainty accumulates without dividing by feature count; a poorly observed
nearly circular plane or failed current registration cannot qualify an axis.
"""
import cv2
import numpy as np
from native_plane_motion import estimate_plane_motion
from native_history_buffer import NativeHistoryBuffer
from axis_amplitude_bridge import current_axis_qualified


def circle_tensor(n):
    n=np.asarray(n,float);length=float(np.linalg.norm(n))
    if n.shape!=(2,) or not np.isfinite(n).all() or not 1e-6<length<=1+1e-6:return None
    n=n/length;t=np.array([-n[1],n[0]])
    return np.outer(t,t)+max(0.,1-min(length,1.)**2)*np.outer(n,n)


def transform_tensor(C,A):
    Q=A@C@A.T;values,U=np.linalg.eigh((Q+Q.T)/2)
    if values[-1]<=1e-12:return None
    ratio=float(np.clip(values[0]/values[-1],0,1));n=U[:,0]*np.sqrt(1-ratio)
    if n[1]>0:n=-n
    return Q/values[-1],n,1-ratio


class ModelAxisTransport:
    def __init__(self):
        self.C=None;self.t=None;self.variance=None;self.direct_t=None
        self.images=None;self.mask=None;self.history=NativeHistoryBuffer()

    def save(self,frame,t):
        mask=frame['native_mask'];bounds=cv2.boundingRect(mask.view(np.uint8))
        self.images,self.mask=self.history.update((frame['on'],frame['off']),mask,bounds);self.t=float(t)

    def step(self,frame,t):
        st=frame['structure'];direct=dict(st);direct.pop('model_axis_transport',None)
        ngood=current_axis_qualified(direct);n=direct.get('axis_xy')
        body=frame['body'];info=dict(qualified=False,current_individual_rotors_redetected=False,
            hidden_frequencies_created=False,physical_RPM_or_COM_truth=False)
        if ngood and n is not None and body.get('detected') and not body.get('center_provisional'):
            C=circle_tensor(n)
            if C is not None:
                self.C=C;self.direct_t=float(t);self.variance=np.deg2rad(max(5.,float(st.get('metric_axis_sigma_proxy_deg') or 5.)))**2
                self.save(frame,t)
            st['model_axis_transport']=dict(info,reason='existing_direct_axis_retained');return
        if self.C is None or self.t is None or not 0<t-self.t<=.12 or not body.get('detected') or body.get('center_provisional'):
            st['model_axis_transport']=dict(info,reason='no_recent_observed_plane_or_current_target');return
        box=body['bbox'];reference=(np.asarray(box[:2])+box[2:])/2
        shift,proof=estimate_plane_motion(self.images,self.mask,(frame['on'],frame['off']),frame['native_mask'],reference)
        if shift is None or not proof.get('qualified'):
            from native_shape_registration import estimate
            shift,shape=estimate(self.mask,frame['native_mask'])
            shape['preceding_texture_registration']=proof;proof=shape
            if shift is None or not proof.get('qualified'):
                st['model_axis_transport']=dict(info,reason='current_native_transform_rejected',transform=proof);return
        A=np.asarray(shift['shared_affine'])[:,:2];mapped=transform_tensor(self.C,A)
        if mapped is None:
            st['model_axis_transport']=dict(info,reason='singular_transported_plane');return
        C,n,gap=mapped
        yy,xx=np.nonzero(self.mask);span=max(float(np.ptp(xx)),float(np.ptp(yy)),1.)*proof['span_fraction']
        pixel_variance=max(float(np.linalg.eigvalsh(np.asarray(shift['displacement_covariance_px2'])).max()),4.)
        angle_variance=self.variance+4*pixel_variance/max(span*span*gap*gap,1e-12)
        sigma=float(np.sqrt(angle_variance));qualified=bool(sigma<=np.deg2rad(15) and np.linalg.norm(n)>1e-6)
        info.update(qualified=qualified,reason='current_model_supported_axis' if qualified else 'plane_direction_information_insufficient',
            transform=proof,angle_sigma_proxy_deg=float(np.rad2deg(sigma)),direct_axis_age_s=t-self.direct_t,
            source_t=self.t,cutoff_t=float(t),current_transform_not_inertial_prediction=True,
            plane_tensor_eigen_gap=gap,axis_uncertainty_not_divided_by_feature_count=True)
        if qualified:
            self.C=C;self.variance=angle_variance;self.save(frame,t)
            st.update(axis_xy=n.tolist(),pose_source='current_native_model_transform',
                metric_axis_sigma_proxy_deg=float(np.rad2deg(sigma)),geometric_weight=.35,
                transported_axis_does_not_change_visible_rotor_count=True)
        st['model_axis_transport']=info


def install(front):
    original=front.step;helper=ModelAxisTransport()
    def step(events,t):
        frame=original(events,t);helper.step(frame,t);return frame
    front.step=step
    return helper
