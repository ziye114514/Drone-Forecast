"""NN localization plus native rotor geometry/visibility and two-force states.

NN observations are supplied by the caller, not trained or changed here.
The four experimental arms consume the same observation tape, with no branch
feedback into detection scheduling. Hidden thrust remains in the unknown term.
"""
from functools import partial
import copy
import numpy as np
from fast_structure_front import FastStructure,roi_bounds
from native_axis_workspace import NativeAxisWorkspace
from research.stage6.silhouette import body_measurement
from rotor_support_front import rotor_support
from rotor_only_front import native_reconstruction
from local_pixel_phase import local_pixel_phase
from live_structure_bridge import QualifiedCenter
from rotor_force_observer import RotorForceObserver
from rotor_thrust_bank import RotorThrustBankProxy
from physical_publication import evaluate

ARMS=('NN_motion','NN_geometry','NN_four_force','NN_visible_force')


def unavailable(t,body,reason):
    return dict(t=float(t),body=copy.deepcopy(body),structure=dict(mode='insufficient',geometry_epoch=0,
        observed_count=0,observed_model_ids=[],model_centers=None,axis_xy=None,
        current_geometric_center=None,joint_center_is_current_measurement=False,geometric_weight=0.),
        geometry=dict(rotors=[],reason=reason),rates=[],pooled=dict(qualified=False,frequency_hz=None),
        native_available=False,current_pose=False,reason=reason)


class NativeNNRotorFeatures:
    def __init__(self,worker):
        self.worker=worker
        self.geometry=FastStructure(shared_arc_recovery=True,partial_arc_information=True,
            optimized_compute=True,native_histogram=True,reuse_uncertainty=True,side_axis_budget=True)
        self.geometry.body_workspace=NativeAxisWorkspace()
        self.geometry.measure_body=partial(body_measurement,skip_unqualified_lines=True,
            body_workspace=self.geometry.body_workspace)

    def step(self,events,t,body):
        frame=rotor_support(events,t,period_estimator=self.worker,fast_spatial=True,native_only=True,retain_candidates=False)
        native,diagnostic=native_reconstruction(frame);counts20=frame['native_counts20']
        def current_counts(_events,bounds,_t,_selector):
            x,y,X,Y=bounds;keep=native[y:Y,x:X]
            return [np.where(keep,a[y:Y,x:X],0.).astype(np.float32,copy=False) for a in counts20]
        self.geometry.count_current=current_counts
        structure,_,geom,_=self.geometry.step(events[:0],t,body,None,np.zeros(2),defer_rates=True)
        roi=np.zeros(len(frame['period_pixels_unique']),bool)
        pixels=np.asarray(frame['period_pixels_unique'],np.int64)
        if body.get('detected'):
            x,y,X,Y=roi_bounds(body['bbox'],padding=.35)
            roi=(pixels%1280>=x)&(pixels%1280<X)&(pixels//1280>=y)&(pixels//1280<Y)
        pix=pixels[roi];frequencies=frame['frequency_hz'].reshape(-1)[pix]
        accepted=frequencies>0;pix=pix[accepted];frequencies=frequencies[accepted]
        px,py=pix%1280,pix//1280;rates=[]
        for rotor in geom.get('rotors',[]):
            angle=np.deg2rad(rotor['angle']);dx=px-rotor['cx'];dy=py-rotor['cy']
            u=dx*np.cos(angle)+dy*np.sin(angle);v=-dx*np.sin(angle)+dy*np.cos(angle)
            inside=(u/max(rotor['major']/2,1e-6))**2+(v/max(rotor['minor']/2,1e-6))**2<=1.
            fs=frequencies[inside];median=float(np.median(fs)) if len(fs) else None
            spread=float(np.median(abs(np.log(fs/median)))) if len(fs) else None
            rates.append(dict(valid=bool(len(fs)>=6 and spread<=.08),frequency_hz=median,
                periodic_pixels=len(fs),log_frequency_MAD=spread,mechanical_RPM_measured=False))
        pose=bool(len(geom.get('rotors',[]))>=2 or geom.get('side',{}).get('valid'))
        pooled=dict(qualified=bool(pose and len(frequencies)>=12),
            frequency_hz=float(np.median(frequencies)) if len(frequencies)>=12 else None,
            selection_source='NN_ROI_global_pixel_median')
        local=dict(qualified=False,reason='nonflat_or_no_NN')
        if body.get('detected') and structure.get('mode') not in ('four','three','partial_two'):
            local_frame=dict(frame,period_pixels_unique=pixels[roi],
                period_candidate_frequencies=frame['period_candidate_frequencies'][roi],
                first_harmonic_statistics=frame['first_harmonic_statistics'][roi] if frame.get('first_harmonic_statistics') is not None else None)
            local=local_pixel_phase(local_frame,body['center'])
            if local['qualified']:
                pooled=dict(qualified=pose,frequency_hz=local['frequency_hz'],selection_source='local_pixel_first_harmonic')
        return dict(t=float(t),body=copy.deepcopy(body),structure=structure,geometry=geom,rates=rates,
            pooled=pooled,current_pose=pose,native_available=True,counts=frame['counts'],
            reconstruction=diagnostic,local_temporal_phase=local,
            NN_used_for_localization_and_ROI=True,rotor_features_from_native_ONOFF_only=True)


class NNObservationObserver(RotorForceObserver):
    def _measurement(self,z,body,motion,lag):
        H,R,measurement,used=super()._measurement(z,body,None,lag)
        if body.get('position_covariance_proxy') is not None:R=np.asarray(body['position_covariance_proxy'],float)
        flow=body.get('fixed_current_flow') or {}
        if not flow.get('valid'):return H,R,measurement,used
        delay=self.t-float(flow.get('midpoint_t',self.t))
        covariance=np.asarray(flow.get('velocity_covariance'),float)
        velocity=np.asarray(flow.get('velocity'),float)
        if not 0<=delay<=.12 or covariance.shape!=(2,2) or velocity.shape!=(2,) or not np.isfinite(covariance).all():
            return H,R,measurement,used
        H=np.zeros((4,self.state_dim));H[:2,:2]=np.eye(2);H[2:,2:4]=np.eye(2)
        H[2:,4:6]=-delay*np.eye(2);H[2:,self.gain_index]=-delay*self.u
        # NN/KLT localization and flow share images. A conservative block upper
        # bound2*diag(Rp,Rv) avoids declaring their errors independent.
        C=np.zeros((4,4));C[:2,:2]=2*R;C[2:,2:]=2*covariance
        return H,C,np.r_[z,velocity],True


class NNRotorFusion:
    def __init__(self,arm):
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm;self.center=QualifiedCenter();self.observer=NNObservationObserver()
        self.proxy=RotorThrustBankProxy(vector_pool=True)

    def step(self,observation):
        t=observation['t'];body=copy.deepcopy(observation['body']);s=observation['structure']
        center_info=dict(applied=False,reason='NN_position_control')
        if self.arm!='NN_motion':
            body,center_info=self.center.step(body,s,observation['geometry'].get('rotors',[]))
            if center_info.get('applied'):body['position_covariance_proxy']=center_info['covariance_proxy']
        body['fixed_current_flow']=copy.deepcopy(observation.get('fixed_current_flow',{}))
        u=None;quality=0.;meta=dict(observed_components={},input_sigma_proxy=0.,learning_input_current=False,
            source='no_rotor_force_control',hidden_frequencies_measured=0)
        if self.arm in ('NN_four_force','NN_visible_force'):
            u,quality,meta=self.proxy.step(t,s,observation['geometry'].get('rotors',[]),observation['rates'],observation['pooled'])
            if not observation.get('current_pose'):
                u=None;quality=0.;meta=dict(meta,observed_components={},input_sigma_proxy=0.,learning_input_current=False)
            if self.arm=='NN_four_force' and not (s.get('mode')=='four' and len(meta.get('current_model_ids',[]))==4 and meta.get('rotor_bank_used')):
                u=None;quality=0.;meta=dict(meta,observed_components={},input_sigma_proxy=0.,learning_input_current=False,
                    reason='four_only_control_rejects_partial_or_axis_input')
        coefficient,valid,force,state=self.observer.step(t,body,u,quality,
            allow_gain_learning=meta.get('learning_input_current',False),input_meta=meta)
        row=dict(t=t,body=body,coefficient=coefficient,valid=valid,force=force,state=state,
            structure=s,center_bridge=center_info,period_selection=meta,arm=self.arm,
            original_NN_position=observation['body'].get('center'),NN_observation_changed=False)
        row['physical']=evaluate(row,[.1,.2,.4]) if coefficient else {}
        return row
