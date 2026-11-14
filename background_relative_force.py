"""Causal dominant-background reference for the existing rotor two-force state.

Only target-excluded native event flow supplies this nuisance reference. It is
not calibrated ego-motion or world force. The rotor detector, conics and RPM
proxies remain in original pixels. Missing camera evidence is not rotor force.
"""
from collections import deque
import copy
import numpy as np
from adaptive_force_memory import kernel

TAU=.35

class BackgroundReference:
    def __init__(self,noise_only=False,shared_endpoints=False,preserve_foreground_floor=False):
        self.noise_only=noise_only;self.offset=np.zeros(2);self.variance=np.zeros((2,2))
        self.shared_endpoints=bool(shared_endpoints)
        self.preserve_foreground_floor=bool(preserve_foreground_floor or shared_endpoints)
        self.velocity=np.zeros(2);self.acceleration=np.zeros(2)
        self.last=None;self.last_observed=None;self.history=deque();self.latest={}

    def advance(self,evidence):
        t=float(evidence['t']);dt=0. if self.last is None else t-self.last
        if dt<0:raise ValueError('Increasing source times required')
        pair=evidence.get('pair');proof=evidence.get('proof') or {}
        observed=bool(pair and self.last is not None and abs(pair['source_t']-self.last)<1e-6)
        age=float('inf') if self.last_observed is None else t-self.last_observed
        q=np.zeros((2,2));delta=np.zeros(2)
        if observed:
            delta=np.asarray(pair['displacement_xy'],float);q=np.asarray(pair['covariance_px2'],float)
            self.velocity=np.asarray(proof['velocity_xy'] if proof.get('qualified') else delta/dt,float)
            self.acceleration=np.asarray(proof['acceleration_xy'] if proof.get('qualified') else [0.,0.],float)
            self.last_observed=t
        elif age<=.12+1e-8:
            delta=dt*self.velocity+kernel(-1/TAU,dt)*self.acceleration
            self.velocity+=TAU*(-np.expm1(-dt/TAU))*self.acceleration
            self.acceleration*=np.exp(-dt/TAU)
            # Reuse the existing1000px/s² disturbance prior; no new fitted Q.
            q=np.eye(2)*(1000.*dt*dt/2)**2
        else:self.velocity=np.zeros(2);self.acceleration=np.zeros(2)
        self.offset+=delta;self.variance+=q;self.last=t
        self.history.append((t,self.offset.copy(),self.variance.copy(),observed))
        while self.history and t-self.history[0][0]>.5:self.history.popleft()
        enabled=bool(self.last_observed is not None and t-self.last_observed<=.12+1e-8)
        self.latest=dict(t=t,offset_xy=self.offset.tolist(),velocity_xy=self.velocity.tolist() if enabled else [0.,0.],
            acceleration_xy=self.acceleration.tolist() if enabled else [0.,0.],observed_current_pair=observed,
            derivative_three_pairs_qualified=bool(proof.get('qualified')),enabled=enabled,
            source_age_s=None if self.last_observed is None else t-self.last_observed,
            calibrated_camera_pose=False,future_camera_command_observed=False,noise_only_control=self.noise_only)
        return self.latest

    def inputs(self,t,body,motion):
        b=copy.deepcopy(body);m=copy.deepcopy(motion);offset=np.zeros(2) if self.noise_only else self.offset
        if b.get('center') is not None:b['center']=(np.asarray(b['center'],float)-offset).tolist()
        layout=m.get('layout_observation')
        if layout and layout.get('center') is not None:layout['center']=(np.asarray(layout['center'],float)-offset).tolist()
        d=m.get('dt_s');shift=m.get('estimate')
        if d and shift is not None and 'dx' in shift:
            previous=next((q for q in reversed(self.history) if abs(q[0]-(t-d))<1e-6),None)
            if previous is not None:
                difference=np.zeros(2) if self.noise_only else self.offset-previous[1]
                covariance=(self.variance-previous[2]);covariance=(covariance+covariance.T)/2
                intervals=[q for q in self.history if t-d+1e-6<q[0]<=t+1e-9]
                fully_observed=bool(intervals and all(q[3] for q in intervals))
                if self.shared_endpoints and fully_observed:
                    # Intermediate background-image errors cancel when all
                    # adjacent measured increments are added. Only two image
                    # endpoints remain; preserve every extra residual term.
                    covariance-=4.*max(0,len(intervals)-1)*np.eye(2)
                    assert np.linalg.eigvalsh(covariance).min()>-1e-7
                raw=np.array([shift['dx'],shift['dy']],float);value=raw-difference
                shift.update(dx=float(value[0]),dy=float(value[1]),
                    raw_image_displacement_before_background_reference=raw.tolist(),
                    subtracted_background_displacement=difference.tolist())
                foreground=np.asarray(shift.get('displacement_covariance_px2',np.eye(2)*4.))
                if self.preserve_foreground_floor and np.any(covariance):
                    # The inherited observer floors the foreground at4px².
                    # Apply that sensor floor BEFORE summing background noise;
                    # a1px² LK proxy must not consume the camera's4px² floor.
                    eigen,U=np.linalg.eigh((foreground+foreground.T)/2)
                    foreground=(U*np.maximum(eigen,4.))@U.T
                shift['displacement_covariance_px2']=(foreground+covariance).tolist()
                shift['background_reference_added_covariance_px2']=covariance.tolist()
                m['background_reference_interval']=dict(fully_observed=fully_observed,
                    measured_intervals=len(intervals),shared_endpoint_floor_enabled=self.shared_endpoints,
                    background_pair_floor_variance_px2=4.)
        return b,m

    def attach(self,row,original,*,readout_in_image_coordinates=False):
        r=copy.deepcopy(row);relative=copy.deepcopy(r.get('coefficient'));meta=copy.deepcopy(self.latest)
        offset=np.zeros(2) if self.noise_only else self.offset
        velocity=np.zeros(2) if self.noise_only else np.asarray(meta['velocity_xy'])
        acceleration=np.zeros(2) if self.noise_only else np.asarray(meta['acceleration_xy'])
        meta.update(offset_xy=offset.tolist(),velocity_xy=velocity.tolist(),acceleration_xy=acceleration.tolist(),
            relative_coefficient=relative,correlation_s=TAU,
            readout_in_image_coordinates=bool(readout_in_image_coordinates),
            current_native_detector_and_motor_evidence_unchanged=True,
            gravity_wind_drag_hidden_rotor_still_owned_by_unknown=True,
            background_is_coordinate_motion_not_a_third_mechanical_force=True,
            covariance_is_engineering_proxy=True)
        r['background_frame_motion']=meta;r['background_relative_motion']=r['motion']
        r['body']=copy.deepcopy(original['body']);r['motion']=copy.deepcopy(original['motion'])
        if relative:
            r['coefficient']=dict(relative,position=(np.asarray(relative['position'])+offset).tolist(),
                velocity=(np.asarray(relative['velocity'])+velocity).tolist(),
                acceleration=(np.asarray(relative['acceleration'])+acceleration).tolist())
            r['force']['coordinate_frame']='dominant_background_relative_translation_proxy'
            r['force']['apparent_camera_acceleration_not_mechanical_force']=acceleration.tolist()
            from physical_publication import evaluate
            r['physical']=evaluate(r,[.1,.2,.4])
        return r

def predict(row,horizons):
    from physical_publication import evaluate
    r=copy.deepcopy(row);info=r.pop('background_frame_motion')
    r['coefficient']=info['relative_coefficient']
    values=evaluate(r,horizons);v=np.asarray(info['velocity_xy']);a=np.asarray(info['acceleration_xy'])
    offset=np.asarray(info['offset_xy']) if info.get('readout_in_image_coordinates') else np.zeros(2)
    return {str(float(h)):(np.asarray(values[str(float(h))])+offset+float(h)*v+kernel(-1/float(info['correlation_s']),float(h))*a).tolist() for h in horizons}
