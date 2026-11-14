"""Separate measured position correction from whole-support reference bias.

bbox_center-p = (bbox_center-z_material) + (z_material-p).
Weak support can measure z_material without measuring the full envelope.
Withholding a geometric reference must not suppress its current translation.
"""
import copy
import numpy as np
from calibrated_force_rollout import envelope


class SeparatedNativeReference:
    def __init__(self):
        self.geometry_offset=np.zeros(2);self.position_correction=np.zeros(2)
        self.bounds=None;self.stamp=None;self.last=None;self.observations=None
    def step(self,row):
        r=copy.deepcopy(row);t=float(r['t']);obs=r.get('state',{}).get('observations');c=r.get('coefficient');b=r['body']
        if (obs is not None and self.observations is not None and obs<self.observations) or (self.last is not None and t-self.last>.5):
            self.geometry_offset=np.zeros(2);self.position_correction=np.zeros(2);self.bounds=None;self.stamp=None
        self.last=t;self.observations=obs
        shift=b.get('measurement_reference_shift_xy')
        if shift is not None:
            shift=np.asarray(shift,float);self.geometry_offset-=shift
            if self.bounds is not None:self.bounds-=np.tile(shift,2)
        if c is None:return r
        position_observed=bool(b.get('detected') and not b.get('center_provisional') and r.get('motion',{}).get('current_observation'))
        z=np.asarray(b['center'],float) if position_observed else None
        if z is not None:self.position_correction=z-np.asarray(c['position'])
        current=envelope(r) if z is not None else None;held=False
        if current is not None:
            bounds=np.asarray(b['bbox'],float)-np.tile(z,2)
            if self.bounds is not None and self.stamp is not None and 0<t-self.stamp<=.12+1e-8:
                contained=bool(np.all(bounds[:2]>=self.bounds[:2]-2.) and np.all(bounds[2:]<=self.bounds[2:]+2.))
                contraction=bool(np.any(bounds[2:]-bounds[:2]<self.bounds[2:]-self.bounds[:2]-4.))
                held=contained and contraction
            if not held:self.geometry_offset=current[0]-z;self.bounds=bounds;self.stamp=t
        if r.get('conditional_unknown_force'):
            r['conditional_unknown_force']['position_offset']=(self.geometry_offset+self.position_correction).tolist()
        r['native_reference_roles']=dict(current_position_evidence_used=position_observed,
            geometry_reference_withheld_for_support_loss=held,geometry_offset_xy=self.geometry_offset.tolist(),
            current_position_correction_xy=self.position_correction.tolist(),last_geometric_reference_t=self.stamp,
            current_internal_p_v_a_force_detector_and_FOV_unchanged=True,shape_qualification_is_engineering_prior=True)
        from physical_publication import evaluate
        r['physical']=evaluate(r,[.1,.2,.4]);return r


class SeparatedNativeReferenceResponse:
    def __init__(self,response):self.response=response;self.reference=SeparatedNativeReference()
    def step(self,row):return self.reference.step(self.response.step(row))
