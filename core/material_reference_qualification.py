"""Presence and image motion do not prove that a contracted support is whole.

Only withhold reference-point recalibration on geometric support loss. Do not
smooth the trajectory or alter actual detector bounds, FOV extent, force or
the current motion state. Geometry bounds are an engineering support prior.
"""
import copy
import numpy as np
from calibrated_force_rollout import envelope


class MaterialReferenceQualification:
    def __init__(self):
        self.offset=np.zeros(2);self.bounds=None;self.stamp=None;self.last=None;self.observations=None

    def step(self,row):
        r=copy.deepcopy(row);t=float(r['t']);obs=r.get('state',{}).get('observations');c=r.get('coefficient')
        if (obs is not None and self.observations is not None and obs<self.observations) or (self.last is not None and t-self.last>.5):
            self.offset=np.zeros(2);self.bounds=None;self.stamp=None
        self.last=t;self.observations=obs
        shift=r['body'].get('measurement_reference_shift_xy')
        if shift is not None:
            shift=np.asarray(shift,float);self.offset-=shift
            if self.bounds is not None:self.bounds-=np.tile(shift,2)
        if c is None:return r
        current=envelope(r);held=False
        if current is not None:
            b=np.asarray(r['body']['bbox'],float)-np.tile(c['position'],2)
            if self.bounds is not None and self.stamp is not None and 0<t-self.stamp<=.12+1e-8:
                # Existing relative displacement noise floor is2px. Both
                # sides contribute to an extent, hence4px, not a test-fit cut.
                contained=bool(np.all(b[:2]>=self.bounds[:2]-2.) and np.all(b[2:]<=self.bounds[2:]+2.))
                contraction=bool(np.any(b[2:]-b[:2]<self.bounds[2:]-self.bounds[:2]-4.))
                held=contained and contraction
            if not held:
                self.offset=current[0]-np.asarray(c['position']);self.bounds=b;self.stamp=t
        if r.get('conditional_unknown_force'):
            r['conditional_unknown_force']['position_offset']=self.offset.tolist()
        r['material_reference_qualification']=dict(current_complete_reference_withheld_for_support_loss=held,
            held_offset_xy=self.offset.tolist(),last_trusted_reference_t=self.stamp,
            current_native_reference_candidate=current is not None,
            current_p_v_a_force_detector_and_FOV_extent_unchanged=True,
            uncertainty_is_engineering_prior=True,body_heading_used=False)
        from physical_publication import evaluate
        r['physical']=evaluate(r,[.1,.2,.4]);return r


class MaterialReferenceResponse:
    def __init__(self,response):self.response=response;self.reference=MaterialReferenceQualification()
    def step(self,row):return self.reference.step(self.response.step(row))
