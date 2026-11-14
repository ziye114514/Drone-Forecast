"""Robust reference qualification, not a low-pass trajectory predictor.

A frame can contain strong rotor events but only a partial envelope. Keep a
real, temporally consistent native reference packet; do not change detector
boxes, current physical p/v/a, hidden-force ownership, or future drive inputs.
"""
from collections import deque
import copy
import numpy as np
from calibrated_force_rollout import envelope


def medoid(packets):
    z=np.asarray([np.r_[p['offset'],np.asarray(p['extent'])/2] for p in packets])
    costs=np.linalg.norm(z[:,None]-z[None,:],axis=2).sum(1)
    # Selecting an observed packet avoids fabricating a new geometric shape.
    return int(len(costs)-1-np.argmin(costs[::-1]))


class StableNativeReference:
    def __init__(self):
        self.samples=deque(maxlen=3);self.offset=np.zeros(2);self.extent=None
        self.stamp=None;self.last=None;self.observations=None;self.initialized=False

    def step(self,row):
        r=copy.deepcopy(row);c=r.get('coefficient');t=float(r['t']);obs=r.get('state',{}).get('observations')
        if (obs is not None and self.observations is not None and obs<self.observations) or (self.last is not None and t-self.last>.5):
            self.samples.clear();self.offset=np.zeros(2);self.extent=None;self.stamp=None;self.initialized=False
        self.last=t;self.observations=obs
        shift=r['body'].get('measurement_reference_shift_xy')
        if shift is not None:
            self.offset-=np.asarray(shift,float);self.samples.clear();self.initialized=False
        if c is None:return r
        candidate=envelope(r);selected=None
        if candidate is not None:
            if self.samples and t-self.samples[-1]['t']>.12+1e-8:self.samples.clear();self.initialized=False
            b=np.asarray(r['body']['bbox'],float)
            packet=dict(t=t,offset=(candidate[0]-np.asarray(c['position'])).tolist(),extent=(b[2:]-b[:2]-12+1).tolist())
            if min(packet['extent'])>0:
                self.samples.append(packet)
                if not self.initialized:selected=packet;self.initialized=True
                elif len(self.samples)>=3:selected=list(self.samples)[medoid(self.samples)]
        if selected is not None:
            self.offset=np.asarray(selected['offset']);self.extent=selected['extent'];self.stamp=selected['t']
        if r.get('conditional_unknown_force'):
            r['conditional_unknown_force']['position_offset']=self.offset.tolist()
        if r.get('native_fov_observation') and self.extent is not None:
            r['native_fov_observation'].update(full_extent_px=list(self.extent),last_complete_source_t=self.stamp)
        r['stable_native_reference']=dict(offset=self.offset.tolist(),full_extent_px=self.extent,
            selected_reference_t=self.stamp,qualified_recent_samples=len(self.samples),current_reference_qualified=candidate is not None,
            detector_boxes_unchanged=True,current_p_v_a_unchanged=True,reference_is_support_prior_not_COM_truth=True,
            sample_count_not_used_to_reduce_noise=True)
        from physical_publication import evaluate
        r['physical']=evaluate(r,[.1,.2,.4]);return r


class StableReferenceResponse:
    def __init__(self,response):self.response=response;self.reference=StableNativeReference()
    def step(self,row):return self.reference.step(self.response.step(row))
