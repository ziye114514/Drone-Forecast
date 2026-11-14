"""Position evidence can be ready before the whole-object reference is ready.

Only initialise the geometric reference robustly. After the existing five
qualified-reference warm-up, preserve the old readout exactly. This is not
trajectory smoothing or a persistent partial-support correction.
"""
from collections import deque
import copy
import numpy as np
from calibrated_force_rollout import envelope


class BootstrapNativeReference:
    def __init__(self):
        self.samples=deque(maxlen=3);self.count=0;self.geometry=np.zeros(2);self.position=np.zeros(2)
        self.last=None;self.observations=None
    def step(self,row):
        r=copy.deepcopy(row);t=float(r['t']);b=r['body'];c=r.get('coefficient');obs=r.get('state',{}).get('observations')
        new_track=bool((obs is not None and self.observations is not None and obs<self.observations) or (self.last is not None and t-self.last>.5))
        new_anchor=bool(self.last is not None and b.get('measurement_source') in ('first_complete_rotor_region_anchor','rotor_region_initialization'))
        if new_track:
            self.samples.clear();self.count=0;self.geometry=np.zeros(2);self.position=np.zeros(2)
        elif new_anchor:self.samples.clear();self.count=0
        shift=b.get('measurement_reference_shift_xy')
        if shift is not None:
            self.geometry-=np.asarray(shift,float);self.samples.clear();self.count=0
        self.last=t;self.observations=obs
        if c is None:return r
        observed=bool(b.get('detected') and not b.get('center_provisional') and r.get('motion',{}).get('current_observation'))
        z=np.asarray(b['center'],float) if observed else None
        if z is not None:self.position=z-np.asarray(c['position'])
        current=envelope(r) if z is not None else None
        if current is not None:
            value=current[0]-z;self.count+=1;self.samples.append(value.copy())
            if self.count==1 or self.count>=5:self.geometry=value
            elif len(self.samples)==3:
                a=np.asarray(self.samples);cost=np.linalg.norm(a[:,None]-a[None,:],axis=2).sum(1)
                index=len(a)-1-int(np.argmin(cost[::-1]));self.geometry=a[index]
        boot=bool(self.count<5)
        if boot and r.get('conditional_unknown_force'):
            r['conditional_unknown_force']['position_offset']=(self.geometry+self.position).tolist()
            from physical_publication import evaluate
            r['physical']=evaluate(r,[.1,.2,.4])
        r['native_reference_bootstrap']=dict(active=boot,qualified_reference_observations=self.count,
            original_readout_exact_after_warmup=not boot,fresh_position_evidence_used=observed,
            geometric_reference_offset=self.geometry.tolist(),current_position_correction=self.position.tolist(),
            no_physical_state_force_detector_or_FOV_change=True,bootstrapped_reference_not_COM_truth=True)
        return r


class BootstrapReferenceResponse:
    def __init__(self,response):self.response=response;self.reference=BootstrapNativeReference()
    def step(self,row):return self.reference.step(self.response.step(row))
