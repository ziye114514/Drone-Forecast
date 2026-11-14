"""Explicit rotor-only foreground + native background-reference candidate.

The existing front still owns every drone/rotor decision. Background evidence
only changes coordinates used by the same two-force state. No manual ROI,
annotation, video ID, NN, reconstructed intensity or hidden RPM enters here.
"""
import copy
import numpy as np
from observed_rotor_predictor import LocalContrastObservedPredictor
from background_relative_force import BackgroundReference
from background_unknown_response import BackgroundMotion

class ForceReferenceBridge:
    def __init__(self,inner,reference):
        self.inner=inner;self.reference=reference;self.original=None;self.relative_body=None;self.relative_motion=None
        self.static_observer=None;self.static_result=None
    def __getattr__(self,name):return getattr(self.inner,name)
    @property
    def two_point_init(self):return self.inner.two_point_init
    @two_point_init.setter
    def two_point_init(self,value):
        self.inner.two_point_init=value
        if self.static_observer is not None:self.static_observer.two_point_init=value
    def step(self,t,body,u,*args,**kwargs):
        self.original=dict(body=copy.deepcopy(body),motion=copy.deepcopy(kwargs.get('motion') or {}))
        if self.static_observer is not None:
            self.static_result=self.static_observer.step(t,copy.deepcopy(body),u,*args,**copy.deepcopy(kwargs))
        b,m=self.reference.inputs(t,body,kwargs.get('motion') or {})
        self.relative_body=b;self.relative_motion=m;kwargs=dict(kwargs,motion=m)
        return self.inner.step(t,b,u,*args,**kwargs)

class RelativePhysicalInput:
    def __init__(self,inner,bridge):self.inner=inner;self.bridge=bridge
    def __getattr__(self,name):return getattr(self.inner,name)
    def step(self,row):
        r=copy.deepcopy(row);r['body']=copy.deepcopy(self.bridge.relative_body)
        r['motion']=copy.deepcopy(self.bridge.relative_motion)
        return self.inner.step(r)

class ImageReadout:
    def __init__(self,inner,bridge):self.inner=inner;self.bridge=bridge
    def __getattr__(self,name):return getattr(self.inner,name)
    def step(self,row):
        r=self.bridge.reference.attach(row,self.bridge.original,readout_in_image_coordinates=True)
        return self.inner.step(r)

class BackgroundTranslationPredictor(LocalContrastObservedPredictor):
    use_shared_background_endpoints=False
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.legacy_force or self.inverse_normal_response is None:
            raise ValueError('Background reference requires the current native rotor two-force entry')
        reference=BackgroundReference(shared_endpoints=self.use_shared_background_endpoints)
        if self.use_shared_background_endpoints:
            from background_endpoint_observer import BackgroundEndpointObserver
            self.force=BackgroundEndpointObserver()
        self.force=ForceReferenceBridge(self.force,reference)
        self.inverse_normal_response=RelativePhysicalInput(self.inverse_normal_response,self.force)
        self.learned_response=ImageReadout(self.learned_response,self.force)
        original=self.front.step;self.background=BackgroundMotion(translation_only=True)
        def step(events,t):
            frame=original(events,t);body=frame['body'];point=body.get('center')
            proxy=dict(t=t,body=body,coefficient=None if point is None else dict(position=point))
            count=frame['on']+frame['off'];info=self.background.step(proxy,(count,np.zeros_like(count)))
            ev=dict(t=t,proof=info,pair=None)
            if self.background.transforms and abs(self.background.transforms[-1]['t']-t)<1e-9:
                q=self.background.transforms[-1];M=np.asarray(q['M'])
                p=np.asarray(point if point is not None else [640.,360.],float)
                delta=p-np.linalg.solve(M[:,:2],p-M[:,2])
                ev['pair']=dict(source_t=q['source'],t=t,displacement_xy=delta.tolist(),
                    covariance_px2=q['Q'].tolist(),affine=M.tolist(),local_background_point=p.tolist())
            reference.advance(ev);frame['background_reference_evidence']=ev
            return frame
        self.front.step=step

class BackgroundEndpointPredictor(BackgroundTranslationPredictor):
    use_shared_background_endpoints=True
