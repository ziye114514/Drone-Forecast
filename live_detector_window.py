"""Native SDK accumulation with a source-time cutoff, no grayscale recovery."""
import numpy as np

class DetectorWindow:
    def __init__(self,w,h,window_ms):
        if window_ms not in (20,60):raise ValueError('Supported fixed windows are20 or60ms')
        from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
        self.algorithm=OnDemandFrameGenerationAlgorithm(w,h,int(window_ms*1000))
        self.window_ms=window_ms;self.last_cutoff=None;self.last_generated=None
        self.image=np.empty((h,w,3),np.uint8)

    def feed(self,clean,t):
        if self.last_cutoff is not None:assert t>self.last_cutoff
        if len(clean):assert clean['t'].max()<round(t*1e6)
        self.algorithm.process_events(clean);self.last_cutoff=t

    def generate(self,t):
        assert t==self.last_cutoff
        if self.last_generated is not None:assert t>self.last_generated
        self.algorithm.generate(round(t*1e6),self.image);self.last_generated=t
        return self.image
