"""Own only previously used rotor pixels, in reusable native-shaped planes.

Both existing motion routes mask the past image before filtering. Pixels
outside that support are therefore exactly unused, not sampled observations.
Keeping them zero avoids full-frame copies while protecting shared-slot reuse.
"""
import numpy as np


class NativeHistoryBuffer:
    def __init__(self,shape=(720,1280)):
        self.shape=shape;self.planes=tuple(np.zeros(shape,np.float32) for _ in range(2))
        self.mask=np.zeros(shape,bool);self.indices=np.empty(0,np.int64)

    def update(self,current,mask,bounds):
        if mask.shape!=self.shape:raise ValueError('Native history shape changed')
        x,y,width,height=bounds
        local=np.flatnonzero(mask[y:y+height,x:x+width])
        ids=(local//width+y)*self.shape[1]+local%width+x if width else np.empty(0,np.int64)
        for dst,src in zip(self.planes,current):
            flat=dst.reshape(-1);flat[self.indices]=0.;flat[ids]=src.reshape(-1)[ids]
        flat=self.mask.reshape(-1);flat[self.indices]=False;flat[ids]=True
        self.indices=ids
        return self.planes,self.mask
