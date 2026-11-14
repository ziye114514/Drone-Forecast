"""Reuse LSD workspace and the EXACT existing log(count) input, no reconstruction."""
import time
import cv2,numpy as np
from native_counts_numba import njit


@njit(cache=True,nogil=True)
def lookup(crop,table):
    image=np.empty(crop.shape,np.uint8);valid=True
    for y in range(crop.shape[0]):
        for x in range(crop.shape[1]):
            v=crop[y,x]
            if not np.isfinite(v) or v<0 or (v<54 and v!=int(v)):
                valid=False;image[y,x]=0
            else:image[y,x]=table[54 if v>=54 else int(v)]
    return image,valid


class BodyWorkspace:
    def __init__(self):
        began=time.perf_counter();self.tables={};self.fallbacks=0;self.calls=0
        for dtype in (np.dtype('float32'),np.dtype('float64')):
            table=np.uint8(np.clip(np.log1p(np.arange(55,dtype=dtype))/4,0,1)*255)
            self.tables[dtype]=table
            # Both contiguous and real strided ROI signatures are warmed.
            lookup(np.zeros((4,4),dtype),table);lookup(np.zeros((4,4),dtype)[:,::2],table)
        self.detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        self.provenance=dict(warm_compile_load_ms=(time.perf_counter()-began)*1000,
            existing_log_count_image_exact=True,G_channel_or_gray_reconstruction=False,
            cached_installed_OpenCV_LSD=True,count_or_threshold_sampling=False)

    def render(self,crop):
        self.calls+=1;table=self.tables.get(crop.dtype)
        if table is not None:
            image,valid=lookup(crop,table)
            if valid:return image
        self.fallbacks+=1
        return np.uint8(np.clip(np.log1p(crop)/4,0,1)*255)

    def detect(self,image):return self.detector.detect(image)[0]
