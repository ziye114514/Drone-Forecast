"""Reuse the existing side-axis primitive without NumPy2.4/Numba coupling.

Exact integer ON/OFF log-count display input and existing point sampling.
This accelerates the existing LSD/PCA observer; it is not a new detector.
"""
import cv2
import numpy as np


class NativeAxisWorkspace:
    def __init__(self):
        self.table=np.uint8(np.clip(np.log1p(np.arange(256,dtype=np.float32))/4,0,1)*255)
        self.detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        self.provenance=dict(integer_native_count_input=True,original_point_sampling_preserved=True,
            reconstruction_used=False,Numba_required=False)

    @staticmethod
    def points(mask):
        found=cv2.findNonZero(mask.view(np.uint8))
        if found is None:return np.empty((0,2),float)
        points=found.reshape(-1,2)
        # Keep exactly the old row-major sampling if an OpenCV build returns
        # another traversal order. No coordinate quantization is introduced.
        key=points[:,1].astype(np.int64)*mask.shape[1]+points[:,0]
        if np.any(np.diff(key)<0):points=points[np.argsort(key,kind='stable')]
        if len(points)>5000:points=points[::int(np.ceil(len(points)/5000))]
        return points.astype(float)

    def render(self,crop):
        if crop.dtype!=np.float32:
            return np.uint8(np.clip(np.log1p(crop)/4,0,1)*255)
        # Only the native integer-count frontend installs this workspace.
        # Values>=54 all map to255 under the original expression.
        return cv2.LUT(np.minimum(crop,54).astype(np.uint8),self.table)

    def detect(self,image):return self.detector.detect(image)[0]
