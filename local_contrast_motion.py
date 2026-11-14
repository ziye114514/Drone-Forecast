"""Local contrast normalization for native event-count registration.

The same rotor can change event energy without translating. Normalize local
offset/scale after the existing blur; keep the31px native local-statistic
window. Counts used for geometry and frequency are never altered.
This is a conventional photometric descriptor, not a new rotation estimator.
"""
import cv2
import numpy as np


def normalize(image,size=31):
    image=np.asarray(image,np.float32)
    mean=cv2.boxFilter(image,-1,(size,size),borderType=cv2.BORDER_REFLECT)
    second=cv2.boxFilter(image*image,-1,(size,size),borderType=cv2.BORDER_REFLECT)
    variance=np.maximum(second-mean*mean,0.)
    # A zero/constant event region contributes no texture, not fabricated
    # high confidence. Relative machine precision is not an event threshold.
    floor=np.finfo(np.float32).eps*max(float(np.max(variance)),1.)
    result=np.zeros_like(image);valid=variance>floor
    result[valid]=(image[valid]-mean[valid])/np.sqrt(variance[valid])
    return result
