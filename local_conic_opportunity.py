"""Global support flatness is not local rotor-disc flatness.

This grants only a search opportunity, never a rotor detection or force input.
Reuse the existing .16 line/conic threshold and already filtered contours.
"""
import cv2
import numpy as np


def nonflat_contour_present(contours):
    for contour in contours:
        m=cv2.moments(contour)
        if m['m00']<=0:continue
        covariance=np.array([[m['mu20'],m['mu11']],[m['mu11'],m['mu02']]])/m['m00']
        values=np.linalg.eigvalsh(covariance)
        if np.sqrt(max(values[0],0.)/max(values[1],1e-9))>.16:return True
    return False
