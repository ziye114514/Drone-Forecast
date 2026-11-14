"""Coarse discovery, native-resolution matching and explicit fallback.

Only inter-frame search is downsampled; event filtering, geometry and final
peak evaluation use the unchanged native arrays. This is approximate search,
not a theorem that the whole-image maximum is always retained.
"""
import cv2
import numpy as np


def normalized_search(target,patch,factor=2,centered=False):
    method=cv2.TM_CCOEFF_NORMED if centered else cv2.TM_CCORR_NORMED
    def match(a,b):return cv2.matchTemplate(a,b,method)*float(np.linalg.norm(b))
    shape=(target.shape[0]-patch.shape[0]+1,target.shape[1]-patch.shape[1]+1)
    if min(shape)<10 or min(patch.shape)<32:
        return match(target,patch),(0,0),dict(coarse=False,reason='small_native_search')
    a=np.ascontiguousarray(target[::factor,::factor]);b=np.ascontiguousarray(patch[::factor,::factor])
    discovery=match(a,b);remaining=discovery.copy();candidates=[]
    for _ in range(2):
        y,x=np.unravel_index(int(np.argmax(remaining)),remaining.shape)
        if not np.isfinite(remaining[y,x]):break
        candidates.append((int(y)*factor,int(x)*factor))
        remaining[max(0,y-2):y+3,max(0,x-2):x+3]=-np.inf
    best=None
    for y,x in candidates:
        y0=max(0,y-4);x0=max(0,x-4);Y=min(shape[0],y+5);X=min(shape[1],x+5)
        score=match(target[y0:Y+patch.shape[0]-1,x0:X+patch.shape[1]-1],patch)
        value=float(score.max())
        if best is None or value>best[0]:best=(value,score,(y0,x0))
    _,score,origin=best;iy,ix=np.unravel_index(int(np.argmax(score)),score.shape)
    # A candidate outside the fine neighbourhood can leave a maximum on its
    # edge. Repeat the original full native search in that case, preserving
    # failures as failures rather than manufacturing a boundary displacement.
    local_edge=iy in (0,score.shape[0]-1) or ix in (0,score.shape[1]-1)
    global_edge=iy+origin[0] in (0,shape[0]-1) or ix+origin[1] in (0,shape[1]-1)
    if local_edge and not global_edge:
        return match(target,patch),(0,0),dict(coarse=True,fallback=True,reason='fine_peak_at_internal_boundary',candidate_count=len(candidates))
    return score,origin,dict(coarse=True,fallback=False,candidate_count=len(candidates),factor=factor,
        native_search_shape=list(shape),native_refinement_shape=list(score.shape))
