"""Same147 arc hypotheses/selection; cached indices and batched first residuals.

No thresholds, point sampling policy, seed, refinement or uncertainty changes.
The OpenCV AMS calls and the original candidate order are retained.
"""
from functools import lru_cache
import numpy as np
from event_drone_forecast.geometry import _fit,_distance,_coverage,_uncertainty
from research.stage147.curvature147 import PARAMS

# Explicit opt-in only. A failed native load is an error, never a silent
# backend change. The same seed order, refinement and acceptance gates apply.
NATIVE_BACKEND=None
CACHE_EQUIVALENT_REFINEMENT=False
CACHE_COUNTS=dict(lookups=0,hits=0)

def configure_native(backend=None):
    global NATIVE_BACKEND
    NATIVE_BACKEND=backend

def residual(points,p):
    return _distance(points,p) if NATIVE_BACKEND is None else NATIVE_BACKEND.distance(points,p)


@lru_cache(maxsize=256)
def hypothesis_indices(n):
    rng=np.random.default_rng(147)
    a=np.array([np.sort(rng.choice(n,7,replace=False)) for _ in range(PARAMS['random_hypotheses_per_arc'])])
    a.setflags(write=False)
    return a


def distances_batch(points,parameters):
    if NATIVE_BACKEND is not None:return NATIVE_BACKEND.distances_batch(points,parameters)
    p=np.asarray(parameters);delta=points[None,:,:]-p[:,None,:2]
    # Scalar trig retains the reference's scalar NumPy arithmetic.
    co=np.array([np.cos(a) for a in p[:,4]])[:,None];si=np.array([np.sin(a) for a in p[:,4]])[:,None]
    x=delta[:,:,0]*co+delta[:,:,1]*si;y=-delta[:,:,0]*si+delta[:,:,1]*co
    a=p[:,2,None];b=p[:,3,None]
    gradient=2*np.sqrt((x/a**2)**2+(y/b**2)**2)
    return ((x/a)**2+(y/b)**2-1)/np.maximum(gradient,1e-6)


def fit_arc_exact(points,parent_extent):
    points=np.asarray(points,float).reshape(-1,2)
    if len(points)<22:return None
    if len(points)>4096:
        from fast_geometry_compute import fit_arc_fast
        return fit_arc_fast(points,parent_extent)
    if NATIVE_BACKEND is not None and hasattr(NATIVE_BACKEND,'search'):
        result=NATIVE_BACKEND.search(points,parent_extent,hypothesis_indices(len(points)))
        if result is None:return None
        p,supported,rms,coverage,support=result
        uncertainty=getattr(NATIVE_BACKEND,'uncertainty',_uncertainty)
        sigma,angle,condition=uncertainty(supported,p)
        if sigma>PARAMS['maximum_center_sigma_px'] or condition>1500:return None
        return dict(parameters=p,points=supported,rms=rms,coverage=coverage,sigma=sigma,angle_sigma=angle,
            condition=condition,support_fraction=support)
    minimum=max(22,.60*len(points));best=None;refinements={}
    def shape(p):return p is not None and 10<=2*p[2]<=parent_extent*1.6 and p[3]>=2.5 and p[3]/p[2]>=.12
    def consider(p,inliers=None):
        nonlocal best
        if not shape(p):return False
        if inliers is None:inliers=abs(residual(points,p))<=1.8
        if inliers.sum()<minimum:return False
        key=None
        if CACHE_EQUIVALENT_REFINEMENT:
            key=inliers.tobytes();CACHE_COUNTS['lookups']+=1
            if key in refinements:
                CACHE_COUNTS['hits']+=1;return refinements[key]
            refinements[key]=False
        p=_fit(points[inliers])
        if p is None:return False
        distances=residual(points,p);inliers=abs(distances)<=1.8;supported=points[inliers]
        if len(supported)<minimum or not shape(p):return False
        rms=float(np.sqrt(np.mean(distances[inliers]**2)));coverage=_coverage(supported,p)
        if coverage<PARAMS['minimum_arc_coverage'] or rms>PARAMS['maximum_arc_rms_px']:return False
        score=inliers.mean()+.2*coverage-.1*rms
        if best is None or score>best[0]:best=score,p,supported,rms,coverage,float(inliers.mean())
        stop=bool(inliers.mean()>=.90 and coverage>=.72 and rms<.8)
        if key is not None:refinements[key]=stop
        return stop
    stop=consider(_fit(points))
    if not stop:
        indices=hypothesis_indices(len(points))
        parameters=[_fit(points[j]) for j in indices] if NATIVE_BACKEND is None else NATIVE_BACKEND(points,indices)
        valid=[p for p in parameters if shape(p)]
        if valid:
            inliers=abs(distances_batch(points,valid))<=1.8
            for p,keep in zip(valid,inliers):
                if consider(p,keep):break
    if best is None:return None
    _,p,supported,rms,coverage,support=best
    sigma,angle,condition=_uncertainty(supported,p)
    if sigma>PARAMS['maximum_center_sigma_px'] or condition>1500:return None
    return dict(parameters=p,points=supported,rms=rms,coverage=coverage,sigma=sigma,angle_sigma=angle,condition=condition,support_fraction=support)
