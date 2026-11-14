"""Optimize prediction, not observation gates: remove first-frame position bias.

Two fixed alternatives: a slow source-offset estimate, and geometry-derived
velocity that is invariant to an unknown constant image-coordinate offset.
"""
from pathlib import Path
from collections import deque
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage107')]
from center107 import BiasFilter,SimpleKF,predict,update,coefficient,PARAMS as KF
PARAMS=dict(offset_tau_s=1.,motion_window_s=.08,motion_min_span_s=.04,motion_min_points=3,
    motion_max_gap_s=.04000001,geometry_position_sigma_floor_px=2.,velocity_sigma_floor_px_s=30.,
    body_KF='Original107 four-state CV, body sigma6px, acceleration sigma6400px/s2. No tuning of the baseline.',
    offset='Source-specific body-minus-geometry EWMA with1s time constant, using current same-time data only. Replaces a permanent single-frame anchor.',
    velocity='Weighted least-squares current geometric-center slope over<=80ms, >=3 samples/40ms. Constant geometric-center/source offset cancels. Update velocity in one body-position KF; no absolute geometry position observation.',
    selection='Current count model preferred, else current occupancy model. All original129 full-boundary/five-frame gates retained. No future label used for selection.')
def current_sources(models):
    return {k:d for k,d in models.items() if d.get('current') and d.get('model_geometric_center') is not None}
class AdaptiveOffset:
    def __init__(self):self.offsets={};self.times={};self.filter=BiasFilter(False);self.seen=None
    def step(self,t,body,models):
        if self.seen is not None and t-self.seen>.5:self.__init__()
        if body.get('detected'):self.seen=t
        sources=current_sources(models)
        for name,d in sources.items():
            target=np.asarray(body['center'])-np.asarray(d['model_geometric_center'])
            if name not in self.offsets:self.offsets[name]=target
            else:
                dt=min(.04,max(0.,t-self.times[name]));a=-np.expm1(-dt/PARAMS['offset_tau_s']);self.offsets[name]+=a*(target-self.offsets[name])
            self.times[name]=t
        name='count' if 'count' in sources else 'occupancy' if 'occupancy' in sources else None
        d=copy.deepcopy(sources[name]) if name else dict(current=False)
        if name:d['model_geometric_center']=(np.asarray(d['model_geometric_center'])+self.offsets[name]).tolist()
        c,diag=self.filter.step(t,body,d,dict(valid=False))
        return dict(coefficient=c,source=name,diagnostic=diag,offsets={k:v.tolist() for k,v in self.offsets.items()})
class RelativeVelocity:
    def __init__(self):self.filter=SimpleKF();self.history={k:deque() for k in ['count','occupancy']};self.seen=None
    def slope(self,name,t,d):
        h=self.history[name]
        if not d.get('current') or d.get('model_geometric_center') is None:return None
        if h and t-h[-1][0]>PARAMS['motion_max_gap_s']:h.clear()
        m=d['measurement'];sigma=max(PARAMS['geometry_position_sigma_floor_px'],m.get('center_noise_proxy_px',0.),m.get('fit',{}).get('rms_px',0.))
        h.append((t,np.asarray(d['model_geometric_center']),sigma))
        while h and t-h[0][0]>PARAMS['motion_window_s']+1e-8:h.popleft()
        if len(h)<PARAMS['motion_min_points'] or t-h[0][0]<PARAMS['motion_min_span_s']-1e-8:return None
        tau=np.array([a[0]-t for a in h]);p=np.array([a[1] for a in h]);weight=1/np.array([a[2] for a in h])**2
        mu=np.sum(weight*tau)/sum(weight);center=np.sum(weight[:,None]*p,axis=0)/sum(weight);S=float(np.sum(weight*(tau-mu)**2))
        v=np.sum((weight*(tau-mu))[:,None]*(p-center),axis=0)/S
        residual=p-(center+(tau-mu)[:,None]*v);chi=float(np.sum(weight[:,None]*residual**2)/max(2*(len(h)-2),1))
        variance=max(PARAMS['velocity_sigma_floor_px_s']**2,max(1.,chi)/S)
        return dict(velocity=v.tolist(),variance=variance,samples=len(h),span=t-h[0][0],residual_scale=chi)
    def step(self,t,body,models):
        if self.seen is not None and t-self.seen>.5:self.__init__()
        if body.get('detected'):self.seen=t
        slopes={k:self.slope(k,t,models.get(k,{})) for k in self.history}
        z=np.asarray(body['center']) if body.get('detected') else None;c=self.filter.step(t,z)
        name='count' if slopes['count'] is not None else 'occupancy' if slopes['occupancy'] is not None else None
        if name is not None and c is not None and body.get('detected'):
            q=slopes[name];H=np.c_[np.zeros((2,2)),np.eye(2)]
            self.filter.x,self.filter.P=update(self.filter.x,self.filter.P,np.array(q['velocity']),R=np.eye(2)*q['variance'],H=H)
            c=coefficient(self.filter.x)
        return dict(coefficient=c,source=name,motion=slopes.get(name) if name else None)
class Optimized:
    def __init__(self):self.offset=AdaptiveOffset();self.motion=RelativeVelocity()
    def step(self,t,body,models):return dict(adaptive_offset=self.offset.step(t,body,models),relative_velocity=self.motion.step(t,body,models))
