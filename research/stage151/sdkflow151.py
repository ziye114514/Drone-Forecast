"""Independent current-event velocity, reusing Prophesee sparse optical flow.

One vote per native track, not per event. Rotor angular motion is an outlier
to a spatially distributed common translation; confidence can remain absent.
Bounding boxes select support only and never supply a velocity measurement.
"""
from pathlib import Path
import sys
import numpy as np
import metavision_sdk_cv as sdk

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/stage107'))
from center107 import SimpleKF,update,coefficient

PARAMS=dict(sdk_preset='FastObjects',minimum_tracks=4,minimum_track_span_s=.04,
    minimum_inlier_fraction=.60,minimum_spatial_span_fraction=.25,
    maximum_speed_px_s=5000.,velocity_sigma_floor_px_s=60.,
    maximum_scatter_px_s=120.,track_memory_s=.10,body_gap_s=.50,
    current_support_only=True,default_promoted=False)

def native_copy(buffer):
    q=buffer.numpy();info=buffer._buffer_info()
    assert info.ndim==1 and tuple(info.shape)==q.shape
    assert info.itemsize==q.dtype.itemsize
    if q.strides!=tuple(info.strides):
        assert q.base is not None
        q=np.lib.stride_tricks.as_strided(q,shape=tuple(info.shape),
            strides=tuple(info.strides),writeable=False)
    return q.copy()

class FlowObservation:
    def __init__(self,w,h):
        self.w=w;self.h=h;self.engine=sdk.SparseOpticalFlowAlgorithm(w,h)
        self.output=self.engine.get_empty_output_buffer();self.history={}
        self.last_body=None;self.last_t=None

    def step(self,events,t,body):
        if self.last_body is not None and t-self.last_body>PARAMS['body_gap_s']:
            self.__init__(self.w,self.h)
        if self.last_t is not None:assert t>self.last_t
        self.last_t=t
        result=dict(valid=False,reason='no_current_body',velocity=None,
            variance=None,independent_event_motion=True,source='SDK SparseOpticalFlow FastObjects',
            max_input_event_t=int(events['t'].max()) if len(events) else None)
        assert not len(events) or int(events['t'].max())<round(t*1e6)
        if not body.get('detected') or body.get('bbox') is None:return result
        self.last_body=t;b=np.asarray(body['bbox'],float);size=b[2:]-b[:2]
        pad=np.maximum(4.,.10*size);lo=np.maximum(0,b[:2]-pad)
        hi=np.minimum([self.w,self.h],b[2:]+pad)
        keep=(events['x']>=lo[0])&(events['x']<hi[0])&(events['y']>=lo[1])&(events['y']<hi[1])
        local=events[keep];result['input_events']=len(local)
        if not len(local):result['reason']='no_current_events';return result
        self.engine.process_events(local,self.output);q=native_copy(self.output)
        if len(q):
            assert np.all(q['t']<=local['t'][-1]) and np.all(q['t']>=local['t'][0])
            # A spinning patch can emit thousands of events. Its last track
            # estimate still receives only one vote in the common translation.
            _,ii=np.unique(q['id'][::-1],return_index=True);q=q[len(q)-1-ii]
        selected=[]
        for p in q:
            identifier=int(p['id']);stamp=float(p['t'])/1e6
            first,_=self.history.get(identifier,(stamp,stamp))
            self.history[identifier]=(first,stamp)
            if stamp-first<PARAMS['minimum_track_span_s']-1e-8:continue
            value=np.array([p['vx'],p['vy']],float)
            if not np.isfinite(value).all() or np.linalg.norm(value)>PARAMS['maximum_speed_px_s']:continue
            selected.append((value,np.array([p['center_x'],p['center_y']],float)))
        self.history={k:v for k,v in self.history.items() if t-v[1]<=PARAMS['track_memory_s']}
        result.update(current_tracks=len(q),mature_tracks=len(selected),reason='too_few_mature_tracks')
        if len(selected)<PARAMS['minimum_tracks']:return result
        velocities=np.array([p[0] for p in selected]);xy=np.array([p[1] for p in selected])
        median=np.median(velocities,axis=0);distance=np.linalg.norm(velocities-median,axis=1)
        spread=max(30.,1.4826*float(np.median(distance)))
        inlier=distance<=max(60.,2.5*spread)
        fraction=float(inlier.mean());v=np.median(velocities[inlier],axis=0)
        scatter=float(np.sqrt(np.mean(np.sum((velocities[inlier]-v)**2,axis=1))))
        spatial=np.ptp(xy[inlier],axis=0)/np.maximum(size,1.)
        valid=(int(inlier.sum())>=PARAMS['minimum_tracks'] and
            fraction>=PARAMS['minimum_inlier_fraction'] and
            max(spatial)>=PARAMS['minimum_spatial_span_fraction'] and
            scatter<=PARAMS['maximum_scatter_px_s'])
        result.update(valid=bool(valid),reason='qualified_common_event_motion' if valid else 'incoherent_or_local_motion',
            velocity=v.tolist() if valid else None,
            variance=max(PARAMS['velocity_sigma_floor_px_s']**2,scatter**2/max(int(inlier.sum()),1)) if valid else None,
            proposed_velocity=v.tolist(),inlier_tracks=int(inlier.sum()),inlier_fraction=fraction,
            scatter_px_s=scatter,spatial_span_fraction=spatial.tolist(),
            track_velocities=velocities.tolist(),track_centers=xy.tolist())
        return result

class MotionState:
    def __init__(self):self.filter=SimpleKF()
    def step(self,t,body,observation,baseline_coefficient,baseline_predictions,structure_supported=False):
        c=self.filter.step(t,body['center'] if body.get('detected') else None)
        active=bool(c is not None and body.get('detected') and observation.get('valid') and not structure_supported)
        if active:
            H=np.c_[np.zeros((2,2)),np.eye(2)]
            self.filter.x,self.filter.P=update(self.filter.x,self.filter.P,
                np.asarray(observation['velocity']),R=np.eye(2)*observation['variance'],H=H)
            c=coefficient(self.filter.x)
        predictions=dict(baseline_predictions)
        if active and baseline_coefficient is not None:
            c['acceleration']=list(baseline_coefficient['acceleration'])
            p=np.asarray(c['position']);v=np.asarray(c['velocity']);a=np.asarray(c['acceleration'])
            predictions={str(h):(p+max(.5,1-.9*h*h)*(v*h+.5*a*h*h)).tolist() for h in (.1,.2,.4)}
        return dict(active=active,coefficient=c,observation=observation,predictions=predictions,
            model='same body-position KF plus independent current RAW velocity; existing acceleration/damping',
            structural_geometry_retained=bool(structure_supported))
