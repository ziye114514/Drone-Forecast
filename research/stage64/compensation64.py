"""Reuse21 event translation and6 support-image translation with past body CV."""
from pathlib import Path
import sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage21'),str(ROOT/'research/stage6')]
from windows21 import warp
from silhouette import body_measurement,angle_distance

PARAMS=dict(step_s=.01,decay_s=.02,velocity_expiry_s=.16,body_confidence_min=.45,
    representations=['total','bipolar'],modes=['static','aligned'],
    velocity='Most recently committed20ms CV velocity with a valid forecast and '
             'current NN body confidence>=.45. Source timestamp<=native packet '
             'start, maximum age .16s. No current/end-of-packet velocity fitting.',
    geometry='Original21 integer per-event shift to packet end, then original6 '
             'warpAffine INTER_LINEAR translation of previous density by v*.01. '
             'No speed/shift clipping. Out-of-image events removed only from '
             'geometry support; original RAW events are never modified.',
    scope='Pure translational approximation. Rotor rotation, camera rotation, '
          'depth/parallax and unknown acceleration not compensated. '
          'Synthetic event-support samples are not a calibrated sensor simulator.')


class PastVelocity:
    def __init__(self):self.t=None;self.v=np.zeros(2)
    def commit(self,row):
        q=row['frontends']['delayed_feedback'];body=row['body_observation']
        if q['valid'] and body.get('detected') and body.get('confidence',0)>=PARAMS['body_confidence_min']:
            self.t=float(row['t']);self.v=np.asarray(q['coefficients']['cv']['velocity'],float)
    def before(self,begin):
        assert self.t is None or self.t<=begin+1e-8
        valid=self.t is not None and begin-self.t<=PARAMS['velocity_expiry_s']
        return (self.v.copy() if valid else np.zeros(2)),dict(source_t=self.t,valid=valid,
                  age_s=None if self.t is None else begin-self.t)


def features(events,w,h):
    pix=events['y'].astype(np.int32)*w+events['x']
    total=np.bincount(pix,minlength=w*h).reshape(h,w).astype(np.float32)
    on=np.bincount(pix[events['p']>0],minlength=w*h).reshape(h,w).astype(np.float32)
    bipolar=np.minimum(cv2.GaussianBlur(on,(3,3),.8),cv2.GaussianBlur(total-on,(3,3),.8))
    return dict(total=total,bipolar=bipolar)


class Supports:
    def __init__(self,w,h):
        self.w,self.h=w,h
        self.density={f'{m}_{r}':np.zeros((h,w),np.float32) for m in PARAMS['modes'] for r in PARAMS['representations']}
    def step(self,events,t,velocity):
        velocity=np.asarray(velocity,float);v_nonzero=bool(np.any(velocity!=0))
        raw=features(events,self.w,self.h)
        aligned_events=warp(events,t,velocity,self.w,self.h) if v_nonzero else events
        aligned=features(aligned_events,self.w,self.h) if v_nonzero else raw
        for mode,ff in [('static',raw),('aligned',aligned)]:
            for representation,value in ff.items():
                key=mode+'_'+representation;old=self.density[key]
                if mode=='aligned' and v_nonzero:
                    old=cv2.warpAffine(old,np.c_[np.eye(2),velocity*PARAMS['step_s']].astype(np.float32),
                                       (self.w,self.h),flags=cv2.INTER_LINEAR)
                old*=np.exp(-PARAMS['step_s']/PARAMS['decay_s']);old+=value
                self.density[key]=old
        return dict(raw_events=len(events),aligned_geometry_events=len(aligned_events),
                    shift_px=(velocity*PARAMS['step_s']).tolist())


def synthetic():
    dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);w,h=640,480;duration=.12
    scenarios=[dict(angle=angle,velocity=v,legs=legs,ratio=.06) for angle in [-35.,0.,30.]
               for v in [(0.,0.),(0.,1500.),(1100.,-1100.)] for legs in [False,True]]
    scenarios += [dict(angle=angle,velocity=v,legs=False,ratio=.45) for angle in [-35.,0.,30.]
                  for v in [(0.,0.),(0.,1500.)]]
    result=[]
    for index,s in enumerate(scenarios):
        mask=np.zeros((180,220),np.uint8)
        cv2.ellipse(mask,(110,80),(62,max(2,round(62*s['ratio']))),s['angle'],0,360,1,-1)
        if s['legs']:
            R=np.array([[np.cos(np.deg2rad(s['angle'])),-np.sin(np.deg2rad(s['angle']))],
                        [np.sin(np.deg2rad(s['angle'])),np.cos(np.deg2rad(s['angle']))]])
            for x in [-35,35]:
                p=R@np.array([x,7.])+[110,80];q=R@np.array([x,28.])+[110,80]
                cv2.line(mask,tuple(p.astype(int)),tuple(q.astype(int)),1,3)
        yy,xx=np.nonzero(mask);offset=np.c_[xx-110,yy-80];velocity=np.asarray(s['velocity'])
        clouds=[];rng=np.random.default_rng(6400+index)
        for j in range(12):
            e=np.zeros(12000,dtype);ts=rng.integers(j*10000,(j+1)*10000,len(e));picked=offset[rng.integers(0,len(offset),len(e))]
            points=np.rint(picked+[320,240]+(ts[:,None]/1e6-duration)*velocity).astype(int)
            keep=(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
            e['x']=np.clip(points[:,0],0,w-1);e['y']=np.clip(points[:,1],0,h-1);e['t']=ts;e['p']=rng.integers(0,2,len(e))
            e=e[keep];e.sort(order='t');clouds.append(e)
        for estimate_name,factor in [('correct',1.),('wrong_sign',-1.)]:
            memory=Supports(w,h)
            for j,e in enumerate(clouds):memory.step(e,(j+1)*.01,velocity*factor)
            if np.all(velocity==0):
                for rep in PARAMS['representations']:np.testing.assert_array_equal(memory.density['static_'+rep],memory.density['aligned_'+rep])
            for name,density in memory.density.items():
                m=body_measurement(density,[205,145,435,335])
                error=float(np.degrees(angle_distance(m['angle_rad'],np.deg2rad(s['angle'])))) if 'angle_rad' in m else None
                result.append(dict(scenario=index,**s,estimate=estimate_name,method=name,
                                   accepted=m['valid'],axis_error_deg=error,reason=m['reason'],
                                   apparent_ratio=m.get('ratio')))
    # Validate timestamp refusal, not just the inferred numerical result.
    p=PastVelocity();p.t=1.
    refused=False
    try:p.before(.99)
    except AssertionError:refused=True
    assert refused and not p.before(1.17)[1]['valid']
    return dict(scenarios=len(scenarios),rows=result,zero_velocity_support_exact=True,
                future_velocity_refused=True,stale_velocity_refused=True,
                caveat='Known raster event-support orientation, not hardware events or actual drone pose truth. '
                       'All rejections and wrong-sign controls retained; no gain assertion.')
