"""Persistent SDK frequency observer in a FIXED body-translation frame.

Unlike re-anchoring and replaying a window at every cutoff, reference pixels
stay in one frame for the entire qualified track. Each original event enters
the period observer once. Translation comes from the existing current KLT;
timestamps/polarities, detector input and geometry remain unchanged.
"""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'research/stage79'))
from native79 import engine,checked_process,summarize
from live_frequency_pixels import latest_pixels

class BodyPeriodStream:
    def __init__(self,width=1280,height=720,configuration='fast3'):
        if configuration not in ('strict7','fast3'):raise ValueError('Unknown observer configuration')
        self.width=width;self.height=height;self.configuration=configuration
        self.algorithm=self.output=None;self.offset=None;self.integral=np.zeros(2);self.last_t=None;self.epoch=0
    def reset(self):
        self.algorithm=self.output=None;self.offset=None;self.integral[:]=0.;self.epoch+=1
    def step(self,t,events,body,tracking,current,translation_qualified=True):
        start=time.perf_counter();old=self.last_t;self.last_t=t
        if old is not None and t<=old:raise ValueError('Strictly increasing cutoffs required')
        if len(events) and (events['t'].max()>=round(t*1e6) or
            (old is not None and events['t'].min()<round(old*1e6))):
            raise ValueError('Only original events in the current source interval are accepted')
        b=np.asarray(body.get('bbox',[]),float);dt=0. if old is None else t-old
        valid=bool(translation_qualified and 0.<dt<=.08+1e-8 and body.get('detected') and
            not body.get('low_confidence') and b.shape==(4,) and np.isfinite(b).all() and
            min(b[2:]-b[:2])>=12 and np.all(b[:2]>0) and np.all(b[2:]<[self.width-1,self.height-1]) and
            tracking.get('reason')=='current_image_translation' and tracking.get('points',0)>=8)
        info=dict(applied=False,reason='no_current_qualified_translation',configuration=self.configuration,
            fixed_body_reference_frame=True,replay_past_events=False,original_event_timestamps_polarities=True,
            native_period_unchanged=True,geometric_pixels_unchanged=True,mechanical_RPM_measured=False)
        if not valid:
            self.reset();return dict(current),dict(info,CPU_ms=(time.perf_counter()-start)*1000,epoch=self.epoch)
        similarity=tracking.get('similarity_motion') or {}
        delta=np.asarray(similarity['displacement'] if similarity.get('valid') else tracking['delta'],float)
        created=self.algorithm is None
        if created:
            self.algorithm=engine(self.width,self.height,self.configuration);self.output=self.algorithm.get_empty_output_buffer()
            self.offset=np.array([self.width/2,self.height/2])-np.asarray(body['center'])+delta
        self.integral+=delta
        pad=np.maximum(.25*(b[2:]-b[:2]),12.)+abs(delta)
        outer=np.r_[np.maximum(0.,b[:2]-pad),np.minimum([self.width,self.height],b[2:]+pad)]
        keep=(events['x']>=outer[0])&(events['x']<outer[2])&(events['y']>=outer[1])&(events['y']<outer[3])
        e=events[keep]
        age=(t-e['t'].astype(float)/1e6)/dt
        xy=np.column_stack((e['x'],e['y'])).astype(float)-self.integral+self.offset+age[:,None]*delta
        xy=np.rint(xy).astype(int);inside=np.all((xy>=0)&(xy<[self.width,self.height]),axis=1)
        warped=e[inside].copy();warped['x']=xy[inside,0];warped['y']=xy[inside,1]
        box=b-np.tile(self.integral-self.offset,2)
        # A tracked body leaving the reference canvas requires reinitializing
        # the observer, never folding/wrapping coordinates into old pixels.
        if np.any(box[:2]<12) or np.any(box[2:]>[self.width-12,self.height-12]):
            self.reset();return dict(current),dict(info,reason='body_outside_reference_canvas',CPU_ms=(time.perf_counter()-start)*1000,epoch=self.epoch)
        q=checked_process(self.algorithm,warped,self.output,self.width,self.height) if len(warped) else self.output.numpy()[:0].copy()
        q=latest_pixels(q,self.width);pooled=summarize(q,box,t,self.width,self.height,unique_pixels=True)
        info.update(reason='streaming_body_period_still_weak',aligned_period=pooled,
            current_ROI_events=len(e),reference_events=len(warped),epoch=self.epoch,
            observer_initialized_now=created,observer_reset_each_tick=False,
            reference_pixel_votes_not_independent_sensor_votes=True)
        if current.get('qualified'):return dict(current),dict(info,reason='strong_native_period_preserved',CPU_ms=(time.perf_counter()-start)*1000)
        if not pooled.get('qualified'):return dict(current),dict(info,CPU_ms=(time.perf_counter()-start)*1000)
        pooled.update(bbox=b.tolist(),reason='qualified_fixed_body_frame_'+self.configuration,
            source='persistent_body_translation_frequency',original_native_period=current,
            reference_pixels_not_independent_sensor_votes=True)
        return pooled,dict(info,applied=True,reason='qualified_persistent_body_period',CPU_ms=(time.perf_counter()-start)*1000)
