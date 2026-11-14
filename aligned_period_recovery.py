"""Bounded, optional recovery of weak periods in body-translation coordinates.

Reuse stage79's installed SDK strict7 observer and current20ms/ring rules.
Reinitialize the observer on <=160ms of measured ROI history: pixel identities
change under stabilization, so native observer state must never be reused.
The unmodified native observer remains separate. Current geometric pixels are
not warped; no hidden motors or metric RPM are output.
"""
from collections import deque
import sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'research/stage79'))
from native79 import engine,checked_process,summarize
from live_frequency_pixels import latest_pixels

class AlignedPeriodRecovery:
    def __init__(self,width=1280,height=720,configuration='strict7'):
        if configuration not in ('strict7','fast3'):raise ValueError('Unknown native period configuration')
        self.configuration=configuration
        self.width=width;self.height=height;self.rows=deque();self.integral=np.zeros(2);self.last_t=None;self.tick=0
    def step(self,t,events,body,tracking,current,axis_qualified=True,translation_qualified=True):
        start=time.perf_counter();self.tick+=1;old=self.last_t;self.last_t=t
        if old is not None and t<=old:raise ValueError('Increasing source time required')
        b=np.asarray(body.get('bbox',[]),float)
        dt=0. if old is None else t-old
        tracked=bool(translation_qualified and 0.<dt<=.08+1e-8 and tracking.get('reason')=='current_image_translation' and tracking.get('points',0)>=8)
        valid=bool(body.get('detected') and not body.get('low_confidence') and b.shape==(4,) and
            np.isfinite(b).all() and min(b[2:]-b[:2])>=12 and np.all(b[:2]>0) and np.all(b[2:]<[self.width-1,self.height-1]))
        info=dict(applied=False,reason='no_current_qualified_body_and_translation',maximum_history_s=.16,
            native_period_channel_unchanged=True,detector_geometry_pixels_unchanged=True,
            timestamps_polarities_unchanged=True,pixel_coordinates_stabilized=True,mechanical_RPM_measured=False)
        if not tracked or not valid:
            self.rows.clear();self.integral[:]=0.
            return dict(current),dict(info,CPU_ms=(time.perf_counter()-start)*1000)
        similarity=tracking.get('similarity_motion') or {}
        delta=np.asarray(similarity['displacement'] if similarity.get('valid') else tracking['delta'],float)
        self.integral+=delta
        pad=np.maximum((b[2:]-b[:2])*.25,12.)
        outer=np.r_[np.maximum(b[:2]-pad,0),np.minimum(b[2:]+pad,[self.width,self.height])]
        # Include both ends of the measured within-frame translation, so the
        # current surrounding ring is not silently cropped from older input.
        stored=np.r_[np.maximum(outer[:2]-abs(delta),0),np.minimum(outer[2:]+abs(delta),[self.width,self.height])]
        inside=(events['x']>=stored[0])&(events['x']<stored[2])&(events['y']>=stored[1])&(events['y']<stored[3])
        self.rows.append((t,old,self.integral.copy(),delta.copy(),events[inside].copy(),stored.copy()))
        while self.rows and t-self.rows[0][0]>.16+1e-8:self.rows.popleft()
        if current.get('qualified'):return dict(current),dict(info,reason='strong_native_period_preserved',CPU_ms=(time.perf_counter()-start)*1000)
        if not axis_qualified or self.tick%3!=0 or len(self.rows)<4 or t-self.rows[0][1]<.12-1e-8:
            return dict(current),dict(info,reason='aligned_recovery_schedule_or_warmup',CPU_ms=(time.perf_counter()-start)*1000)
        lo=np.floor(outer[:2]).astype(int);hi=np.ceil(outer[2:]).astype(int);size=hi-lo;packets=[];coverage=[]
        for end,begin,position,displacement,e,rect in self.rows:
            if not len(e):continue
            age=(end-e['t'].astype(float)/1e6)/(end-begin)
            xy=np.column_stack((e['x'],e['y'])).astype(float)+self.integral-position+age[:,None]*displacement
            xy=np.rint(xy-lo).astype(int)
            keep=np.all((xy>=0)&(xy<size),axis=1)&(e['t']>=round((t-.16)*1e6))
            q=e[keep].copy();q['x']=xy[keep,0];q['y']=xy[keep,1];packets.append(q)
            covered=rect+np.tile(self.integral-position,2)
            covered[:2]+=np.maximum(displacement,0);covered[2:]+=np.minimum(displacement,0)
            coverage.append(covered)
        if not packets:return dict(current),dict(info,reason='no_stabilized_events',CPU_ms=(time.perf_counter()-start)*1000)
        cover=np.asarray(coverage);common=np.r_[np.maximum(outer[:2],np.max(cover[:,:2],axis=0)),
            np.minimum(outer[2:],np.min(cover[:,2:],axis=0))]
        body_intersection=np.prod(np.maximum(0.,np.minimum(common[2:],b[2:])-np.maximum(common[:2],b[:2])))
        ring=np.prod(outer[2:]-outer[:2])-np.prod(b[2:]-b[:2])
        common_ring=max(0.,np.prod(np.maximum(0.,common[2:]-common[:2]))-body_intersection)
        if body_intersection<.9*np.prod(b[2:]-b[:2]) or common_ring<.75*ring:
            return dict(current),dict(info,reason='insufficient_shared_body_and_ring_coverage',CPU_ms=(time.perf_counter()-start)*1000)
        e=np.concatenate(packets);algorithm=engine(int(size[0]),int(size[1]),self.configuration);buffer=algorithm.get_empty_output_buffer()
        q=checked_process(algorithm,e,buffer,int(size[0]),int(size[1]));q=latest_pixels(q,int(size[0]))
        pooled=summarize(q,b-np.tile(lo,2),t,int(size[0]),int(size[1]),unique_pixels=True)
        info.update(reason='stabilized_period_still_weak',aligned_period=pooled,
            current_ROI_history_events=len(e),native_configuration=self.configuration,
            native_filter_length=7 if self.configuration=='strict7' else 3,observer_reset_each_attempt=True,
            period_attempt_interval_ticks=3,history_rows=len(self.rows),history_span_s=t-self.rows[0][1])
        if not pooled.get('qualified'):return dict(current),dict(info,CPU_ms=(time.perf_counter()-start)*1000)
        pooled.update(bbox=b.tolist(),reason='qualified_translation_aligned_'+self.configuration,
            source='measured_translation_stabilized_pixel_period',physical_rotor_count_observed=False,
            original_native_period=current,reference_pixels_not_independent_sensor_votes=True)
        return pooled,dict(info,applied=True,reason='qualified_translation_aligned_period',CPU_ms=(time.perf_counter()-start)*1000)
