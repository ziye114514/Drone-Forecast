"""Short causal support union for weak current per-pixel period observations.

No rotor identity. Native pixel IDs are deduplicated across time. Frequency is
read from CURRENT pixels; older points only confirm it. Original timestamps/
polarities are never warped into fabricated pixel event trains.
"""
from collections import deque
import numpy as np


class TemporalPeriodSupport:
    window_s=.1
    def __init__(self):self.rows=deque();self.last_t=None
    def step(self,t,q,bbox,velocity,current):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing time required')
        self.last_t=t
        while self.rows and t-self.rows[0][0]>self.window_s+1e-9:self.rows.popleft()
        if bbox is None:
            self.rows.clear();return dict(current),dict(applied=False,reason='no_current_body')
        q=q[(q['frequency']>=60)&(q['frequency']<=1200)]
        if len(q):self.rows.append((t,q.copy()))
        if current.get('qualified'):return dict(current),dict(applied=False,reason='current_strong_period_preserved')
        diag=dict(applied=False,reason='insufficient_distinct_current_and_history_support',
            window_s=self.window_s,hidden_rotor_frequencies_measured=0,
            repeated_pixel_votes_counted_once=True,pixels_assumed_independent=False)
        if len(q)<4 or len(self.rows)<2:return dict(current),diag
        b=np.asarray(bbox,float).copy();b[:2]=np.maximum(b[:2],0);b[2:]=np.minimum(b[2:],[1280,720])
        size=b[2:]-b[:2]
        if min(size)<12:return dict(current),diag
        history=np.concatenate([a for _,a in self.rows]);source_frame=np.concatenate([np.full(len(a),s) for s,a in self.rows])
        pixel=history['y'].astype(np.int64)*1280+history['x']
        _,idx=np.unique(pixel[::-1],return_index=True);idx=len(history)-1-idx
        history=history[idx];source_frame=source_frame[idx];pixel=pixel[idx]
        # Advect existing frequency POINTS solely to assign ROI/ring membership.
        # Period estimation itself always saw the original camera pixels.
        age=t-history['t'].astype(float)/1e6
        xy=np.column_stack((history['x'],history['y'])).astype(float)+age[:,None]*np.asarray(velocity)
        inside=np.all((xy>=b[:2])&(xy<b[2:]),axis=1)
        pad=np.maximum(size*.25,12);outer=np.r_[np.maximum(b[:2]-pad,0),np.minimum(b[2:]+pad,[1280,720])]
        ring=np.all((xy>=outer[:2])&(xy<outer[2:]),axis=1)&~inside
        frequency=history['frequency'].astype(float)
        bins=np.floor(np.log(frequency/60.)/np.log(1.02)).astype(int)
        support=np.bincount(bins[inside],minlength=153)
        k=int(np.argmax(np.convolve(support,np.ones(3,dtype=int),'same')))
        selected=inside&(abs(bins-k)<=1)
        current_ids=q['y'].astype(np.int64)*1280+q['x']
        current_selected=selected&np.isin(pixel,current_ids)&(source_frame==t)
        count=int(selected.sum());now=int(current_selected.sum())
        frames=len(np.unique(source_frame[selected]))
        diag.update(distinct_mode_pixels=count,current_mode_pixels=now,source_windows=frames,
            total_distinct_pixels=len(history))
        if count<16 or now<4 or frames<2:return dict(current),diag
        mode=float(np.median(frequency[current_selected]))
        near=np.abs(np.log(frequency/mode))<=np.log(1.02)*1.5
        matching=int(np.sum(inside&near));background=int(np.sum(ring&near))
        area=float(np.prod(size));ring_area=float(np.prod(outer[2:]-outer[:2])-area)
        dominance=count/max(int(inside.sum()),1)
        density=matching/area;bg_density=background/max(ring_area,1.)
        if dominance<.35 or bg_density>=.5*density:return dict(current),dict(diag,reason='weak_or_common_temporal_mode')
        result=dict(current,qualified=True,reason='current_period_with_distinct_temporal_support',
            frequency_hz=mode,mode_pixels=count,mode_fraction=dominance,
            matching_roi_pixels=matching,matching_ring_pixels=background,
            roi_density=density,ring_density=bg_density,temporal_support=True,
            current_mode_pixels=now,source_windows=frames,maximum_support_age_s=self.window_s,
            frequency_from_current_native_pixels=True,harmonic_ambiguous=True)
        return result,dict(diag,applied=True,reason='current_frequency_confirmed_by_distinct_recent_pixels')
