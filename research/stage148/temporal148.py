"""Current-evidence rotor association and short-term exclusive-pixel frequency.

Reuses RotorTrack's kinematics/frequency filter, scipy assignment, 147's real
arc frontend and geometry. Stable visual IDs are not calibrated motor labels.
No missing track is ever inserted into the list of current observations.
"""
from collections import deque,Counter
from pathlib import Path
import copy,sys
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage147'),str(ROOT/'research/stage146')]
import branches146 as old
from residual148 import ResidualExpert
from guided148 import GuidedFrontend
from curvature147 import BranchObserver as Observer147,CurvatureGeometry
from event_drone_forecast.rotor_sensing import RotorTrack

PARAMS=dict(track_expiry_s=.12,assignment_gate_major=.45,assignment_ambiguity_margin=.20,
    measured_center_gain=.70,maximum_center_correction_major=.06,shape_gain=.75,
    maximum_shape_innovation_log=.20,minimum_rate_current_pixels=4,
    rate_history_s=.08,minimum_rate_frames=2,rate_log_band=.08,
    harmonic_confirmation_frames=3,minimum_current_mode_fraction=.35,
    missing_tracks='Never emitted as current observations; no hidden frequency counted as measured.',
    overlap='Ambiguous pixels excluded; only exclusive support is aggregated across stable visual IDs.',
    default_promoted=False)

def tensor(r):
    a=np.deg2rad(r['angle']);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    return R@np.diag([(r['major']/2)**2,(r['minor']/2)**2])@R.T

def from_tensor(T):
    values,vectors=np.linalg.eigh(T);a=vectors[:,-1]
    return float(2*np.sqrt(values[-1])),float(2*np.sqrt(values[0])),float(np.rad2deg(np.arctan2(a[1],a[0]))%180)

class TemporalBank:
    def __init__(self):
        self.tracks=[];self.next_id=0;self.body=None;self.body_t=None;self.last_t=None;self.diagnostic={};self.frame_snapshot=None

    def step(self,t,body,rotors):
        if self.last_t is not None and abs(t-self.last_t)<1e-9:
            self.tracks,self.next_id,self.body,self.body_t=copy.deepcopy(self.frame_snapshot)
        else:self.frame_snapshot=copy.deepcopy((self.tracks,self.next_id,self.body,self.body_t))
        self.tracks=[q for q in self.tracks if t-q['track'].seen<=PARAMS['track_expiry_s']+1e-9]
        current=copy.deepcopy(rotors);family=lambda r:'line' if r.get('mode')=='line' or r['ratio']<.12 else 'disk'
        delta=np.zeros(2)
        has_body_delta=body.get('detected') and self.body is not None and t-self.body_t<=.16
        if has_body_delta:
            delta=np.asarray(body['center'])-self.body
        assigned={};ambiguous=[];costs=np.full((len(self.tracks),len(current)),1e6)
        for i,q in enumerate(self.tracks):
            track=q['track'];r=track.rotor
            expected=np.array([r['cx'],r['cy']])+delta if has_body_delta else track.position(t)
            for j,m in enumerate(current):
                if q['family']!=family(m):continue
                distance=np.linalg.norm(expected-[m['cx'],m['cy']]);scale=max(r['major'],m['major'],10.)
                if distance>max(8.,PARAMS['assignment_gate_major']*scale):continue
                shape=abs(np.log(max(m['major'],1)/max(r['major'],1)))
                if shape>.5:continue
                costs[i,j]=distance/max(6.,scale*.25)+shape
        if costs.size:
            rows,cols=linear_sum_assignment(costs)
            for i,j in zip(rows,cols):
                score=costs[i,j]
                if score>=1e5:continue
                row_other=np.delete(costs[i],j);col_other=np.delete(costs[:,j],i)
                margin=min(np.min(row_other,initial=1e6),np.min(col_other,initial=1e6))-score
                if margin<PARAMS['assignment_ambiguity_margin']:
                    ambiguous.append(j);continue
                assigned[j]=i
        out=[];smooth_count=0
        for j,r in enumerate(current):
            raw_rotor=copy.deepcopy(r);raw_center=np.array([r['cx'],r['cy']]);raw_T=tensor(r);fresh=j not in assigned
            if fresh:
                track=RotorTrack(self.next_id,r,t);self.next_id+=1
                q=dict(track=track,family=family(r),filtered_center=raw_center.copy(),T=raw_T,
                       filtered_t=t,history=deque(),pending_hz=None,pending_count=0)
                self.tracks.append(q)
            else:
                q=self.tracks[assigned[j]];track=q['track']
                expected=q['filtered_center']+delta if has_body_delta else q['filtered_center']+track.velocity*(t-q['filtered_t'])
                innovation=raw_center-expected
                # Bound corrections; a fast geometric change keeps the current
                # measurement instead of being smoothed into a delayed pose.
                if np.linalg.norm(innovation)<max(2.,r['major']*.18) and j not in ambiguous:
                    adjustment=-(1-PARAMS['measured_center_gain'])*innovation
                    cap=PARAMS['maximum_center_correction_major']*r['major']
                    adjustment*=min(1.,cap/max(np.linalg.norm(adjustment),1e-9))
                    center=raw_center+adjustment;r['cx'],r['cy']=map(float,center);smooth_count+=1
                if np.max(abs(np.log(np.linalg.eigvalsh(raw_T)/np.linalg.eigvalsh(q['T']))))<PARAMS['maximum_shape_innovation_log']:
                    smooth_T=PARAMS['shape_gain']*raw_T+(1-PARAMS['shape_gain'])*q['T']
                    r['major'],r['minor'],r['angle']=from_tensor(smooth_T);raw_T=smooth_T
                    r['ratio']=r['minor']/r['major']
                track.update(raw_rotor,t)
            q['filtered_center']=np.array([r['cx'],r['cy']]);q['filtered_t']=t;q['T']=raw_T
            r.update(visual_track_id=track.id,temporal_observations=track.observations,
                     raw_center148=raw_center.tolist(),current_observation=True,association_ambiguous=j in ambiguous,
                     position_filter_applied=not fresh and not np.allclose(raw_center,[r['cx'],r['cy']]),
                     physical_motor_identity_resolved=False)
            out.append(r)
        if body.get('detected'):self.body=np.asarray(body['center']);self.body_t=t
        self.last_t=t
        self.diagnostic=dict(current=len(out),active_memory_tracks=len(self.tracks),new_tracks=sum(j not in assigned for j in range(len(current))),
                             matched=len(assigned),ambiguous=len(ambiguous),centers_smoothed=smooth_count,
                             missing_tracks_emitted=0,body_translation_used=delta.tolist())
        return out

    def get(self,identifier):
        return next((q for q in self.tracks if q['track'].id==identifier),None)


class GeometryAdapter:
    def __init__(self,bank,w,h):self.bank=bank;self.base=CurvatureGeometry(w,h);self.w=w;self.h=h
    def step(self,t,body,rotors):
        raw=copy.deepcopy(rotors);rotors[:]=self.bank.step(t,body,rotors)
        # Filtering cannot turn an out-of-view inferred ellipse into complete evidence.
        for r in rotors:
            T=tensor(r);dx,dy=np.sqrt(np.diag(T))
            if r['cx']-dx<0 or r['cy']-dy<0 or r['cx']+dx>self.w-1 or r['cy']+dy>self.h-1:
                r.update(fov_clipped=True,mode='arc',inferred_shape=True)
        result=self.base.step(t,body,rotors)
        result['temporal148']=dict(self.bank.diagnostic,raw_rotors=raw)
        return result


class TemporalRates:
    def __init__(self,bank):self.bank=bank

    def __call__(self,q,rotors,t,velocity,w):
        original=old_rates(q,rotors,t,velocity,w)
        if len(q):
            pixel=q['y'].astype(np.int64)*w+q['x'];_,ii=np.unique(pixel[::-1],return_index=True)
            q=q[len(q)-1-ii];q=q[(q['t']>=round(t*1e6)-20000)&(q['frequency']>=60)&(q['frequency']<=1200)]
        xy=np.c_[q['x'],q['y']].astype(float)
        if len(q):xy+=(t-q['t']/1e6)[:,None]*np.asarray(velocity)
        masks=[]
        for r in rotors:
            d=xy-[r['cx'],r['cy']];angle=np.deg2rad(r['angle']);c,s=np.cos(angle),np.sin(angle)
            u=d[:,0]*c+d[:,1]*s;v=-d[:,0]*s+d[:,1]*c
            masks.append((u/max(.55*r['major'],1))**2+(v/max(.65*r['minor'],2))**2<=1)
        ambiguous=np.sum(masks,axis=0)>1 if masks else np.zeros(len(q),bool)
        result=[]
        for index,r in enumerate(rotors):
            record=dict(original[index],instantaneous146=original[index],temporally_aggregated=False,
                        harmonic_identity_unverified=True,mechanical_rpm_measured=False)
            item=self.bank.get(r.get('visual_track_id'))
            if item is None or r.get('association_ambiguous'):
                record.update(valid=False,frequency_hz=None,reason='no_unambiguous_visual_identity');result.append(record);continue
            f=q['frequency'][masks[index]&~ambiguous].astype(float)
            while item['history'] and t-item['history'][0][0]>PARAMS['rate_history_s']+1e-8:item['history'].popleft()
            record.update(visual_track_id=r['visual_track_id'],current_exclusive_pixels=len(f),overlap_pixels_excluded=int(np.sum(masks[index]&ambiguous)))
            if len(f)<PARAMS['minimum_rate_current_pixels']:
                record.update(valid=False,frequency_hz=None,reason='no_current_exclusive_support');result.append(record);continue
            bins=np.floor(np.log(f/60)/np.log(1.04)).astype(int);hist=np.bincount(bins,minlength=78)
            k=int(np.argmax(np.convolve(hist,np.ones(3,dtype=int),'same')));keep=abs(bins-k)<=1
            hz=float(np.median(f[keep]));fraction=float(keep.mean())
            item['history'].append((t,hz,fraction,int(keep.sum())))
            coherent=[p for p in item['history'] if abs(np.log(p[1]/hz))<=PARAMS['rate_log_band'] and p[2]>=PARAMS['minimum_current_mode_fraction']]
            if fraction<PARAMS['minimum_current_mode_fraction'] or int(keep.sum())<4 or len(coherent)<PARAMS['minimum_rate_frames']:
                record.update(valid=False,frequency_hz=None,reason='waiting_for_consistent_current_periods');result.append(record);continue
            # Competing half/double modes in current exclusive support are not
            # resolved by choosing whichever is closest to the previous rate.
            competitor=max(np.sum(abs(np.log(f/(hz*factor)))<.08) for factor in (.5,2.))
            if competitor>=.6*keep.sum():
                record.update(valid=False,frequency_hz=None,reason='multiple_harmonic_modes');result.append(record);continue
            track=item['track'];harmonic_jump=False
            if track.frequency is not None:
                ratio=hz/track.frequency;harmonic_jump=abs(np.log(ratio/2))<.12 or abs(np.log(ratio/.5))<.12
                if harmonic_jump:
                    count=0
                    for previous in reversed(item['history']):
                        if abs(np.log(previous[1]/hz))>=.08 or previous[2]<PARAMS['minimum_current_mode_fraction']:break
                        count+=1
                    item['pending_hz']=hz;item['pending_count']=count
                    if item['pending_count']<PARAMS['harmonic_confirmation_frames']:
                        record.update(valid=False,frequency_hz=None,reason='harmonic_transition_pending');result.append(record);continue
                    track.frequency=None
                else:item['pending_hz']=None;item['pending_count']=0
            # Use current frequency, not a long-window average that hides a turn.
            spread=max(2.,float(1.4826*np.median(abs(f[keep]-hz))))
            track.update_frequency(dict(valid=True,blade_pass_hz=hz,quality=fraction,
                frequency_resolution_hz=spread,phase_coherence=0.),t)
            if track.frequency_t!=t:
                record.update(valid=False,frequency_hz=None,reason='frequency_filter_rejected');result.append(record);continue
            record.update(valid=True,frequency_hz=float(track.frequency),conditional_rpm_if_two_blades=float(track.frequency*30),
                          reason='exclusive_current_pixels_and_short_history',mode_fraction=fraction,
                          temporally_aggregated=True,coherent_frames=len(coherent),harmonic_transition_confirmed=harmonic_jump,
                          past_only_frequency_held=False)
            result.append(record)
        return result

# Capture the old function once; synchronous wrappers below restore it on exit.
old_rates=old.rotor_rates

class BranchObserver(Observer147):
    def __init__(self,w,h,threshold=4.,adaptive=False):
        super().__init__(w,h,threshold,adaptive);self.front=GuidedFrontend(w,h,threshold,adaptive);self.reset_temporal()
    def reset_temporal(self):
        self.bank=TemporalBank();self.geometry=GeometryAdapter(self.bank,self.w,self.h)
        self.temporal_rates=TemporalRates(self.bank);self.residual_expert=ResidualExpert()
    def step(self,events,t,prepared,body,coefficient,baseline_predictions):
        if body.get('detected') and self.last_body is not None and t-self.last_body>.5:
            self.reset_temporal();self.proxy=old.ThrustProxy();self.expert=old.CausalPhysicsExpert();self.last_body=t
        original=old.rotor_rates;old.rotor_rates=self.temporal_rates
        try:result=super().step(events,t,prepared,body,coefficient,baseline_predictions)
        finally:old.rotor_rates=original
        result['temporal148']=self.bank.diagnostic
        self.front.guide.update(t,body,result['geometry'])
        absolute=result['expert']
        result['absolute_expert148']=absolute
        result['expert']=self.residual_expert.step(t,body,result['geometry'],result['thrust_proxy'],baseline_predictions)
        result['extra_predictions']=dict(rotor_bias=result['expert']['bias_only_predictions'],
                                        rotor_bias_fused=result['expert']['bias_only_fused'],
                                        rotor_temporal_absolute_fused=absolute['fused'])
        if absolute['physical']:result['extra_predictions']['rotor_temporal_absolute']=absolute['physical']
        return result
