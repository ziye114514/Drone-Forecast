"""Local visual rotor tracks with bounded coasting and confirmed reacquisition.

Reuses RotorTrack and SciPy assignment. Missing predictions are diagnostic only:
they are never inserted into measured geometry or used as fresh rate evidence.
"""
from collections import deque
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from temporal148 import TemporalBank, tensor, from_tensor, RotorTrack

PARAMS = dict(expiry_s=.40, relock_gap_s=.04, relock_observations=2,
    rate_reset_gap_s=.08, maximum_gate_major=.60, ambiguity_margin=.20,
    relative_velocity_gain=.35, relative_speed_major_per_s=4.,
    prediction_is_measurement=False, physical_motor_identity_resolved=False)

class InertialBank(TemporalBank):
    def __init__(self):
        super().__init__()
        self.coasting=[]

    @staticmethod
    def family(r):
        return 'line' if r.get('mode')=='line' or r['ratio']<.12 else 'disk'

    def expected(self,q,t,body):
        tr=q['track']; age=max(0.,t-tr.seen)
        if body.get('detected') and q.get('body_at_seen') is not None:
            return np.array([tr.rotor['cx'],tr.rotor['cy']])+np.asarray(body['center'])-q['body_at_seen']+q['relative_velocity']*age
        return tr.position(t)

    def step(self,t,body,rotors):
        if self.last_t is not None and abs(t-self.last_t)<1e-9:
            self.tracks,self.next_id,self.body,self.body_t=copy.deepcopy(self.frame_snapshot)
        else:
            self.frame_snapshot=copy.deepcopy((self.tracks,self.next_id,self.body,self.body_t))
        expired=sum(t-q['track'].seen>PARAMS['expiry_s']+1e-9 for q in self.tracks)
        self.tracks=[q for q in self.tracks if t-q['track'].seen<=PARAMS['expiry_s']+1e-9]
        current=copy.deepcopy(rotors); costs=np.full((len(self.tracks),len(current)),1e6)
        predictions=[self.expected(q,t,body) for q in self.tracks]
        for i,q in enumerate(self.tracks):
            r=q['track'].rotor; age=t-q['track'].seen
            sigma=max(2.,float(r.get('center_sigma_px',2.)))+age*max(8.,r['major']*.30)
            for j,m in enumerate(current):
                if q['family']!=self.family(m): continue
                scale=max(r['major'],m['major'],10.)
                shape=abs(np.log(max(m['major'],1.)/max(r['major'],1.)))
                distance=np.linalg.norm(predictions[i]-[m['cx'],m['cy']])
                gate=min(PARAMS['maximum_gate_major']*scale,max(8.,.45*scale,3*sigma))
                if shape>.45 or distance>gate: continue
                costs[i,j]=distance/max(6.,scale*.25)+shape
        assigned={}; ambiguous=set()
        # Active tracks have first claim. Dormant hypotheses must not steal a
        # still-visible rotor or make every current association ambiguous.
        active=[i for i,q in enumerate(self.tracks) if t-q['track'].seen<=.040001]
        dormant=[i for i in range(len(self.tracks)) if i not in active]
        for eligible in (active,dormant):
            columns=[j for j in range(len(current)) if j not in assigned]
            if not eligible or not columns: continue
            subset=costs[np.ix_(eligible,columns)]
            rows,cols=linear_sum_assignment(subset)
            for ii,jj in zip(rows,cols):
                i,j=eligible[ii],columns[jj]; value=subset[ii,jj]
                if value>=1e5: continue
                margin=min(np.min(np.delete(subset[ii],jj),initial=1e6),
                           np.min(np.delete(subset[:,jj],ii),initial=1e6))-value
                if margin<PARAMS['ambiguity_margin']:
                    ambiguous.add(j); continue
                assigned[j]=i
                ambiguous.discard(j)
        out=[]; relocked=0; pending=0; smoothed=0; reset_rates=0
        observed_ids=set()
        for j,r in enumerate(current):
            raw=copy.deepcopy(r); p=np.array([r['cx'],r['cy']],float); T=tensor(r)
            fresh=j not in assigned; is_relock=False; just_confirmed=False
            if fresh:
                track=RotorTrack(self.next_id,raw,t); self.next_id+=1
                q=dict(track=track,family=self.family(r),filtered_center=p.copy(),T=T,
                    filtered_t=t,history=deque(),pending_hz=None,pending_count=0,
                    body_at_seen=np.asarray(body['center'],float) if body.get('detected') else None,
                    relative_velocity=np.zeros(2),relock_remaining=0)
                self.tracks.append(q)
            else:
                q=self.tracks[assigned[j]]; track=q['track']; age=t-track.seen
                expected=predictions[assigned[j]]; innovation=p-expected
                is_relock=age>PARAMS['relock_gap_s']+1e-8
                if is_relock:
                    q['relock_remaining']=PARAMS['relock_observations']
                if age>PARAMS['rate_reset_gap_s']+1e-8:
                    q['history'].clear(); track.frequency=None
                    q['pending_hz']=None; q['pending_count']=0; reset_rates+=1
                if age>1e-6 and body.get('detected') and q['body_at_seen'] is not None:
                    motion=(p-[track.rotor['cx'],track.rotor['cy']]-
                            (np.asarray(body['center'])-q['body_at_seen']))/age
                    cap=max(40.,r['major']*PARAMS['relative_speed_major_per_s'])
                    motion*=min(1.,cap/max(np.linalg.norm(motion),1e-9))
                    q['relative_velocity']=(1-PARAMS['relative_velocity_gain'])*q['relative_velocity']+PARAMS['relative_velocity_gain']*motion
                if np.linalg.norm(innovation)<max(2.,r['major']*.18) and not is_relock:
                    adjustment=-.30*innovation
                    adjustment*=min(1.,.06*r['major']/max(np.linalg.norm(adjustment),1e-9))
                    r['cx'],r['cy']=map(float,p+adjustment); smoothed+=1
                if np.max(abs(np.log(np.linalg.eigvalsh(T)/np.linalg.eigvalsh(q['T']))))<.20:
                    T=.75*T+.25*q['T']
                    r['major'],r['minor'],r['angle']=from_tensor(T); r['ratio']=r['minor']/r['major']
                track.update(raw,t)
                if q['relock_remaining']>0:
                    q['relock_remaining']-=1
                    if q['relock_remaining']==0: relocked+=1; just_confirmed=True
            tentative=q['relock_remaining']>0
            pending+=int(tentative)
            q['filtered_center']=np.array([r['cx'],r['cy']]); q['filtered_t']=t; q['T']=T
            q['body_at_seen']=np.asarray(body['center'],float) if body.get('detected') else None
            observed_ids.add(track.id)
            r.update(visual_track_id=track.id,temporal_observations=track.observations,
                raw_center148=p.tolist(),current_observation=True,
                association_ambiguous=j in ambiguous or tentative,
                association_state='relock_pending' if tentative else ('new' if fresh else 'measured'),
                reacquisition_started=is_relock,reacquisition_confirmed=just_confirmed,
                position_filter_applied=not np.allclose(p,[r['cx'],r['cy']]),
                physical_motor_identity_resolved=False)
            out.append(r)
        self.coasting=[]
        for q in self.tracks:
            tr=q['track']
            if tr.id in observed_ids: continue
            age=max(0.,t-tr.seen); p=self.expected(q,t,body)
            self.coasting.append(dict(visual_track_id=tr.id,center=p.tolist(),
                major=tr.rotor['major'],minor=tr.rotor['minor'],angle=tr.rotor['angle'],
                age_s=age,sigma_px=max(2.,tr.rotor['major']*.03)+age*tr.rotor['major']*.30,
                confidence=max(0.,1-age/PARAMS['expiry_s']),current_observation=False,
                frequency_hz=None,state='coasting'))
        if body.get('detected'): self.body=np.asarray(body['center']); self.body_t=t
        self.last_t=t
        self.diagnostic=dict(current=len(out),active_memory_tracks=len(self.tracks),
            new_tracks=sum(j not in assigned for j in range(len(current))),
            matched=len(assigned),ambiguous=len(ambiguous),centers_smoothed=smoothed,
            missing_tracks_emitted=0,coasting_tracks=copy.deepcopy(self.coasting),
            relock_pending=pending,relocks_confirmed=relocked,expired_tracks=expired,
            frequency_histories_reset=reset_rates,
            prediction_is_measurement=False)
        return out
