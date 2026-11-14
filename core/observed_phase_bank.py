"""Current nonflat ring phase -> visible input, without a full four-slot lock.

Track IDs denote signal provenance only. Every frequency is revalidated on
current raw events using the existing early/late spatial-phase test. Hidden
rotors, line projections, unvalidated hypotheses never enter the known sum.
This accuracy prototype is not a30Hz implementation of the spectral search.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from observed_ring_phase import ring_phase_prior


class ObservedPhaseBank:
    def __init__(self):
        self.tracks={};self.next_id=0;self.previous_t=None;self.velocity=None

    def step(self,t,events,rotors,motion):
        if self.previous_t is not None and t<=self.previous_t:raise ValueError('Increasing cutoffs required')
        if self.previous_t is not None and t-self.previous_t>.5:self.tracks.clear();self.velocity=None
        self.tracks={i:r for i,r in self.tracks.items() if t-r['seen']<=.12+1e-8}
        candidates=[dict(r) for r in rotors if r.get('current_observation',True) and not r.get('fov_clipped')
            and r.get('ratio',0)>.16 and r.get('center_sigma_px',99)<=3
            and r.get('axis_angle_sigma_deg',99)<=15
            and (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated'))]
        ids=list(self.tracks);mapping={};delta=np.zeros(2)
        shift=(motion or {}).get('estimate') or {};dt=(motion or {}).get('dt_s')
        if dt and not shift.get('peak_at_search_boundary') and shift.get('current_displacement_qualified',True):
            delta=np.array([shift['dx'],shift['dy']],float)
        if ids and candidates:
            cost=np.full((len(ids),len(candidates)),1e6)
            for a,i in enumerate(ids):
                old=self.tracks[i]['ellipse'];expected=np.array([old['cx'],old['cy']])+delta
                for b,r in enumerate(candidates):
                    scales=np.array([r['major']/old['major'],r['minor']/old['minor']])
                    distance=np.linalg.norm([r['cx']-expected[0],r['cy']-expected[1]])
                    gate=max(8.,.6*max(r['major'],old['major']))
                    if scales.min()>=.65 and scales.max()<=1.5 and distance<=gate:cost[a,b]=distance/gate
            aa,bb=linear_sum_assignment(cost)
            mapping={int(b):ids[a] for a,b in zip(aa,bb) if cost[a,b]<1e5}
        # Use only the previous qualified native velocity, predating the
        # complete current window, for translation compensation.
        e=events[(events['t']>=round(t*1e6)-33333)&(events['t']<round(t*1e6))]
        e=e[((e['x'].astype(np.int32)+3*e['y'])%4)==0]
        velocity=np.zeros(2)
        if self.velocity is not None and self.velocity[0]<=t-.033333+1e-7 and t-self.velocity[0]<=.12:
            velocity=self.velocity[1]
        xy=np.c_[e['x'],e['y']].astype(float)+(t-e['t']/1e6)[:,None]*velocity
        good=(xy[:,0]>=0)&(xy[:,0]<1280)&(xy[:,1]>=0)&(xy[:,1]<720)
        aligned=e[good].copy();aligned['x']=np.clip(np.rint(xy[good,0]),0,1279).astype(np.uint16)
        aligned['y']=np.clip(np.rint(xy[good,1]),0,719).astype(np.uint16)
        components={};records=[];rapid=[]
        for j,r in enumerate(candidates):
            i=mapping.get(j)
            if i is None:i=self.next_id;self.next_id+=1
            old=self.tracks.get(i)
            current=dict(r,cx=r['cx']+.010*velocity[0],cy=r['cy']+.010*velocity[1])
            phase=ring_phase_prior(aligned,current,t)
            self.tracks[i]=dict(seen=t,ellipse=r,frequency=phase.get('phase_hz'))
            records.append(dict(track_id=i,ellipse=current,phase=phase))
            if not phase['qualified']:continue
            f=phase['phase_hz'];angle=np.deg2rad(r['angle'])
            n=np.array([-np.sin(angle),np.cos(angle)])*np.sqrt(max(0.,1-r['ratio']**2))
            if n[1]>0:n=-n
            if old is not None and old.get('frequency') and abs(np.log(f/old['frequency']))>=.08:rapid.append(i)
            components[f'observed_ring_phase:{i}']=(f*f/360000.*n).tolist()
        # Multiple fits are not automatically distinct physical motors.
        qualified=[r for r in records if r['phase']['qualified']]
        ambiguous=len(qualified)>4
        for a in range(len(qualified)):
            for b in range(a):
                x,y=qualified[a]['ellipse'],qualified[b]['ellipse']
                if np.linalg.norm([x['cx']-y['cx'],x['cy']-y['cy']])<.3*max(x['major'],y['major']):ambiguous=True
        if ambiguous:components={}
        u=np.sum(list(components.values()),axis=0) if components else None
        magnitude=float(np.sqrt(sum(np.dot(v,v) for v in components.values())))
        meta=dict(source='current_unordered_visible_ring_phase_sum' if components else 'no_current_validated_ring_phase',
            observed_components=components,input_sigma_proxy=.04*magnitude,
            input_uncertainty_calibrated=False,learning_input_current=bool(components and not rapid),
            rotor_bank_used=bool(components),current_model_ids=[],hidden_model_ids=[],
            hidden_frequencies_measured=0,hidden_force_owner='joint_unknown_state_d',
            physical_motor_identity_required=False,complete_five_circle_required=False,
            metric_thrust_measured=False,mechanical_RPM_measured=False,
            qualified_current_rings=len(components),ambiguous_multiple_conics=ambiguous,
            geometry_window_midpoint_compensation_s=.010,raw_phase_pixel_subset_fraction=.25,
            records=records)
        if dt and shift.get('score',0)>0 and not shift.get('peak_at_search_boundary') and shift.get('current_displacement_qualified',True):
            self.velocity=(t,delta/dt)
        self.previous_t=t
        return u,.7 if u is not None else 0.,meta
