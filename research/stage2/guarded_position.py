"""Bounded position continuity after rotor confirmation; no semantic detector.

Default keeps every original measured rotor position. Activity cannot acquire
an unknown target: it may only bridge a recent, spatially consistent rotor loss.
"""
import copy,numpy as np

class GuardedPosition:
    def __init__(self):
        self.confirmed=None;self.last_t=None;self.last_center=None;self.velocity=np.zeros(2);self.extent=60.;self.offset=None

    def step(self,rotor,activity):
        t=rotor['t']
        if rotor['detected']:
            center=np.asarray(rotor['center']);box=np.asarray(rotor['bbox']);extent=float(max(box[2:]-box[:2]))
            if self.last_t is not None and 0<t-self.last_t<.3:
                self.velocity=.7*self.velocity+.3*np.clip((center-self.last_center)/(t-self.last_t),-2500,2500)
            else:self.velocity[:]=0
            if rotor.get('observed_rotor_count',len(rotor.get('rotors',[])))==4 and rotor.get('geometry_quality',0)>.45:
                self.confirmed=t
            if activity['detected'] and np.linalg.norm(np.asarray(activity['center'])-center)<max(30.,extent*.5):
                difference=center-np.asarray(activity['center'])
                self.offset=difference if self.offset is None else .8*self.offset+.2*difference
            self.last_t=t;self.last_center=center;self.extent=extent
            result=copy.deepcopy(rotor);result['position_source']='rotor_observation';return result
        allowed=(self.confirmed is not None and t-self.confirmed<=.5 and self.last_t is not None and t-self.last_t<=.3)
        if allowed and activity['detected'] and self.offset is not None:
            center=np.asarray(activity['center'])+self.offset
            expected=self.last_center+self.velocity*(t-self.last_t)
            if np.linalg.norm(center-expected)<=max(25.,self.extent*.5):
                result=copy.deepcopy(activity);result['center']=center.tolist();result['geometry_quality']=0.
                result['position_source']='activity_after_rotor_confirmation';result['identity_confirmation_age_s']=t-self.confirmed
                return result
        result=dict(t=t,detected=False,confidence=0.,rotors=[],n_events=rotor['n_events'],geometry_quality=0.,position_source='no_confirmed_target')
        return result
