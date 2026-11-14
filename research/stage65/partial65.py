"""A body line corrects only projected heading, never unobserved tilt or RPM."""
from pathlib import Path
from collections import deque
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage62'),str(ROOT/'research/stage6')]
from independent62 import Predictor62,Managed62,Selector,MODELS,prior
from silhouette import angle_distance

METHODS=['axis62','body_static','body_aligned']
PARAMS=dict(confirmation_steps=3,maximum_step_gap_s=.04,maximum_axis_change_deg=12.,
    minimum_projected_reference=.1,axis_expiry_s=.16,
    body_quality_min=.45,
    scope='Only replace axis-only rollout direction when current rotor-axis '
          'measurement is absent and original axis62 rollout is still active. '
          'Body direction must pass unchanged6 gates and three-frame confirmation. '
          'Preserve n_z and ||n_xy|| from recent ellipse/rotor axis. No new3D '
          'normal initialization or extended freshness, no line-derived regression '
          'samples. All original learning and frequency experts remain exact62.')


class PartialDirection:
    def __init__(self):self.history=deque(maxlen=PARAMS['confirmation_steps'])
    def propose(self,t,measurement,reference):
        if not measurement.get('valid') or measurement.get('confidence',0)<PARAMS['body_quality_min']:
            self.history.clear();return None,dict(applied=False,reason='no_qualified_body_direction')
        angle=float(measurement['angle_rad'])
        if self.history:
            last_t,last_angle=self.history[-1]
            if t-last_t>PARAMS['maximum_step_gap_s']+1e-8 or np.degrees(angle_distance(angle,last_angle))>PARAMS['maximum_axis_change_deg']:
                self.history.clear()
        self.history.append((t,angle))
        if len(self.history)<PARAMS['confirmation_steps']:return None,dict(applied=False,reason='body_direction_confirming',streak=len(self.history))
        if reference is None:return None,dict(applied=False,reason='no_reference_tilt')
        n=np.asarray(reference,float);length=np.linalg.norm(n[:2])
        if length<PARAMS['minimum_projected_reference']:return None,dict(applied=False,reason='weak_projected_reference')
        z=np.mean([np.exp(2j*a) for _,a in self.history]);angle=float(np.angle(z)/2)
        normal=np.array([-np.sin(angle),np.cos(angle)])
        if normal@n[:2]<0:normal=-normal
        result=n.copy();result[:2]=length*normal
        change=float(np.degrees(np.arccos(np.clip(n[:2]@normal/length,-1,1))))
        return result,dict(applied=True,reason='projected_heading_only',streak=len(self.history),
                           correction_deg=change,reference_nz=float(n[2]),reference_projected_norm=float(length),
                           body_angle_rad=angle,reference_tilt_preserved=True)


class Predictor65(Predictor62):
    def __init__(self,method):
        assert method in METHODS;self.correction_method=method;self.current_body=None
        super().__init__('axis_learn')
    def reset_features(self):
        super().reset_features();self.partial_direction=PartialDirection()
        self.correction_diagnostic=dict(applied=False,reason='initial')
    def step(self,row):
        q,baseline=super().step(row);t=row['t']
        self.correction_diagnostic=dict(applied=False,reason='baseline')
        if self.correction_method=='axis62':return q,baseline
        assert self.current_body is not None and abs(self.current_body[0]-t)<1e-8
        n,d=self.partial_direction.propose(t,self.current_body[1],self.axis.n)
        if n is None:self.correction_diagnostic=d;return q,baseline
        md=q['feature_diagnostic']['models']['adaptive_axis_only']
        fresh_current=self.axis.last_measurement_t is not None and abs(self.axis.last_measurement_t-t)<1e-8
        if fresh_current:d.update(applied=False,reason='current_rotor_axis_preferred')
        elif not md['active']:d.update(applied=False,reason='original_axis_gate_inactive')
        elif self.axis.last_measurement_t is None or t-self.axis.last_measurement_t>PARAMS['axis_expiry_s']:
            d.update(applied=False,reason='reference_tilt_expired')
        if d['applied']:
            # Run the original model with the same learned parameters/state and
            # retained projection magnitude. Do not feed the line into RLS.
            model=self.models['adaptive_axis_only'];cv=q['coefficients']['cv']
            times,points=model.rollout(cv['position'],cv['velocity'],n,self.axis.omega,1.)
            q['coefficients']['adaptive_axis_only']=dict(copy.deepcopy(cv),
                acceleration=model.acceleration(n,1.,np.asarray(cv['velocity'])).tolist(),
                rollout_times=times.tolist(),rollout_positions=points.tolist())
            q['predictions']['adaptive_axis_only']={str(h):prior.evaluate(q['coefficients']['adaptive_axis_only'],h) for h in prior.PARAMS['horizons']}
        self.correction_diagnostic=d
        return q,baseline


class Managed65(Managed62):
    def __init__(self,method):
        super().__init__('axis_learn');self.predictor=Predictor65(method)


def controls():
    n=np.array([.4,-.6,.7]);n/=np.linalg.norm(n)
    a=PartialDirection();b=PartialDirection()
    m=dict(valid=True,confidence=.8,angle_rad=.3)
    for j in range(3):
        x,dx=a.propose(j*.02,m,n);y,dy=b.propose(j*.02,dict(m,angle_rad=.3+np.pi),n)
    assert dx['applied'] and dy['applied']
    np.testing.assert_allclose(x,y,rtol=0,atol=1e-14)
    np.testing.assert_allclose(x[2],n[2],rtol=0,atol=0)
    np.testing.assert_allclose(np.linalg.norm(x[:2]),np.linalg.norm(n[:2]),rtol=0,atol=1e-15)
    c=PartialDirection()
    for j in range(3):z,d=c.propose(j*.02,m,None)
    assert z is None and d['reason']=='no_reference_tilt'
    assert a.propose(.06,dict(valid=False),n)[0] is None
    assert a.propose(.08,m,n)[0] is None
    return dict(projected_magnitude_and_nz_preserved=True,head_tail_invariant=True,
                cannot_initialize_unknown_tilt=True,invalid_measurement_resets_confirmation=True,
                RLS_and_frequency_inputs_not_modified=True)
