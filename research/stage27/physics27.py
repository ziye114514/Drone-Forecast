"""Same-state projected-thrust ablations; no labels, RAW I/O or future observations."""
from pathlib import Path
import copy,importlib.util,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('robust23_physics27',ROOT/'research/stage23/robust23.py')
robust=importlib.util.module_from_spec(spec);spec.loader.exec_module(robust)
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.paper_physics import ThrustAxisFilter,ProjectedThrustModel

PHYSICS=['nominal_axis_freq','adaptive_axis_only','adaptive_axis_freq','adaptive_fixed_axis_freq']
MODELS=robust.MODELS+PHYSICS
PARAMS=dict(feature_expiry_s=.16,ratio_clip=[.2,5.],active_quality=.15,
    shared_feature_gate=True,geometry_axis_differential_input=0.,
    horizons=robust.PARAMS['horizons'],baseline_parameters=robust.PARAMS)


def evaluate(c,h):
    if 'rollout_positions' in c:
        times=np.asarray(c['rollout_times'])
        if h<times[0]-1e-8 or h>times[-1]+1e-8:raise ValueError('Outside saved rollout')
        return [float(np.interp(h,times,np.asarray(c['rollout_positions'])[:,j])) for j in range(2)]
    return robust.base.evaluate(c,h)


class Predictor:
    def __init__(self):
        self.baseline=robust.Predictor();self.cfg=PaperConfig();self.reset_features()

    def reset_features(self):
        self.axis=ThrustAxisFilter(self.cfg)
        self.models={name:ProjectedThrustModel(self.cfg,name!='nominal_axis_freq') for name in PHYSICS}
        self.last_force_t=None;self.last_ratio=1.;self.fixed_normal=None

    def step(self,row):
        t=row['t'];obs=row['observation'];sensing=row['sensing']
        if self.baseline.base.last_seen is not None and t-self.baseline.base.last_seen>robust.PARAMS['reset_gap_s']:
            self.reset_features()
        q,old=self.baseline.step(t,obs['center'] if obs.get('detected') else None)
        baseline=copy.deepcopy(q)
        rotors=[r for r in sensing['rotors'] if r.get('observed_now')]
        # Use the same geometry-only axis for every branch; frequency differential
        # is deliberately zero so axis_only does not receive a hidden RPM input.
        n=self.axis.step(t,rotors,0.,True)
        if sensing['four_frequencies'] and sensing['thrust_ratio'] is not None:
            self.last_ratio=float(np.clip(sensing['thrust_ratio'],*PARAMS['ratio_clip']));self.last_force_t=t
        frequency_age=None if self.last_force_t is None else t-self.last_force_t
        axis_age=None if self.axis.last_measurement_t is None else t-self.axis.last_measurement_t
        quality=float(sensing['frequency_quality']*obs.get('geometry_quality',0))
        active=bool(n is not None and frequency_age is not None and frequency_age<=PARAMS['feature_expiry_s']
                    and axis_age is not None and axis_age<=PARAMS['feature_expiry_s'] and quality>PARAMS['active_quality'])
        if active and self.fixed_normal is None:self.fixed_normal=np.asarray(n).copy()
        state=self.baseline.base.models['cv'].x
        diagnostics={}
        for name,model in self.models.items():
            direction=self.fixed_normal if name=='adaptive_fixed_axis_freq' else n
            ratio=1. if name=='adaptive_axis_only' else self.last_ratio
            if state is not None:
                model.update(t,state,direction,ratio,quality,bool(obs.get('detected')),bool(sensing['four_visible'] and active))
            branch_active=bool(active and q['valid'])
            if q['valid']:
                cv=q['coefficients']['cv']
                if branch_active:
                    omega=np.zeros(3) if name=='adaptive_fixed_axis_freq' else self.axis.omega
                    times,points=model.rollout(cv['position'],cv['velocity'],direction,omega,ratio)
                    c=dict(copy.deepcopy(cv),acceleration=model.acceleration(direction,ratio,np.asarray(cv['velocity'])).tolist(),
                        rollout_times=times.tolist(),rollout_positions=points.tolist())
                else:c=copy.deepcopy(cv)
                q['coefficients'][name]=c
            diagnostics[name]=dict(active=branch_active,updates=model.updates,update_allowed=model.update_allowed,
                effective_parameters=model.parameters.tolist(),disturbance_px_s2=model.disturbance.tolist(),
                normal=None if direction is None else np.asarray(direction).tolist(),ratio=ratio)
        q['predictions']={m:{str(h):evaluate(c,h) for h in PARAMS['horizons']} for m,c in q['coefficients'].items()}
        q['feature_diagnostic']=dict(active=bool(active and q['valid']),frequency_age_s=frequency_age,axis_age_s=axis_age,
            quality=quality,four_visible=sensing['four_visible'],four_frequencies=sensing['four_frequencies'],models=diagnostics)
        return q,baseline
