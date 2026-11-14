"""Isolate geometry-only axis admission without changing frequency experts."""
from pathlib import Path
import inspect, sys, textwrap
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'core'),str(ROOT / 'research/stage52'), str(ROOT / 'research/stage29')]
from feedback52 import Predictor52, Managed52, prior
from select29 import Selector, MODELS

METHODS = ['shared', 'axis_gate', 'axis_learn']
PARAMS = dict(
    axis_expiry_s=prior.PARAMS['feature_expiry_s'],
    prediction_quality=prior.PARAMS['active_quality'],
    learning_quality=.45,
    scope='Geometry axis only: no new rotor/line detector, no guessed RPM. '
          'axis_gate changes rollout admission only; axis_learn also changes '
          'axis-only regressor admission. All RPM experts retain original gates.',
    current_support='New axis-only regressors require current observed rotor '
                    'support and axis measurement timestamp equal to current t. '
                    'Stored axes expire after original .16s. Original RLS .6s '
                    'warmup/.2s delayed feedback/.26s expiry/bounds preserved.',
    fallback='All branches retain identical CV fallback and six history experts. '
             'Pure history is reported separately. Axis-only assumes constant '
             'effective thrust input (ratio=1), not measured RPM or true thrust.')


def step_source():
    source = inspect.getsource(prior.Predictor.step)
    old = "        if active and self.fixed_normal is None:self.fixed_normal=np.asarray(n).copy()"
    new = old + '''
        current_support=bool(rotors and self.axis.last_measurement_t is not None
                             and abs(self.axis.last_measurement_t-t)<1e-8)
        axis_quality=float(self.axis.quality)
        axis_active=bool(n is not None and axis_age is not None
                         and axis_age<=PARAMS['feature_expiry_s']
                         and axis_quality>PARAMS['active_quality'])
        axis_launch=bool(current_support and axis_active)
'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = "            if state is not None:\n                model.update(t,state,direction,ratio,quality,bool(obs.get('detected')),bool(sensing['four_visible'] and active))\n            branch_active=bool(active and q['valid'])"
    new = '''            independent=bool(name=='adaptive_axis_only' and self.admission!='shared')
            learn_independent=bool(independent and self.admission=='axis_learn')
            if state is not None:
                model.update(t,state,direction,ratio,
                             axis_quality if learn_independent else quality,
                             bool(obs.get('detected')),
                             axis_launch if learn_independent else bool(sensing['four_visible'] and active))
            branch_active=bool((axis_active if independent else active) and q['valid'])'''
    assert source.count(old) == 1
    return textwrap.dedent(source.replace(old, new))


namespace = dict(prior.Predictor.step.__globals__)
exec(compile(step_source(), '<62 independent axis admission>', 'exec'), namespace)


class Predictor62(Predictor52):
    step = namespace['step']

    def __init__(self, admission):
        assert admission in METHODS
        self.admission = admission
        super().__init__('canonical_static')


class Managed62(Managed52):
    def __init__(self, admission):
        super().__init__()
        self.predictor = Predictor62(admission)


def diagnostics(managed, t, sensed):
    p = managed.predictor
    age = None if p.axis.last_measurement_t is None else t-p.axis.last_measurement_t
    return dict(axis_quality=p.axis.quality, axis_age_s=age,
                current_axis_support=age is not None and abs(age)<1e-8,
                observed_rotors=sum(bool(r.get('observed_now')) for r in sensed['rotors']),
                no_four_visible=not sensed['four_visible'],
                axis_queue=len(p.models['adaptive_axis_only'].queue),
                interpretation='Fresh stored axis is not independently validated attitude.')
