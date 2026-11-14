"""Paper-consistent projected-axis magnitude ablation, same publication clock.

Thakur2604.18289 uses thrust times a3D unit body normal. Its2D projection need
not be unit length.69 deliberately normalised xy;97 retains the weak-projection
magnitude from existing geometry. This is a reproduction/design diagnostic,
not a new observation or calibrated3D attitude.
"""
from pathlib import Path
import inspect,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage88'),str(ROOT/'research/stage69')]
from paired88 import Paired
from force69 import Direction


class ProjectedDirection(Direction):
    def step(self,t,q,body,rotor_quality):
        n,diagnostic=super().step(t,q,body,rotor_quality)
        if diagnostic['current']:
            scale=1.
            if diagnostic['source']=='current_rotor_projection':
                scale=float(np.clip(np.linalg.norm(q['feature_diagnostic']['models']['adaptive_axis_only']['normal'][:2]),0,1))
            # Near-edge-on body line carries only direction: magnitude1 is
            # an explicit approximation, not a measured tilt.
            self.normal=self.normal*scale
        diagnostic.update(projected_magnitude=None if self.normal is None else float(np.linalg.norm(self.normal)),
            interpretation97='Weak-normal xy magnitude retained; body-line magnitude1 assumed. No calibrated pose.')
        return (self.normal.copy() if diagnostic['active'] else None),diagnostic


source=inspect.getsource(Paired.step)
old="fresh=bool(eligible and inc['direction']['current'] and inc['features']['periodic']['current'])"
assert source.count(old)==1
source=source.replace(old,"fresh=bool(q['force_clock97'])")
import textwrap
namespace=dict(Paired.step.__globals__)
exec(compile(textwrap.dedent(source),'<97 matched reference publication clock>','exec'),namespace)


class ProjectedPaired(Paired):
    projected_step=namespace['step']
    def reset(self):
        super().reset();self.inc.direction=ProjectedDirection();self.reference=Paired();self.reference_last=None
    def step(self,q,obs,body,activity,quality):
        if self.last_seen is not None and q['t']-self.last_seen>.5:self.reset()
        reference=self.reference.step(q,obs,body,activity,quality)
        prepared=dict(q,force_clock97=reference['diagnostics']['fresh'])
        result=self.projected_step(prepared,obs,body,activity,quality)
        assert result['diagnostics']['fresh']==reference['diagnostics']['fresh']
        self.reference_last=reference
        return result
