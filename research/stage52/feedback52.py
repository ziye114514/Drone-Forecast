"""Separate launch-feature validity from delayed position-feedback validity."""
from pathlib import Path
import inspect,sys,textwrap
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/stage51'))
from reference51 import Managed,Predictor,prior

def update_source():
    source=textwrap.dedent(inspect.getsource(prior.ProjectedThrustModel.update))
    substitutions={
        'if not self.update_allowed or t-self.started<.6:return':'if not self.adaptive or not observed or t-self.started<.6:return',
        'self.queue.append((t,state[:2].copy(),state[2:4].copy(),np.asarray(n).copy(),s,quality))':'if self.update_allowed:self.queue.append((t,state[:2].copy(),state[2:4].copy(),np.asarray(n).copy(),s,quality))'}
    for old,new in substitutions.items():assert source.count(old)==1;source=source.replace(old,new)
    return source

namespace=dict(prior.ProjectedThrustModel.update.__globals__)
exec(compile(update_source(),'<stage52 guarded delayed feedback>','exec'),namespace)
class DelayedFeedback(prior.ProjectedThrustModel):update=namespace['update']

class Predictor52(Predictor):
    def reset_features(self):
        super().reset_features()
        self.models={name:DelayedFeedback(self.cfg,name!='nominal_axis_freq') for name in prior.PHYSICS}

class Managed52(Managed):
    def __init__(self):
        super().__init__('stable_reference');self.predictor=Predictor52('canonical_static')
