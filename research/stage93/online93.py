"""RAW inline complementary history; unchanged49 plus six explicit candidates."""
from pathlib import Path
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage84'),str(ROOT/'research/stage92'),str(ROOT/'research/stage74')]
from batched84 import OnlineExtras as Previous
from hybrid74 import ControlledFilter,prior
from router92 import Router92

# Selected using TRAIN0/4/6/10 only in90. No annotation/score/config read in
# production. The old defaults/49 branches remain unchanged.
SIGMA=6400.
NAMES={'history_router':'complement_history','history_fixed':'complement_history_fixed',
    'fallback_router':'complement_fallback','full_router':'complement_full','full_fixed':'complement_full_fixed'}


class Kinematic:
    def __init__(self):self.reset()
    def reset(self):self.filter=None;self.last_seen=None;self.last_t=None
    def step(self,t,z):
        if self.last_seen is not None and t-self.last_seen>.5:self.reset()
        if z is not None:self.last_seen=t
        dt=.02 if self.last_t is None else t-self.last_t;self.last_t=t
        if self.filter is None:
            if z is not None:self.filter=ControlledFilter(z)
        else:
            f=self.filter;f.dt=dt;f.sigma=SIGMA;f.a=np.zeros(2);f.predict()
            if z is not None:f.update(z)
        if self.filter is None:return None
        f=self.filter
        return dict(position=f.x[:2].tolist(),velocity=f.x[2:].tolist(),acceleration=[0.,0.])


class OnlineExtras(Previous):
    def __init__(self,w,h):super().__init__(w,h);self.kinematic=Kinematic();self.router92=Router92()
    def forecast(self,q,obs,rotor_quality):
        q=super().forecast(q,obs,rotor_quality)
        k=self.kinematic.step(q['t'],obs['center'] if obs.get('detected') else None)
        primary=dict(t=q['t'],valid=q['valid'],coefficients={},diagnostics={'fresh':q['separated_diagnostic']['current']})
        kin=dict(t=q['t'],valid=q['valid'],coefficients={})
        if q['valid']:
            assert k is not None
            primary['coefficients']['event']=q['coefficients']['mmae_state_fresh']
            kin['coefficients']={m:q['coefficients'][m] for m in ['cv','trusted_acc']};kin['coefficients']['kf_tuned']=k
        routed=self.router92.step(primary,kin,obs)
        q=dict(q,coefficients=dict(q['coefficients']),predictions=dict(q['predictions']))
        if q['valid']:
            q['coefficients']['kinematic_train6400']=k
            for old,new in NAMES.items():q['coefficients'][new]=copy.deepcopy(routed['coefficients'][old])
            for name in ['kinematic_train6400',*NAMES.values()]:q['predictions'][name]={str(h):prior.evaluate(q['coefficients'][name],h) for h in [.1,.2,.4]}
        q['complement_diagnostic']=routed['diagnostics']
        return q
