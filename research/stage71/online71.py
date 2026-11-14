"""Actual RAW inline integration of68 activity,64 axis,69 force and70 guard."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage68'),str(ROOT/'research/stage69'),str(ROOT/'research/stage70'),str(ROOT/'research/stage64')]
from activity68 import Activity
from force69 import ForceForecast
from guard70 import Guard
from compensation64 import Supports,PastVelocity,body_measurement

class OnlineExtras:
    def __init__(self,w,h):
        self.activity=Activity(w,h);self.supports=Supports(w,h);self.velocity=PastVelocity()
        self.force=ForceForecast();self.guard=Guard();self.current=None
    def measure(self,t,events,body):
        v,va=self.velocity.before(t-.02)
        box=body['bbox'] if body.get('detected') and body.get('confidence',0)>=.45 else None
        activity=self.activity.step(events,box,t,v)
        for cutoff in [t-.01,t]:
            vv,audit=self.velocity.before(cutoff-.01)
            e=events[(events['t']>=round((cutoff-.01)*1e6))&(events['t']<round(cutoff*1e6))]
            self.supports.step(e,cutoff,vv)
        measurement=body_measurement(self.supports.density['aligned_total'],box) if box is not None else dict(valid=False,reason='no_current_NN_identity')
        self.current=dict(t=t,activity=activity,body=measurement,velocity_audit=va)
        return activity,measurement,va
    def forecast(self,q,obs,rotor_quality):
        assert self.current is not None and q['t']==self.current['t']
        q=self.force.step(q,obs,self.current['body'],self.current['activity'],rotor_quality)
        return self.guard.step(q,obs)
    def commit(self,t,body,q):self.velocity.commit(dict(t=t,body_observation=body,frontends=dict(delayed_feedback=q)))
