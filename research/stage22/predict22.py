"""Causal cached-position predictor; labels are not imported here."""
from pathlib import Path
import importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reused_motion14',ROOT/'research/stage14/motion.py')
motion=importlib.util.module_from_spec(spec);spec.loader.exec_module(motion)
PARAMS=dict(min_real_observations=5,max_coast_s=.16,reset_gap_s=.5,horizons=[.1,.2,.4])


class Predictor:
    def __init__(self):
        self.models={'cv':motion.CVFilter(),'cv_hist_acc':motion.HistoryAccelCV()}
        self.last_t=None;self.last_seen=None;self.last_center=None;self.observations=0

    def step(self,t,center):
        if self.last_seen is not None and t-self.last_seen>PARAMS['reset_gap_s']:
            self.models={'cv':motion.CVFilter(),'cv_hist_acc':motion.HistoryAccelCV()}
            self.last_seen=None;self.last_center=None;self.observations=0
        dt=None if self.last_t is None else max(t-self.last_t,1e-6)
        for model in self.models.values():
            if model.x is None:
                if center is not None:model.initialize(center)
                continue
            if dt is not None:model.predict(dt)
            if center is not None:model.update(center)
        if center is not None:self.last_seen=t;self.last_center=np.array(center,float);self.observations+=1
        self.last_t=t;age=None if self.last_seen is None else t-self.last_seen
        valid=self.observations>=PARAMS['min_real_observations'] and age is not None and age<=PARAMS['max_coast_s']
        coefficients={}
        if valid:
            coefficients['hold']=dict(position=self.last_center.tolist(),velocity=[0.,0.],acceleration=[0.,0.])
            for name,model in self.models.items():
                acceleration=model.acceleration() if name=='cv_hist_acc' else np.zeros(2)
                coefficients[name]=dict(position=model.x[:2].tolist(),velocity=model.x[2:4].tolist(),acceleration=acceleration.tolist())
        return dict(t=t,valid=bool(valid),last_measurement_age_s=age,real_observations=self.observations,
            anchor_has_current_measurement=center is not None,coefficients=coefficients,
            predictions={name:{str(h):evaluate(c,h) for h in PARAMS['horizons']} for name,c in coefficients.items()})


def evaluate(coefficients,horizon):
    p=np.array(coefficients['position']);v=np.array(coefficients['velocity']);a=np.array(coefficients['acceleration'])
    return (p+v*horizon+.5*a*horizon*horizon).tolist()
