"""Controlled event erasure after clear4-point acquisition; no real-GT claim."""
from pathlib import Path
import importlib.util
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

layout=module('layout19_for_occlusion',ROOT/'research/stage19/layout19.py')
windows=module('windows21_for_occlusion',ROOT/'research/stage21/windows21.py')
prediction=module('prediction22_for_occlusion',ROOT/'research/stage22/predict22.py')
PARAMS=dict(start=23.,duration=1.8,erase_start=23.4,ellipse_margin_px=[8.,6.],ellipse_radius_fraction=.65,
    occlusion_durations=[.12,.30,.60],patterns=['adjacent','opposite'],reference_error_threshold_px=20.)


class OracleMask:
    """Uncorrupted detector sets mask geometry; never handed to learner state."""
    def __init__(self):self.rotors=None;self.previous=None

    def update(self,observation):
        if not observation['detected'] or len(observation['rotors'])!=4:return False
        current=observation['rotors'];p=np.array([[r['cx'],r['cy']] for r in current])
        if self.previous is None:
            center=p.mean(0);order=np.argsort(np.arctan2(p[:,1]-center[1],p[:,0]-center[0]));p=p[order];current=[current[i] for i in order]
        else:
            rows,cols=linear_sum_assignment(np.linalg.norm(self.previous[:,None,:]-p[None,:,:],axis=2))
            order=cols[np.argsort(rows)];p=p[order];current=[current[i] for i in order]
        self.previous=p;self.rotors=current;return True

    def erase(self,events,pattern):
        if self.rotors is None:return events,False
        slots=[0,1] if pattern=='adjacent' else [0,2]
        keep=np.ones(len(events),bool)
        for i in slots:
            r=self.rotors[i];dx=events['x'].astype(float)-r['cx'];dy=events['y'].astype(float)-r['cy'];angle=np.deg2rad(r['angle'])
            u=dx*np.cos(angle)+dy*np.sin(angle);v=-dx*np.sin(angle)+dy*np.cos(angle)
            a=PARAMS['ellipse_radius_fraction']*r['major']+PARAMS['ellipse_margin_px'][0]
            b=PARAMS['ellipse_radius_fraction']*r['minor']+PARAMS['ellipse_margin_px'][1]
            keep&=(u/a)**2+(v/b)**2>1.
        return events[keep],True


class Learner:
    def __init__(self,w,h,kind):
        self.kind=kind;self.w=w;self.h=h;self.predictor=prediction.Predictor()
        self.front=windows.Observer(w,h,'native20') if kind=='neighbor' else layout.Observer(w,h,False,True)
        self.spectrum=dict(mask=np.zeros(((h+7)//8,(w+7)//8),bool),hz=np.zeros(((h+7)//8,(w+7)//8)))

    def step(self,events,t):
        if self.kind=='neighbor':q=self.front.step(events,t)
        else:
            prepared=layout.counts(events,self.w,self.h);prepared['on']/=2;prepared['off']/=2
            q=self.front.step(events,t,prepared,self.spectrum)
        o=q['observation'];forecast=self.predictor.step(t,o['center'] if o['detected'] else None)
        return dict(observation=o,layout=q.get('layout',{}),forecast_valid=forecast['valid'],forecast_measurement_age_s=forecast['last_measurement_age_s'])
