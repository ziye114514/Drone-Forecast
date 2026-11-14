"""Reuse bounded four-point layout without soft periodic feature changes."""
from pathlib import Path
import importlib.util
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('bounded_layout19',ROOT/'research/stage19/layout19.py')
layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
PARAMS=layout.PARAMS


class Observer:
    def __init__(self,w,h,method='layout20'):
        if method not in ['neighbor20','layout20']:raise ValueError(method)
        self.w,self.h=w,h;self.observer=layout.Observer(w,h,False,method=='layout20')
        self.spectrum=dict(mask=np.zeros(((h+7)//8,(w+7)//8),bool),hz=np.zeros(((h+7)//8,(w+7)//8)))

    def step(self,events,t):
        p=layout.counts(events,self.w,self.h);p['on']/=2;p['off']/=2
        q=self.observer.step(events,t,p,self.spectrum)
        q['audit']=dict(q['audit'],structured_layout=q['layout'])
        return q
