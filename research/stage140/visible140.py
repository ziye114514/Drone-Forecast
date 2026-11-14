"""Dense kinematic fallback needs unclipped body observations across its fit."""
from pathlib import Path
from collections import deque
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage139')]
from dense139 import DenseGate as OriginalGate,Original,PARAMS as PREVIOUS,forecast
PARAMS=dict(PREVIOUS,visibility='Every detected body box in the same.35s fit window must lie strictly inside sensor pixel-center limits, reusing129 FOV convention. A truncated appearance box cannot qualify extra acceleration.',
    sensor=[1280,720],design='139 failure is known before this change: unmodeled dense fit alone is unsafe. One FOV-based reliability refinement; no new numeric threshold/endpoint rule.')
class VisibleGate(OriginalGate):
    def __init__(self):super().__init__();self.visible=deque()
    def step(self,t,body,output):
        box=np.asarray(body.get('bbox',[]),float);detected=bool(body.get('detected'));complete=bool(box.shape==(4,) and np.isfinite(box).all() and np.all(box[:2]>0) and np.all(box[2:]<[1279,719]))
        self.visible.append((t,not detected or complete))
        while self.visible and t-self.visible[0][0]>PARAMS['window_s']+1e-8:self.visible.popleft()
        result=super().step(t,body,output);visible=all(v for _,v in self.visible)
        if result['dense_motion_fallback']['active'] and not visible:
            diagnostic=result['dense_motion_fallback'];result=copy.deepcopy(output);result['dense_motion_fallback']=dict(diagnostic,active=False,reason='truncated_body_in_fit_window')
        result['dense_motion_fallback']['unclipped_fit_window']=visible
        return result
class Predictor(Original):
    def __init__(self):super().__init__();self.visible_gate=VisibleGate()
    def step(self,t,body,models):return self.visible_gate.step(t,body,super().step(t,body,models))
