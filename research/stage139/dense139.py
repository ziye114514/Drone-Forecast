"""Confidence-conditioned kinematic fallback when no rotor structure exists.

Uses current/past detector support, not event count as thrust or a guessed
normal. Keeps135 geometry motion unchanged once a rotor model has learned.
"""
from pathlib import Path
from collections import deque
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage135')]
from predictor135 import Predictor as Original,forecast
PARAMS=dict(window_s=.35,step_s=.02,minimum_span_s=.33,minimum_coverage=.9,maximum_detection_gap_s=.04000001,
    maximum_median_residual_px=6.,minimum_inlier_fraction=.9,
    reuse='Existing132/23 robust acceleration and uncertainty shrink,3000px/s2 cap. No new fit or parameter sweep.',
    gate='Without qualified rotor model: require current detection, dense current.35s support, no gap>40ms, full-span robust fit within original6px measurement noise and>=.9 inlier fraction.',
    interpretation='Kinematic fallback reliability only. No inferred rotor center, RPM, thrust, wind, COM or physical attitude; not a novelty claim.',
    prior='135 development results known. One fixed confidence rule before139 scores; exposed historical observations, no blind-test claim.')

class DenseGate:
    def __init__(self):self.history=deque()
    def step(self,t,body,output):
        self.history.append((t,bool(body.get('detected'))))
        while self.history and t-self.history[0][0]>PARAMS['window_s']+1e-8:self.history.popleft()
        times=[s for s,valid in self.history if valid];coverage=len(times)/18.;span=times[-1]-times[0] if times else 0.;gap=max(np.diff(times),default=0.)
        d=output['acceleration']['raw_estimate'];eligible=bool(body.get('detected') and coverage>=PARAMS['minimum_coverage'] and span>=PARAMS['minimum_span_s']-1e-8 and gap<=PARAMS['maximum_detection_gap_s'] and d.get('valid') and d.get('median_abs_residual_px',np.inf)<=PARAMS['maximum_median_residual_px'] and d.get('robust_inlier_fraction',0)>=PARAMS['minimum_inlier_fraction'])
        result=copy.deepcopy(output);active=eligible and not output['structure_supported']
        if active:
            for c in result['coefficients'].values():
                if c is not None:c['acceleration']=list(output['acceleration']['acceleration'])
        result['predictions']={name:{str(h):forecast(c,h) for h in [.1,.2,.4]} for name,c in result['coefficients'].items() if c is not None}
        result['dense_motion_fallback']=dict(active=active,eligible=eligible,coverage=coverage,span_s=span,maximum_gap_s=float(gap),physical_thrust_estimated=False)
        return result

class Predictor(Original):
    def __init__(self):super().__init__();self.dense=DenseGate()
    def step(self,t,body,models):return self.dense.step(t,body,super().step(t,body,models))
