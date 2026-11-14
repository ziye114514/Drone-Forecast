"""Body-box candidate gating after complete global contour extraction."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage9'),str(ROOT/'research/stage55')]
from recovery import RecoveryFrontend
from roi55 import GeometryROI

PARAMS=dict(margin_fraction=.25,expiry_us=160000,
    stage='After original full contour fitting, before original candidate ranking/grouping.',
    preserved='All current events, global nonzero density quantile, morphology and contour shapes. Original full RAW frequency history/stride, tracking, phase and15 models unchanged.',
    prior='Use actual55 GeometryROI on an empty event view only to derive the identical validated box. Candidate centroid is a gate, not a fabricated rotor or a body-center truth.')

class CandidateFrontend(RecoveryFrontend):
    def __init__(self,w,h,method):
        super().__init__(w,h,method);self.region=GeometryROI(w,h,'body');self.candidate_roi=None
        self.gate_counts=dict(candidate_calls=0,candidates_before_gate=0,candidates_after_gate=0)

    def set_hint(self,events,hint,t,mode):
        assert mode in ['off','body']
        self.region.mode=mode
        _,diagnostic=self.region.apply(events[:0],hint,t)
        self.candidate_roi=diagnostic['roi'] if diagnostic['mode']=='body_roi' else None
        if self.candidate_roi is not None:diagnostic['mode']='body_candidate_roi'
        diagnostic.update(raw_events=len(events),geometry_events=len(events),geometry_input='complete_current_events',candidate_only=True)
        self.gate_counts=dict(candidate_calls=0,candidates_before_gate=0,candidates_after_gate=0)
        return diagnostic

    def filter_candidates(self,found):
        self.gate_counts['candidate_calls']+=1;self.gate_counts['candidates_before_gate']+=len(found)
        if self.candidate_roi is None:kept=found
        else:
            lo=np.asarray(self.candidate_roi[:2]);hi=np.asarray(self.candidate_roi[2:])
            kept=[r for r in found if np.all(np.asarray(r['center'])>=lo) and np.all(np.asarray(r['center'])<hi)]
        self.gate_counts['candidates_after_gate']+=len(kept)
        return kept

    def _candidates(self,*args,**kwargs):
        return self.filter_candidates(super()._candidates(*args,**kwargs))
