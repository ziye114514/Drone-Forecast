"""Past full-view leave-one-out checks choose partial geometry, one shared KF.

Only current four-observation frames can update the adequacy history. There
is no full-four reference lookup during a three-observation frame. This is
self-consistency supervision, not real body-center ground truth.
"""
from pathlib import Path
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage119')]
from partial119 import PartialModel,native_H
from study119 import CenterBank
PARAMS=dict(validation_tau_s=.2,minimum_valid_full_checks=5,
    score='Per full frame, mean squared leave-one-rotor-out model center disagreement with current full-four fixed-model center, across four removals. No COM truth.',
    decision='On three-observation frames choose affine3 or conic3 by strictly previous EMA score after five successful full-view checks; otherwise body_hold. One common CenterBank, no position/velocity reset at switches.',
    unknowns='Generic cross-validation and adaptive model selection have prior art. Here it checks whether event-derived rotor ellipses support a retained projective model. Self-consistency can favor a biased four-view model.')

class ValidatedSelector:
    def __init__(self):
        self.prior=None;self.errors=None;self.checks=0;self.last_validation=None;self.last_body=None;self.bank=CenterBank()
    def step(self,r):
        t=r['t'];body=r['body'];rotors=r['input_rotors']
        if self.last_body is not None and t-self.last_body>.5:self.__init__()
        if body.get('detected'):self.last_body=t
        before=None if self.errors is None else dict(self.errors);chosen='conic3' if len(rotors)==4 else 'body_hold'
        if len(rotors)==3 and self.errors is not None and self.checks>=5:
            chosen=min(self.errors,key=self.errors.get)
        candidate=r['outputs'][chosen]['center']
        if len(rotors)==3 and not candidate['current']:chosen='body_hold';candidate=r['outputs'][chosen]['center']
        d=copy.deepcopy(candidate);q=self.bank.step(t,body,d,r['axis']);validation=None
        # Validation happens AFTER this step's decision and filter update.
        if len(rotors)==4 and candidate['current'] and self.prior is not None:
            f=candidate['measurement']['fit'];target=np.array(f['geometric_center']);loss={m:[] for m in ['affine3','conic3']}
            for m in loss:
                proto=PartialModel(m);proto.H=np.array(self.prior['H']);proto.shape=np.array(self.prior['shape']);proto.body=np.array(self.prior['body'])
                for i in range(4):
                    measurement,_=proto.measure(body,rotors[:i]+rotors[i+1:])
                    if measurement['valid']:loss[m].append(float(np.sum((np.array(measurement['center'])-target)**2)))
            if all(len(z)==4 for z in loss.values()):
                current={m:float(np.mean(v)) for m,v in loss.items()}
                if self.errors is None:self.errors=current
                else:
                    dt=t-self.last_validation;alpha=float(-np.expm1(-dt/PARAMS['validation_tau_s']))
                    self.errors={m:self.errors[m]+alpha*(current[m]-self.errors[m]) for m in current}
                self.last_validation=t;self.checks+=1;validation=dict(mean_squared_center_disagreement_px2=current,full_reference_current=True,heldout_fits_each=4)
        # A current selected conic/affine model can warm-start the next full
        # validation. Retained state is never treated as a new observation.
        if candidate['current']:
            f=candidate['measurement']['fit']
            if 'pixel_homography' in f or 'homography_normalized' in f:
                self.prior=dict(H=f.get('pixel_homography') or native_H(f).tolist(),shape=[f['radius_in_template_units'],f['frame_aspect']],body=body['center'])
        return dict(center=d,forecast=q,selection=dict(chosen=chosen,prior_errors_px2=before,checks_before=self.checks-int(validation is not None),
            validation=validation,errors_after_px2=self.errors,uses_four_observations_when_three_available=False,filter_reset_on_switch=False))
