"""Matured force-model credit with the existing future-process uncertainty.

This does not change issued forecasts, remove maneuver errors, or observe a
future command. It changes ONLY how already matured errors teach subsequent
model credit. Existing .35s/1000px/s² Singer process supplies Qpp(h).
"""
import numpy as np
from adaptive_force_memory import AdaptiveForceMemory
from singer_position_state import transition
from equilibrium_gain_prior import CovarianceOnlyObserver


class HorizonAwareMemory(AdaptiveForceMemory):
    def step(self,t,*args,**kwargs):
        previous_weight=self.weight
        result,info=super().step(t,*args,**kwargs)
        if info['updated']:
            numerator=0.;denominator=36.
            for stamp,delta,error,h in self.rows:
                _,Q=transition(round(h*1e6),.35);variance=36.+float(Q[0,0])
                residual=error-previous_weight*delta
                robust=min(1.,np.sqrt(variance)/max(float(np.linalg.norm(residual)),1e-9))
                w=np.exp(-(t-stamp)/.35)/3.*36./variance
                numerator+=w*robust*float(delta@error);denominator+=w*robust*float(delta@delta)
            self.weight=float(np.clip(numerator/denominator,0.,1.))
        info.update(weight=self.weight,forecast_process_variance_in_credit=True,
            error_threshold_increases_with_existing_process_variance=True,
            old_predictions_and_scoring_unchanged=True)
        return result,info


class HorizonCreditObserver(CovarianceOnlyObserver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.credit.memory=HorizonAwareMemory()
