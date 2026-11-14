"""Hypothesis: zero net acceleration need not mean zero normal drive.

Only the unmeasured normal-response prior mean changes. The mean is the
half-normal moment of the EXISTING1000px/s^2 prior, with its full old variance
retained conservatively. Present net acceleration is preserved; an external
counterbalance is implicit in e=d-h*n. This is not a measured thrust or a
globally enforced positivity constraint. Position evidence may contradict it.
"""
import numpy as np
from diffuse_rotor_startup import DiffuseStartupObserver,DiffuseSeparatedObserver


class PositiveNormalPrior:
    exact_initial_moments=False
    def _initialize(self,z):
        super()._initialize(z)
        self.x[self.response_index]=float(np.sqrt(self.P[self.response_index,self.response_index])*np.sqrt(2/np.pi))
        if self.exact_initial_moments:
            self.P[self.response_index,self.response_index]*=1-2/np.pi


class PositiveSeparatedObserver(PositiveNormalPrior,DiffuseSeparatedObserver):pass


class PositiveStartupObserver(PositiveNormalPrior,DiffuseStartupObserver):
    alternative_type=PositiveSeparatedObserver
