"""Qualification-matched fast-process control for the directional-force test."""
import numpy as np
from separated_force_response import IsotropicFastNoiseControl

class MatchedIsotropicControl(IsotropicFastNoiseControl):
    def _transition(self,dt):
        eligible=self.normal is not None and float(self.normal@self.normal)>1e-6
        previous=self.fast_isotropic_control;self.fast_isotropic_control=eligible
        try:return super()._transition(dt)
        finally:self.fast_isotropic_control=previous
