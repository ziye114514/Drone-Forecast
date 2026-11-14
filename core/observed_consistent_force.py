"""Use the published net-force prior also between strong image observations.

The older candidate uses a random-walk acceleration update on strong frames
but a decaying-force future rollout. Reuse the existing exact Singer update;
weak/missing-evidence policies, endpoint factors and all coefficients stay.
"""
from equilibrium_gain_prior import EquilibriumGainObserver,EquilibriumSeparatedObserver


class ObservedConsistentMixin:
    observed_consistent=True
    def __init__(self,default_gain=200.,velocity_sigma=1e6,startup_fit=False,
                 consistent_dynamics=True,two_point_init=False):
        super().__init__(default_gain,velocity_sigma,startup_fit,self.observed_consistent,two_point_init)


class ObservedConsistentSeparated(ObservedConsistentMixin,EquilibriumSeparatedObserver):pass

class ObservedConsistentObserver(ObservedConsistentMixin,EquilibriumGainObserver):
    alternative_type=ObservedConsistentSeparated


class PreservedSeparated(ObservedConsistentSeparated):observed_consistent=False

class PreservedObserver(ObservedConsistentObserver):
    observed_consistent=False
    alternative_type=PreservedSeparated
