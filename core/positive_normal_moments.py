"""Moment-match the SAME half-normal startup hypothesis, no tuned variance.

Mean=sigma*sqrt(2/pi), variance=sigma^2*(1-2/pi). Only startup moments are
approximated by a Gaussian; posterior constraints are not repeatedly imposed.
"""
import numpy as np
from positive_normal_startup import PositiveStartupObserver,PositiveSeparatedObserver
from equilibrium_gain_prior import EquilibriumGainObserver,EquilibriumSeparatedObserver


class MatchedMoments:
    exact_initial_moments=True
    normal_variance_after_two_point=1e6*(1-2/np.pi)

class MomentSeparated(MatchedMoments,PositiveSeparatedObserver):pass
class MomentObserver(MatchedMoments,PositiveStartupObserver):alternative_type=MomentSeparated

class MomentGainSeparated(MatchedMoments,EquilibriumSeparatedObserver):pass
class MomentGainObserver(MatchedMoments,EquilibriumGainObserver):alternative_type=MomentGainSeparated
