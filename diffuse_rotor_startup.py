"""An entering target has no measured zero-velocity constraint.

The very broad initial velocity prior approximates diffuse conditioning.
Only its first two measured images can constrain velocity; the tentative
birth displacement remains a search hint. All measurement factors, force
priors, default observer classes, and model coefficients remain unchanged.
"""
from shared_endpoint_observer import ProjectionQualifiedObserver, ProjectionQualifiedSeparatedObserver


class DiffuseVelocityPrior:
    def __init__(self, default_gain=200., velocity_sigma=1e6, startup_fit=False,
                 consistent_dynamics=False, two_point_init=False):
        super().__init__(default_gain, 1e6, startup_fit, consistent_dynamics, two_point_init)


class DiffuseSeparatedObserver(DiffuseVelocityPrior, ProjectionQualifiedSeparatedObserver):
    pass


class DiffuseStartupObserver(DiffuseVelocityPrior, ProjectionQualifiedObserver):
    alternative_type=DiffuseSeparatedObserver
