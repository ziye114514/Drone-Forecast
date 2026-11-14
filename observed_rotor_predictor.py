"""Explicit current-observation candidate; preserved learned baseline unchanged."""
from learned_rotor_predictor import LearnedRotorPredictor
from observation_roles import RoleSeparatedContinuation


class ObservedRotorPredictor(LearnedRotorPredictor):
    def __init__(self,*args,**kwargs):
        if kwargs.get('periodic_continuation'):
            raise ValueError('Use the explicit observation candidate without legacy continuation')
        super().__init__(*args,**kwargs)
        self.front.periodic_continuation=RoleSeparatedContinuation()


class PriorConsistentObservedPredictor(ObservedRotorPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.periodic_continuation=RoleSeparatedContinuation(prior_consistent=True)


class VisibleObservedPredictor(PriorConsistentObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from native_fov_observation import VisibleEnvelopeResponse
        self.visible_envelope=VisibleEnvelopeResponse()
    def step(self,events,t):
        import time
        started=time.perf_counter();row,frame=super().step(events,t)
        row=self.visible_envelope.step(row);row['compute_ms']=(time.perf_counter()-started)*1000
        return row,frame


class BoundedObservedPredictor(VisibleObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.periodic_continuation=RoleSeparatedContinuation(prior_consistent=True,allow_period_gap=True)


class ReferenceGuardedObservedPredictor(BoundedObservedPredictor):
    """Accept a current weak displacement without redefining body reference."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from reference_qualification import QualifiedReferenceResponse
        self.learned_response=QualifiedReferenceResponse(self.learned_response)


class EndpointObservedPredictor(ReferenceGuardedObservedPredictor):
    """Same native measurements, with shared-image relative-noise ownership."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from shared_endpoint_observer import EndpointAdaptiveObserver
        self.force=EndpointAdaptiveObserver()


class ProjectionQualifiedObservedPredictor(EndpointObservedPredictor):
    """A line/conic representation switch must not fabricate hidden force."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from shared_endpoint_observer import ProjectionQualifiedObserver
        self.force=ProjectionQualifiedObserver()


class CompleteObservedPredictor(ProjectionQualifiedObservedPredictor):
    """Qualified early rotor birth plus a separate reference warm-up."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from temporal_rotor_birth import TemporalBirthContinuation
        from bootstrap_native_reference import BootstrapReferenceResponse
        self.front.periodic_continuation=TemporalBirthContinuation(seed_motion_search=True)
        self.learned_response=BootstrapReferenceResponse(self.learned_response)


class AccuracyObservedPredictor(CompleteObservedPredictor):
    """Diffuse entering velocity; explicit20ms native-envelope target."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from diffuse_rotor_startup import DiffuseStartupObserver
        from window_position_observation import WindowPositionObservation
        self.force=DiffuseStartupObserver()
        self.window_observation=WindowPositionObservation(33333,20000)

    def step(self,events,t):
        import time
        started=time.perf_counter();row,frame=super().step(events,t)
        row=self.window_observation.step(row)
        row['compute_ms']=(time.perf_counter()-started)*1000
        return row,frame


class LocalGeometryObservedPredictor(AccuracyObservedPredictor):
    """Do not let thin global layouts preclude supported local arc splitting."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.geometry.side_axis_budget=False
        self.front.geometry.local_conic_curvature=True


class PositivePriorObservedPredictor(AccuracyObservedPredictor):
    """An explicit unmeasured positive normal-drive startup hypothesis."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from positive_normal_startup import PositiveStartupObserver
        self.force=PositiveStartupObserver()


class PartialForceObservedPredictor(PositivePriorObservedPredictor):
    """Anonymous visible frequency groups and a broad frequency-scale prior."""
    covariance_only=True
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from equilibrium_gain_prior import EquilibriumGainObserver,CovarianceOnlyObserver,NominalFrequencyBank
        from anonymous_rotor_group import AnonymousRotorGroup
        self.force=CovarianceOnlyObserver() if self.covariance_only else EquilibriumGainObserver()
        self.period_proxy=NominalFrequencyBank(AnonymousRotorGroup(self.period_proxy))


class PolarityPartialObservedPredictor(PartialForceObservedPredictor):
    """Current ON/OFF disagreement qualifies motion uncertainty, not position."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.polarity_motion_quality=True


class ApertureObservedPredictor(PolarityPartialObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.peak_shape_uncertainty=True


class JointPeakObservedPredictor(ApertureObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.joint_peak_refinement=True


class JointOnlyObservedPredictor(JointPeakObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.peak_shape_uncertainty=False


class FullJointObservedPredictor(JointOnlyObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.joint_peak_full_neighborhood=True


class MultichannelObservedPredictor(JointOnlyObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.polarity_separated_motion=True


class SymmetricObservedPredictor(JointOnlyObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.symmetric_native_motion=True


class SupportedPlaneObservedPredictor(JointOnlyObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.plane_motion=True
        self.front.plane_center_hull_required=True


class TimestampedMotionObservedPredictor(JointOnlyObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.timestamped_motion=True


class SymmetricSurfaceObservedPredictor(JointOnlyObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.symmetric_surface_motion=True


class CompleteSymmetricSurfaceObservedPredictor(SymmetricSurfaceObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.symmetric_surface_complete_only=True


class LocalContrastObservedPredictor(JointOnlyObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.local_contrast_motion=True


class LocalContrastSymmetricObservedPredictor(LocalContrastObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.symmetric_native_motion=True


class LocalContrastPlaneObservedPredictor(LocalContrastObservedPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.front.plane_motion=True
        self.front.plane_center_hull_required=True
        self.front.plane_local_contrast=True
