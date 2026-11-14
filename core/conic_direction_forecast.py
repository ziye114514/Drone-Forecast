"""Native conic uncertainty -> future plane evolution; current state unchanged.

No artificial reduction by the number of ellipse children. Native per-conic
axis variance and disagreement with the shared plane both remain in the
uncertainty. If current independent nonflat arcs are insufficient, the old
uncertainty and all side-line rules are retained exactly.
"""
import copy
import numpy as np
from joint_phase_bank import usable
from normal_turn_forecast import NormalTurnForecast
from endpoint_normal_trend import EndpointNormalTrend


def current_uncertainty(row):
    n=np.asarray(row['structure'].get('axis_xy') if row['structure'].get('axis_xy') is not None else [],float)
    if n.shape!=(2,) or np.linalg.norm(n)<1e-8:return dict(qualified=False,reason='no_current_shared_plane')
    angle=float(np.arctan2(n[1],n[0]));members=[]
    for r in row['geometry'].get('rotors',[]):
        # inferred_shape labels a CURRENT partial-arc ellipse in this project;
        # it does not mean a hidden model point. usable() already demands
        # current qualified arc/curvature support and bounded fit uncertainty.
        if not usable(r) or r.get('shared_arc_recovered'):continue
        se=r.get('axis_angle_sigma_deg')
        if se is None or not np.isfinite(se) or se<=1e-6:continue
        theta=np.deg2rad(float(r['angle'])+90);delta=.5*float(np.angle(np.exp(2j*(theta-angle))))
        members.append(dict(angle_disagreement_rad=delta,fit_sigma_rad=float(np.deg2rad(se))))
    if len(members)<2:return dict(qualified=False,reason='need_two_current_independently_fitted_nonflat_arcs')
    variance=float(np.median([r['fit_sigma_rad']**2+r['angle_disagreement_rad']**2 for r in members]))
    return dict(qualified=True,reason='native_conic_axis_uncertainty_plus_shared_plane_disagreement',
        sigma_rad=float(np.sqrt(variance)),sigma_deg=float(np.rad2deg(np.sqrt(variance))),members=members,
        ellipse_child_count_used_to_reduce_noise=False,measurement_uncertainty_is_proxy_not_calibrated_GT=True)


class ConicDirectionForecast:
    def __init__(self,*,endpoint=False,override=True):
        self.trend=EndpointNormalTrend() if endpoint else NormalTurnForecast();self.override=override;self.observations=None

    def step(self,row):
        observations=row.get('state',{}).get('observations')
        if observations is not None and self.observations is not None and observations<self.observations:
            cls=type(self.trend);self.trend=cls()
        self.observations=observations
        noise=current_uncertainty(row)
        self.trend.axis_sigma_override_rad=noise['sigma_rad'] if self.override and noise['qualified'] else None
        motion=row.get('motion') or {};estimate=motion.get('estimate') or {};dt=motion.get('dt_s');flow=dict(valid=False)
        if dt and motion.get('current_observation') and not estimate.get('peak_at_search_boundary') and estimate.get('score',0)>0:
            flow=dict(valid=True,velocity=[estimate['dx']/dt,estimate['dy']/dt])
        trend=self.trend.observe(row['t'],row['structure'],row['coefficient'],dict(observation=flow))
        result=copy.deepcopy(row)
        def apply(r):
            normal=r.get('normal_turn_forecast') or {}
            if normal.get('applied') and trend.get('qualified'):normal['trend']=copy.deepcopy(trend)
            for part in r.get('force',{}).get('response_model_mixture',[]):apply(part)
        apply(result)
        result['conic_direction_forecast']=dict(current_uncertainty=noise,old_side_uncertainty_retained_if_no_nonflat_support=True,
            current_position_velocity_acceleration_unchanged=True,override_enabled=self.override,
            future_plane_trend=trend)
        return result
