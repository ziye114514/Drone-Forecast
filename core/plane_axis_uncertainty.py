"""Separate current plane-direction information from four-hub identity.

Use actual conic angular uncertainty AND disagreement, retaining the existing
5degree metric-axis floor. Correlated conics are never averaged down by N.
No new hubs, periods or model lock are created by this information packet.
"""
import numpy as np


def apply(structure,geometry):
    if structure.get('pose_source')!='current_qualified_visible_conics' or structure.get('metric_axis_sigma_proxy_deg') is not None:return False
    n=np.asarray(structure.get('axis_xy') or [],float)
    if n.shape!=(2,) or np.linalg.norm(n)<1e-6:return False
    current=[r for r in geometry.get('rotors',[]) if r.get('current_observation',True) and not r.get('fov_clipped') and r.get('ratio',0)>.16]
    if len(current)<2:return False
    sigma=np.array([r.get('axis_angle_sigma_deg',float('inf')) for r in current],float)
    if not np.isfinite(sigma).all():return False
    theta=np.arctan2(n[1],n[0]);angles=np.array([np.deg2rad(r['angle'])+np.pi/2 for r in current])
    delta=.5*np.angle(np.exp(2j*(angles-theta)));scatter=float(np.sqrt(np.mean(delta*delta)))
    uncertainty=max(5.,float(np.sqrt(np.max(sigma)**2+np.rad2deg(scatter)**2)))
    structure['metric_axis_sigma_proxy_deg']=uncertainty
    structure['axis_information']=dict(source='current_conic_uncertainty_and_disagreement',angle_sigma_proxy_deg=uncertainty,
        current_conic_count=len(current),correlated_uncertainty_not_divided_by_count=True,
        four_hub_lock_or_motor_identity_created=False,uncertainty_is_not_calibrated_attitude_truth=True)
    return True
