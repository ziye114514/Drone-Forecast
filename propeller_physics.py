"""Known-geometry propeller laws and rank-aware force allocation.

Primary references: UIUC propeller database and ArduPilot multicopter model.
No blade shape, diameter, metric pose, RPM or inertia is fabricated from RAW.
"""
import numpy as np
from scipy.optimize import lsq_linear


def thrust_newtons(revolutions_per_second,diameter_m,CT,air_density=1.225):
    n=np.asarray(revolutions_per_second,float);d=np.asarray(diameter_m,float);ct=np.asarray(CT,float)
    if not np.isfinite(n).all() or not np.isfinite(d).all() or not np.isfinite(ct).all():raise ValueError('Finite physical parameters required')
    if np.any(n<0) or np.any(d<=0) or np.any(ct<0) or air_density<=0:raise ValueError('Invalid propeller parameters')
    return air_density*n*n*d**4*ct


def static_coefficient_curve(text):
    """Read a UIUC-style static RPM/CT/CP table; reject extrapolation."""
    rows=[]
    for line in text.splitlines():
        try:values=[float(v) for v in line.split()]
        except ValueError:continue
        if len(values)==3 and 100<=values[0]<=200000 and 0<=values[1]<=1:rows.append(values[:2])
    if len(rows)<2:raise ValueError('At least two static RPM/CT rows required')
    rows=np.asarray(sorted(rows));rpm,ct=rows.T
    if np.any(np.diff(rpm)<=0):raise ValueError('Distinct RPM samples required')
    def evaluate(n_rev_s):
        q=np.asarray(n_rev_s,float)*60.
        if np.any(q<rpm[0]) or np.any(q>rpm[-1]):raise ValueError('Outside measured static curve; no silent extrapolation')
        return np.interp(q,rpm,ct)
    return evaluate


def parallel_allocation(rotor_positions_m):
    """Map four parallel positive thrusts to total thrust, roll and pitch moment."""
    r=np.asarray(rotor_positions_m,float)
    if r.shape!=(4,3) or not np.isfinite(r).all():raise ValueError('Four metric body-frame hub positions required')
    return np.vstack([np.ones(4),r[:,1],-r[:,0]])


def wrench(rotor_positions_m,thrusts_N,normals_body=None,reaction_torques_Nm=None):
    r=np.asarray(rotor_positions_m,float);T=np.asarray(thrusts_N,float)
    n=np.tile([0.,0.,1.],(4,1)) if normals_body is None else np.asarray(normals_body,float)
    if r.shape!=(4,3) or T.shape!=(4,) or n.shape!=(4,3):raise ValueError('Invalid rotor arrays')
    if not np.isfinite(r).all() or not np.isfinite(T).all() or not np.isfinite(n).all() or np.any(T<0):raise ValueError('Invalid finite nonnegative thrust data')
    if not np.allclose(np.linalg.norm(n,axis=1),1.):raise ValueError('Unit rotor normals required')
    force=T[:,None]*n;moment=np.cross(r,force)
    if reaction_torques_Nm is not None:moment+=np.asarray(reaction_torques_Nm,float)[:,None]*n
    return force.sum(0),moment.sum(0)


def infer_missing_thrust(observed,prior_mean,prior_sigma,H=None,moment_or_total=None,
                         constraint_sigma=None,constraints_calibrated=False,measurement_sigma=.01):
    """Positive MAP completion. Rank is computed WITHOUT the prior.

    H rows must be calibrated total-force or torque equations, not an attitude
    angle. A prior-selected solution is explicitly not a uniquely observed one.
    """
    mean=np.asarray(prior_mean,float);sigma=np.asarray(prior_sigma,float)
    if mean.shape!=(4,) or sigma.shape!=(4,) or not np.isfinite(mean).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):raise ValueError('Finite four-rotor prior and positive sigma required')
    if not np.isfinite(measurement_sigma) or measurement_sigma<=0:raise ValueError('Positive measurement sigma required')
    rows=[];values=[];noise=[]
    for i,value in observed.items():
        if i not in range(4) or not np.isfinite(value) or value<0:raise ValueError('Invalid direct thrust observation')
        rows.append(np.eye(4)[i]);values.append(value);noise.append(measurement_sigma)
    if H is not None:
        if not constraints_calibrated:raise ValueError('Attitude alone is not a calibrated force/moment constraint')
        h=np.atleast_2d(np.asarray(H,float));y=np.atleast_1d(moment_or_total).astype(float)
        s=np.broadcast_to(np.asarray(constraint_sigma,float),y.shape)
        if h.shape!=(len(y),4) or not np.isfinite(h).all() or not np.isfinite(y).all() or not np.isfinite(s).all() or np.any(s<=0):raise ValueError('Calibrated equations and positive noise required')
        rows.extend(h);values.extend(y);noise.extend(s)
    A=np.asarray(rows,float).reshape(-1,4);y=np.asarray(values);noise=np.asarray(noise)
    rank=int(np.linalg.matrix_rank(A)) if len(A) else 0
    design=np.vstack([np.diag(1/sigma),A/noise[:,None]])
    target=np.r_[mean/sigma,y/noise]
    solution=lsq_linear(design,target,bounds=(0.,np.inf),method='bvls')
    if not solution.success:raise RuntimeError(solution.message)
    covariance=np.linalg.inv(design.T@design)
    return dict(thrusts_N=solution.x.tolist(),data_rank=rank,data_nullity=4-rank,
        uniquely_observed=rank==4,prior_dependent=rank<4,
        sum_N=float(solution.x.sum()),sum_sigma_proxy_N=float(np.sqrt(covariance.sum())),
        uncertainty_calibrated=False,hidden_thrust_is_measurement=False,
        observations=sorted(observed),calibrated_constraints_used=H is not None)
