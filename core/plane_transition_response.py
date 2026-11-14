"""Experimental headless plane-rate conditioning of FUTURE unknown rate.

The two weights are native-teacher residual statistics, not hidden motor
thrust estimates. Present p/v/a, measured inputs and validity stay unchanged.
"""
import copy
import numpy as np
from adaptive_force_memory import kernel
from separated_force_response import rate_kernel
from axis_amplitude_bridge import current_axis_qualified
from conditional_unknown_force import HORIZON, TAU


def basis(row):
    trend=(row.get('normal_turn_forecast') or {}).get('trend') or {}
    c=row.get('coefficient');n=np.asarray(row['structure'].get('axis_xy') or [],float)
    if c is None or not trend.get('qualified') or not current_axis_qualified(row['structure']) or n.shape!=(2,) or np.linalg.norm(n)<1e-8:
        return np.zeros((2,2))
    omega=float(trend['shrunk_angular_rate_rad_s'])
    P=np.outer(n,n)/float(n@n);J=np.array([[0.,-1.],[1.,0.]])
    derivative=omega*(J@P-P@J)
    return np.column_stack((HORIZON**2*derivative@np.asarray(c['velocity']),
        HORIZON*kernel(-1/TAU,HORIZON)*derivative@np.asarray(c['acceleration'])))


def apply(row,weights):
    r=copy.deepcopy(row)
    if not r.get('conditional_unknown_force'):return r
    delta=basis(r)@np.asarray(weights,float)
    info=r['conditional_unknown_force'];rate=np.asarray(info['future_unknown_rate_px_s3'])
    info['future_unknown_rate_px_s3']=(rate+delta/rate_kernel(-1/TAU,0.,HORIZON)).tolist()
    r['plane_transition_response']=dict(weights=list(weights),additional_displacement_at04_px=delta.tolist(),
        future_unknown_response_prior=True,hidden_thrust_measured=False,current_state_and_force_unchanged=True)
    from physical_publication import evaluate
    r['physical']=evaluate(r,[.1,.2,.4]);return r
