"""Known causal image kinematics and a local-motion inconsistency case."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_flow_acceleration import FlowAcceleration,DESIGN
OUT=ROOT.parent/'output/flow_acceleration_20261026';OUT.mkdir(exist_ok=False)
def sequence(acceleration,stationary_positions=False,policy='position-consistent'):
    observer=FlowAcceleration();a=np.asarray(acceleration,float);last=None
    for t in np.arange(1.,1.361,.04):
        d=t-1.;p=np.array([600.,350.])+np.array([200.,60.])*d+.5*a*d*d;v=np.array([200.,60.])+a*d
        if stationary_positions:p=np.array([600.,350.])
        body=dict(detected=True,confidence=.9,center=p.tolist(),bbox=(np.r_[p-[50,25],p+[50,25]]).tolist())
        c=dict(position=p.tolist(),velocity=v.tolist(),acceleration=[0.,0.],conditional_scale_rate=None)
        flow=dict(valid=True,velocity=(v-.02*a).tolist(),velocity_covariance=(np.eye(2)*400).tolist(),midpoint_t=t-.02)
        last,info=observer.step(float(t),c,body,flow,policy)
        assert last['position']==c['position'] and last['velocity']==c['velocity'] and c['acceleration']==[0.,0.]
    return last,info
c,info=sequence([1200.,-800.]);assert info['applied'],info
error=float(np.linalg.norm(np.asarray(c['acceleration'])-[1200.,-800.]));assert error<75.,error
constant,ci=sequence([0.,0.]);assert not ci['applied'] and constant['acceleration']==[0.,0.]
local,li=sequence([1200.,-800.],True);assert not li['applied'] and li['reason']=='flow_derivative_disagrees_with_position_history',li
unqualified,ui=sequence([1200.,-800.],True,'velocity-only');assert ui['applied']
checks=dict(known_acceleration_error_px_s2=error,constant_velocity_no_fake_acceleration=True,
    local_motion_rejected_by_position_consistency=True,position_and_velocity_unchanged=True)
(OUT/'mechanism_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(OUT/'design.json').write_text(json.dumps(dict(**DESIGN,
    references=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html'],
    related_polynomial_derivative='https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html',
    method='Existing project Huber conventions adapted to irregular causal velocity derivatives; no centered/future filtering.',
    parent='accumulated_v5_fast_ci',cache_only_initial_experiment=True,globally_blind=False),indent=2),encoding='utf-8')
print(json.dumps(checks),flush=True)
