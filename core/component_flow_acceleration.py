"""Keep a supported image component when the other fails position agreement.

The complete flow-fit gates stay intact. Only the existing position-agreement
test is applied separately to x/y, with unchanged thresholds and history.
Partial agreement never certifies a complete force-balance observation.
"""
import copy
import numpy as np
from curved_flow_motion import CurvedFlowAcceleration


class ComponentFlowAcceleration(CurvedFlowAcceleration):
    def step(self,t,coefficient,body,flow,policy='position-consistent'):
        value,info=super().step(t,coefficient,body,flow,policy)
        if coefficient is None or info.get('applied') or len(self.positions)<6:
            return value,info
        candidate=info
        curve=info.get('curved_motion') or {}
        if curve.get('reason')=='curve_disagrees_with_position_history':
            candidate=curve
        elif info.get('reason')!='flow_derivative_disagrees_with_position_history':
            return value,info
        times=np.array([row[0]-t for row in self.positions])
        if np.ptp(times)<.16-1e-8:return value,info
        p=np.array([row[1] for row in self.positions])
        a=np.asarray(candidate['shrunk_acceleration'])
        curved=candidate is curve
        v=np.asarray(candidate['fitted_current_velocity'] if curved or self.position_velocity=='same-window' else coefficient['velocity'])
        dynamics=times[:,None]*v+.5*times[:,None]**2*a
        if curved:dynamics+=times[:,None]**3*np.asarray(candidate['jerk_px_s3'])/6.
        residual=p-np.median(p-dynamics,axis=0)-dynamics
        med=np.median(abs(residual),axis=0);fraction=np.mean(abs(residual)<=8.07,axis=0)
        selected=(med<=6.)&(fraction>=.9)&(abs(a)>1e-8)
        diagnostic=dict(component_qualified=selected.tolist(),position_median_residual_px=med.tolist(),
            position_inlier_fraction=fraction.tolist(),original_joint_reason=info.get('reason'),
            force_balance_vector_qualified=False,extra_curve_fits=0)
        if not selected.any():return value,dict(info,component_recovery=diagnostic)
        revised=copy.deepcopy(coefficient)
        revised['acceleration']=np.where(selected,a,np.asarray(coefficient['acceleration'])).tolist()
        return revised,dict(candidate,applied=True,reason='position_qualified_acceleration_components',
            component_qualified=selected.tolist(),force_balance_vector_qualified=False,
            component_recovery=diagnostic,curved_motion=curve,linear_control=info)
