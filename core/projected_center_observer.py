"""A projected rotor-layout centre, not an arbitrary patch, owns motion state.

Current qualified4/3 geometry has priority; current complete-FOV thin-line
midpoint is the degraded visible-centre observation. Neither is COM truth.
Current native displacement and centre share events, hence fixed symmetric
information weighting. First reference acquisition is a common coordinate
rebase, NOT a new velocity/thrust impulse. No future geometry is supplied.
"""
import copy
import numpy as np
from relative_rotor_observer import RelativeRotorForceObserver
from adaptive_response_observer import AdaptiveResponseObserver
from separated_force_response import SeparatedForceObserver


def center_packet(row):
    st=row['structure'];g=row['geometry'];body=row['body']
    if not body.get('detected') or body.get('center_provisional'):return None
    packet=row['motion'].get('layout_observation')
    if packet and st.get('mode') in ('four','three') and st.get('joint_center_is_current_measurement'):
        return dict(packet,kind='current_qualified_4_or_3_rotor_center')
    side=st.get('axis_support') or {};bounds=g.get('roi')
    if st.get('mode') not in ('axis_only','edge_on') or not side.get('valid') or side.get('confidence',0)<.45:return None
    if bounds is None or bounds[0]<=0 or bounds[1]<=0 or bounds[2]>=1280 or bounds[3]>=720:return None
    c=np.asarray(side.get('center'),float)
    if c.shape!=(2,) or not np.isfinite(c).all():return None
    scatter=np.asarray(side.get('center_multiscale_covariance_proxy_px2',np.zeros((2,2))),float)
    return dict(center=c.tolist(),covariance_proxy=(np.eye(2)*36+(scatter+scatter.T)/2).tolist(),
        kind='current_complete_FOV_side_projection_midpoint',current_observation=True,
        inferred_hidden_hubs=False,COM_truth=False)


class ProjectedCenterMeasurement:
    def _measurement(self,z,body,motion,lag):
        # The old layout bias b is auxiliary here, not an additional free
        # offset that can hide which object point the output represents.
        H,R,value,flow=RelativeRotorForceObserver._measurement(self,z,body,motion,lag)
        packet=(motion or {}).get('projected_center_observation')
        if packet is None:return H,R,value,flow
        Hg=np.zeros((2,self.state_dim));Hg[:,:2]=np.eye(2)
        Rg=np.asarray(packet['covariance_proxy'],float);cg=np.asarray(packet['center'],float)
        if not self.relative_used:return Hg,Rg,cg,False
        n=len(value);Q=np.zeros((n+2,n+2));Q[:n,:n]=2*R;Q[n:,n:]=2*Rg
        return np.vstack([H,Hg]),Q,np.r_[value,cg],flow


class ProjectedCenterSeparated(ProjectedCenterMeasurement,SeparatedForceObserver):pass


class ProjectedCenterObserver(ProjectedCenterMeasurement,AdaptiveResponseObserver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.alternative=ProjectedCenterSeparated(*args,**kwargs)
        self.projected_reference_initialized=False;self.current_body=None

    def step(self,t,body,u,*args,**kwargs):
        if self.seen is not None and t-self.seen>.5:
            self.__init__(self.default_gain,self.velocity_sigma,self.startup_fit,self.consistent_dynamics,self.two_point_init)
        motion=kwargs.get('motion') or {};packet=motion.get('projected_center_observation')
        body=copy.deepcopy(body);rebase=None
        if packet is not None:
            current=np.asarray(packet['center'],float)
            if not self.projected_reference_initialized:
                rebase=current-np.asarray(body['center'],float)
                old=np.asarray(body.get('measurement_reference_shift_xy',[0.,0.]),float)
                body['measurement_reference_shift_xy']=(old+rebase).tolist()
                self.projected_reference_initialized=True
            body['center']=current.tolist()
            body['projected_center_observation_kind']=packet['kind']
            body['center_is_proven_mass_center']=False
        self.current_body=body
        c,valid,f,s=super().step(t,body,u,*args,**kwargs)
        s.update(output_reference='projected_rotor_layout_center',projected_center_reference_initialized=self.projected_reference_initialized,
            current_projected_center_kind=None if packet is None else packet['kind'],
            first_center_reference_rebase=None if rebase is None else rebase.tolist(),
            current_center_and_native_displacement_share_information=True)
        return c,valid,f,s
