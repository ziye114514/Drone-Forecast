"""Check the new force mechanism, not just output formatting."""
from pathlib import Path
import sys,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from visible_force_dynamics import VisibleForceDynamics,evaluate_visible,scale_aware_gain
from flow_identifiable_gain import fit_gain


def test_row():
    return dict(t=0.,valid=True,coefficient=dict(position=[0.,0.],velocity=[0.,0.],acceleration=[0.,0.]),
        force167_mode='full_fixed',visible_force_dynamics=dict(normal=[0.,-1.],components={'a':.25},
        effective_gain=200.,log_rates_per_s={'a':0.},omega=0.))


if __name__=='__main__':
    r=test_row();assert evaluate_visible(r,[.4])['0.4']==[0.,0.]
    r['visible_force_dynamics']['omega']=.5
    value=np.array(evaluate_visible(r,[.4])['0.4']);assert value[0]>0 and abs(value[1])<abs(value[0])
    # A coordinate-only transfer between current known and unknown forces must
    # not change future rotation of the actual current observed contribution.
    a=copy.deepcopy(r);b=copy.deepcopy(r);a['visible_force_dynamics']['unknown_current']=[0,50]
    b['visible_force_dynamics']['unknown_current']=[300,-100]
    b['visible_force_dynamics']['subtracted_input_reference']=[1.5,-.75]
    assert evaluate_visible(a,[.4])==evaluate_visible(b,[.4])
    r=test_row();r['visible_force_dynamics']['log_rates_per_s']['a']=.5
    assert evaluate_visible(r,[.4])['0.4'][1]<0
    assert evaluate_visible(r,[0.])['0.0']==[0.,0.]
    r['force167_mode']='zero';assert evaluate_visible(r,[.4])['0.4']==[0.,0.]
    times=np.arange(50)/30;f=300*np.sqrt(1+.2*np.sin(9*times));u=np.column_stack((np.zeros(len(times)),-f*f/360000))
    I=np.vstack((np.zeros(2),np.cumsum(.5*(u[1:]+u[:-1])*np.diff(times)[:,None],axis=0)))
    velocity=np.array([10.,20.])+700*I+times[:,None]*np.array([40.,175.])
    model=VisibleForceDynamics(learn=True);gains=[]
    for t,F,V,U in zip(times,f,velocity,u):
        s=dict(mode='axis_only',geometry_epoch=0,axis_xy=[0.,-1.],geometric_weight=.8,
            pose_source='current_qualified_visible_conics')
        obs=dict(structure=s,pooled=dict(qualified=True,frequency_hz=float(F),selection_source='controlled'),
            current_pose=True,geometry=dict(rotors=[]),rates=[])
        base=dict(t=float(t),coefficient=dict(position=[0.,0.],velocity=V.tolist(),acceleration=(700*U+[40.,175.]).tolist()))
        motion=dict(observation=dict(valid=True,midpoint_t=float(t),velocity=V.tolist(),velocity_covariance=(np.eye(2)*.01).tolist()))
        info=model.step(base,obs,motion)
        if info['gain_updated']:gains.append(info['effective_gain'])
    assert gains and abs(gains[-1]-700)<1e-6,(gains,dict(model.reasons))
    for advance in (1,2):
        obs['pooled']['selection_source']='other_incompatible_proxy'
        base['t']=float(times[-1]+advance/30);motion['observation']['midpoint_t']=base['t']
        last=model.step(base,obs,motion)
    assert last['gain_source']=='pool:other_incompatible_proxy' and last['effective_gain']==200.
    constant=np.tile([0.,-.25],(len(times),1));v=700*times[:,None]*constant+times[:,None]*np.array([40.,175.])
    k,diagnostic=fit_gain([(float(t),V,np.eye(2)) for t,V in zip(times,v)],list(zip(times,constant)))
    assert k is None and diagnostic['reason']=='gain_indistinguishable_from_constant_disturbance'
    # Same trajectory under a different arbitrary input normalization: an
    # absolute coefficient cutoff used to reject the equivalent7000 value.
    hist=[(float(t),V,np.eye(2)*.01) for t,V in zip(times,velocity)]
    k1,_=scale_aware_gain(hist,list(zip(times,u)))
    k2,_=scale_aware_gain(hist,list(zip(times,u/10.)))
    assert k1 is not None and k2 is not None and abs(k1-700)<1e-7 and abs(k2-7000)<1e-6
    np.testing.assert_allclose(k1*u,k2*(u/10.),atol=1e-8,rtol=0)
    print('PASS: equilibrium, nonzero raw-amplitude turn, reference gauge, RPM trend, zero/h=0, identifiable k=700')
