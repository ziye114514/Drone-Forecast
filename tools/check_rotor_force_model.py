"""Counterexamples for physical allocation, hidden-state expiry and bridge causality."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from propeller_physics import thrust_newtons,parallel_allocation,wrench,infer_missing_thrust,static_coefficient_curve
from rotor_thrust_bank import RotorThrustBankProxy
from axis_amplitude_bridge import AxisAmplitudeBridge


def inputs(ids,epoch=0):
    centers=np.array([[0.,0.],[100.,0.],[100.,100.],[0.,100.]])
    frequencies=[200.,220.,240.,260.]
    structure=dict(mode={4:'four',3:'three',2:'partial_two'}.get(len(ids),'axis_only'),
        model_centers=centers.tolist(),observed_model_ids=ids,geometry_epoch=epoch,
        axis_xy=[0.,-1.],geometric_weight=.8,pose_source='current_qualified_visible_conics')
    rotors=[dict(cx=centers[i,0],cy=centers[i,1],major=60.,ratio=.4,mode='ellipse',current_observation=True) for i in ids]
    rates=[dict(valid=True,frequency_hz=frequencies[i]) for i in ids]
    return structure,rotors,rates,dict(qualified=False)


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    assert np.isclose(thrust_newtons(200.,.127,.1),1.225*200**2*.127**4*.1)
    assert np.isclose(thrust_newtons(400.,.127,.1)/thrust_newtons(200.,.127,.1),4)
    assert np.isclose(thrust_newtons(200.,.254,.1)/thrust_newtons(200.,.127,.1),16)
    curve=static_coefficient_curve('RPM CT CP\n6000 .08 .03\n12000 .1 .04')
    assert np.isclose(curve(150.),.09)
    try:curve(300.);raise AssertionError('extrapolation accepted')
    except ValueError:pass
    r=np.array([[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]])
    T=np.array([1.,2.,3.,4.]);A=parallel_allocation(r)
    force,moment=wrench(r,T);assert np.allclose(np.r_[force[2],moment[:2]],A@T)
    none=infer_missing_thrust({},np.ones(4),np.ones(4))
    assert none['data_nullity']==4 and none['prior_dependent']
    three=infer_missing_thrust({0:T[0],1:T[1],2:T[2]},np.ones(4),np.ones(4))
    assert three['data_nullity']==1 and not three['uniquely_observed']
    constrained=infer_missing_thrust({0:1.,1:2.,2:3.},np.ones(4),np.ones(4)*100,
        A[1:2],(A@T)[1:2],1e-5,True,measurement_sigma=1e-5)
    assert constrained['data_rank']==4 and np.allclose(constrained['thrusts_N'],T,atol=1e-6)
    diagonal=infer_missing_thrust({0:1.,2:3.},np.ones(4),np.ones(4),A[1:],(A@T)[1:],.01,True)
    adjacent=infer_missing_thrust({0:1.,1:2.},np.ones(4),np.ones(4)*100,A[1:],(A@T)[1:],1e-5,True,measurement_sigma=1e-5)
    assert diagonal['data_rank']==3 and adjacent['data_rank']==4
    assert np.allclose(adjacent['thrusts_N'],T,atol=1e-6)
    try:infer_missing_thrust({0:1.},np.ones(4),np.ones(4),A[1:],np.zeros(2),.1);raise AssertionError('uncalibrated constraints accepted')
    except ValueError:pass
    bank=RotorThrustBankProxy();bridge=AxisAmplitudeBridge()
    records=[]
    for t,ids in [(0.,[0,1,2,3]),(.033333,[0,1,2,3]),(.066666,[0,1,2]),(.099999,[0,1]),(.166666,[0,1])]:
        s,rotors,rates,pool=inputs(ids)
        u,q,d=bank.step(t,s,rotors,rates,pool);u,q,b=bridge.step(t,s,u,q,d)
        records.append(dict(t=t,visible=ids,period=d,bridge=b,u=None if u is None else u.tolist()))
    assert np.isclose(records[1]['period']['relative_f2'],1.)
    for row in records[2:4]:
        assert row['period']['rotor_bank_used'] and np.isclose(row['period']['relative_f2'],1.)
        assert row['period']['learning_input_current']==False and row['bridge']['learning_input_current']==False
    assert records[-1]['u'] is None and bridge.last[0]==.033333
    assert len(records[3]['period']['hidden_model_ids'])==2 and records[3]['period']['hidden_frequencies_measured']==0
    old_reference=bank.legacy.reference
    u,_,d=bank.step(.2,*inputs([0,1],epoch=1));assert u is None
    assert bank.legacy.reference==old_reference
    result=dict(synthetic_only=True,independent_real_thrust_truth=False,n_squared_and_D_fourth_scaling=True,
        unequal_rotor_forward_wrench=True,instantaneous_attitude_is_not_force_constraint=True,
        three_without_torque_rank=three['data_rank'],three_with_calibrated_torque_rank=constrained['data_rank'],
        two_diagonal_with_roll_pitch_rank=diagonal['data_rank'],two_adjacent_with_roll_pitch_rank=adjacent['data_rank'],
        hidden_states_not_equal_visible_mean=True,hidden_expiry_not_self_refreshed=True,
        hidden_prior_not_gain_learning_evidence=True,epoch_reset_rejects_old_ids=True,
        no_CT_extrapolation=True,slot_reset_preserves_whole_body_frequency_gauge=True,records=records)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':main(sys.argv[1])
