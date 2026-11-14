"""Bounded causal force/state/side-qualification checks; no dataset labels."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from responsive_force import ResponsiveForce,DampedResponsiveForce,WindowResidualForce
from force_position_state import ForceDrivenNoiseKF
from live_structure_bridge import NoiseKF
from fast_structure_front import FastStructure


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    truth_acc=np.array([80.,-30.]);u=np.array([0.,-1.]);errors={}
    for name,kind in [('recent',ResponsiveForce),('window',WindowResidualForce)]:
        model=kind()
        for i in range(61):
            t=i/30;p=.5*t*t*truth_acc;v=t*truth_acc
            c=dict(position=p.tolist(),velocity=v.tolist(),acceleration=[0.,0.])
            pred,diag=model.step(t,c,p,u,.8)
        errors[name]=float(np.linalg.norm(np.asarray(diag['acceleration_px_s2'])-truth_acc))
        assert errors[name]<1.,errors
        assert not diag['gain_ever_learned'],'Constant input cannot identify a physical gain'
    a,b=NoiseKF(),ForceDrivenNoiseKF();delta=0.
    for i in range(30):
        p=[i*.8,i*.5];left=a.step(i/30,p);right=b.step(i/30,p)
        delta=max(delta,float(np.max(np.abs(np.asarray(left['velocity'])-right['velocity']))))
    assert delta<1e-10
    controlled=ForceDrivenNoiseKF();controlled.step(0.,[1.,2.]);controlled.set_control([30.,-20.]);c=controlled.step(.05,None)
    np.testing.assert_allclose(c['position'],np.array([1,2])+.5*.05**2*np.array([30,-20]),atol=1e-12)
    np.testing.assert_allclose(c['velocity'],[1.5,-1.],atol=1e-12)
    native=np.load(ROOT.parent/'output/predictive_harmonics_20261027/native_pilot/side/native_window.npz')
    t=float(native['t']);e=native['events']
    rows=[json.loads(s) for s in (ROOT.parent/'output/three_continuation_20261026/raw_release_side/predictions.jsonl').read_text('utf-8').splitlines()]
    row=min(rows,key=lambda r:abs(r['t']-t));model=FastStructure(optimized_compute=True,side_axis_budget=True)
    model.tick=1;model.previous_t=t-1/30;model.last_side_qualified=True
    s,rates,diag,_=model.step(e,t,row['body'],None,[0,0],defer_rates=True)
    assert diag['current_axis_budget_tick']
    assert diag['side']['valid'] and not diag['rotors'] and s['mode']=='axis_only'
    assert s['current_geometric_center'] is None and not s['joint_center_is_current_measurement']
    result=dict(controlled_constant_acceleration_errors_px_s2=errors,
        unexcited_gain_unidentified=True,zero_control_state_max_velocity_difference=delta,
        deterministic_force_state_transition_correct=True,current_side_axis_qualified=True,
        no_invented_rotors_or_joint_center=True,real_trajectory_accuracy_test=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1])
