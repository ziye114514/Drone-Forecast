"""Full-observer invariants for rotor-only innovation, not just matrix tests."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_gain_mixture import RotorGainMixture
from observable_force_gain import ObservableSeparatedForce


def packet(i,change_frequency=False,change_instance=False):
    t=i/30;angle=.15*np.sin(i/8);n=np.array([np.sin(angle),-np.cos(angle)])*.8
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    points=np.array([[-20.,-20.],[20.,-20.],[20.,20.],[-20.,20.]])@R.T
    scale=1+.08*np.sin(i/7);points*=scale
    st=dict(mode='four',continuous=dict(updated=True),geometry_epoch=0,axis_xy=n.tolist(),
        model_centers=points.tolist(),current_geometric_center=[0.,0.])
    row=dict(t=t,structure=st,body=dict(detected=True,center=[0.,0.]),motion={},coefficient=None,
        force=dict(effective_gain_px_s2=200.),normal_turn_forecast=None)
    records=[]
    for j,p in enumerate(points):
        freq=250. if change_frequency and j==0 else 200.
        g=dict(cx=float(p[0]),cy=float(p[1]),major=20.*scale,minor=10.*scale,angle=np.rad2deg(angle))
        records.append(dict(track_id=f'slot:{j}',phase_instance_id=f'instance:{j}:{int(change_instance)}',
            qualified=True,current_force_admitted=True,proxy_twofold_hz=freq,
            result=dict(arms=dict(joint_pose=dict(inferred_geometry=g)))))
    return row,dict(t=t,meta=dict(records=records))


if __name__=='__main__':
    OUT=ROOT.parent/'output/invariants309_20261109';OUT.mkdir(exist_ok=False)
    model=RotorGainMixture();maximum=0.;prior_weight=None
    for i in range(90):
        row,period=packet(i);r=model.step(row,period)
        weights=np.array(r['rotor_mechanical_gain_mixture']['weights'])
        if prior_weight is None:prior_weight=weights
        assert np.max(abs(weights-prior_weight))<1e-10
        for m in model.models:
            if m.x is not None:maximum=max(maximum,float(np.max(abs(m.x[:3]-m.control))))
    assert maximum<1e-10,maximum
    row,period=packet(90,change_frequency=True);r=model.step(row,period)
    impulse=max(abs(v['acceleration']) for v in r['rotor_mechanical_gain_mixture']['member_angular_corrections']);assert impulse>0
    other=RotorGainMixture()
    for i in range(4):
        row,period=packet(i);other.step(row,period)
    row,period=packet(4,change_frequency=True,change_instance=True);r=other.step(row,period)
    false_impulse=max(abs(v['acceleration']) for v in r['rotor_mechanical_gain_mixture']['member_angular_corrections']);assert false_impulse<1e-10
    arms=[ObservableSeparatedForce(default_gain=200.),ObservableSeparatedForce(default_gain=200./7)]
    position=np.zeros(2);speed=np.zeros(2);delta=0.;gate_credit=0.
    for i in range(100):
        t=i/30;n=np.array([.3*np.sin(t*2),-1.]);n/=np.linalg.norm(n)
        amplitude=.25*(1+.5*np.sin(4*t));u=amplitude*n
        acceleration=1000*u+np.array([30.,250.]);position+=speed/30+.5*acceleration/30**2;speed+=acceleration/30
        body=dict(detected=True,center=position.tolist(),bbox=(np.r_[position-10,position+10]).tolist())
        structure=dict(mode='four',axis_xy=n.tolist(),continuous=dict(updated=True))
        outputs=[]
        for scale,m in zip((1.,7.),arms):
            meta=dict(observed_components={'synthetic:observed':(u*scale).tolist()},input_sigma_proxy=.003*scale,learning_input_current=True)
            c,v,f,s=m.step(t,body,u*scale,.7,allow_gain_learning=True,input_meta=meta,structure=structure)
            outputs.append(c);gate_credit+=bool(f.get('input_gain_qualification',{}).get('qualified'))
        for key in ('position','velocity','acceleration'):delta=max(delta,float(np.max(abs(np.array(outputs[0][key])-outputs[1][key]))))
    assert delta<1e-7,(delta,gate_credit);assert gate_credit>0
    result=dict(stationary_body_wobbling_plane_fixed_frequencies_max_observer_increment=maximum,
        fixed_frequency_does_not_learn_gain_from_geometric_jitter=True,
        real_frequency_step_changes_future_angular_acceleration=impulse,
        phase_instance_relock_frequency_jump_has_no_false_impulse=false_impulse,
        complete_force_observer_input_unit_scaling_max_current_state_difference=delta,
        gain_gate_was_active_in_unit_invariance_test=gate_credit,
        both_null_and_active_angular_observers_receive_same_input_uncertainty=True,
        real_RPM_CT_or_inertia_calibration_not_claimed=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),'utf8');print(result,flush=True)
