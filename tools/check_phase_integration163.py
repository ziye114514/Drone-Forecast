"""Partial progressive phase, causal windows, and qualified input step checks."""
from pathlib import Path
import copy,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from shared_phase_fft import SharedPhaseFFT
from phase_rate_bridge import phase_supported_rates
from rotor_thrust_bank import RotorThrustBankProxy
from rotor_force_observer import RotorForceObserver
from probe_observed_force_partition import synthetic


def main(output):
    hubs=np.array([[300,230],[500,230],[500,400],[300,400]],float)
    rotors=[dict(cx=p[0],cy=p[1],major=120.,minor=40.,angle=0.,ratio=1/3,
        mode='ellipse',current_observation=True) for p in hubs]
    body=dict(detected=True,bbox=[220,190,580,440]);models=[SharedPhaseFFT(),SharedPhaseFFT()]
    dtype=[('x','u2'),('y','u2'),('p','u2'),('t','i8')];last=None
    for tick in range(1,25):
        t=tick/30;ids=[0,1] if tick<10 else [0,1,2,3]
        structure=dict(mode='partial_two' if tick<10 else 'four',model_centers=hubs.tolist(),
            observed_model_ids=ids,geometry_epoch=0)
        ts=np.arange(round((tick-1)*1e6/30),round(t*1e6),100);parts=[]
        for i in ids:
            for blade in range(2):
                for p in range(2):
                    theta=2*np.pi*50*ts/1e6+blade*np.pi+p*.04
                    e=np.zeros(len(ts),dtype=dtype);e['t']=ts;e['p']=p
                    e['x']=np.rint(hubs[i,0]+60*np.cos(theta));e['y']=np.rint(hubs[i,1]+20*np.sin(theta));parts.append(e)
        events=np.sort(np.concatenate(parts),order='t')
        future=events.copy();future['t']+=1000000
        a=models[0].step(events,t,body,structure,[rotors[i] for i in ids],[0,0])
        # Overlap consists of PAST events, not duplicates in the current packet.
        overlap=events.copy() if tick>1 else events[:0].copy();overlap['t']-=33334
        b=models[1].step(np.concatenate([overlap,events,future]),t,body,structure,[rotors[i] for i in ids],[0,0])
        if tick>1:
            assert a['rotors']==b['rotors']
        if tick==9:assert a['qualified_count']==2,a
        last=a
    assert last['qualified_count']==4 and last['calibrated_model_ids']==[0,1,2,3],last
    # One current slot safely declines a common plane instead of indexing a
    # nonexistent second distance or inventing a hidden measurement.
    z=models[0].step(events,.834,body,dict(structure,mode='partial_two',observed_model_ids=[0]),rotors[:1],[0,0])
    assert z['qualified_count']==0
    # The bridge selects by CURRENT spatial phase, not a past-frequency latch.
    s,r,rates,pool=synthetic([0,1,2,3]);bank=RotorThrustBankProxy();observer=RotorForceObserver()
    for j in range(9):
        newf=300. if j<8 else 600.
        rates=[dict(valid=True,frequency_hz=newf,spectral_candidates_hz=[300.,600.]) for _ in r]
        phase=dict(rotors=[dict(model_id=i,qualified=True,current_slot_observed=True,frequency_hz=newf) for i in range(4)])
        selected,detail=phase_supported_rates(s,r,rates,phase)
        u,q,info=bank.step(j/30,s,r,selected,pool)
        c,_,force,_=observer.step(j/30,dict(detected=True,center=[0.,0.]),u,q,
            allow_gain_learning=False,input_meta=info)
    np.testing.assert_allclose(c['acceleration'],[0.,-600.],atol=1e-7)
    assert not force['visibility_partition_changed']
    result=dict(synthetic_only=True,partial_two_rotation_qualified=True,progressive_two_to_four_qualified=True,
        future_and_overlapping_past_do_not_change_phase=True,single_current_slot_safely_declined=True,
        current_phase_can_confirm_harmonic_sized_input_step=True,
        last_step_acceleration_px_s2=c['acceleration'],
        conditional_bridge_check_is_not_end_to_end_phase_step_latency=True)
    Path(output).write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
