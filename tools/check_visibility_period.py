"""Mechanism controls: unequal static rotors, common scaling, association, expiry."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from visibility_period import VisibilityPeriodProxy,LegacyPeriodProxy,ConservativeSlotPeriodProxy


def observation(ids,frequencies,mode=None):
    hubs=np.array([[0.,0.],[60.,0.],[60.,40.],[0.,40.]])
    s=dict(mode=mode or ('four' if len(ids)==4 else 'three' if len(ids)==3 else 'partial_two'),
        model_centers=hubs.tolist(),observed_model_ids=ids,axis_xy=[0.,-1.],geometric_weight=.8,geometry_epoch=0)
    rotors=[dict(cx=hubs[i,0],cy=hubs[i,1],major=24.,ratio=.5,mode='ellipse',current_observation=True) for i in ids]
    rates=[dict(valid=True,frequency_hz=float(frequencies[i])) for i in ids]
    return s,rotors,rates,dict(qualified=True,frequency_hz=210.)


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    old,new=LegacyPeriodProxy(),VisibilityPeriodProxy();conservative=ConservativeSlotPeriodProxy();f=np.array([160.,200.,260.,320.]);rows=[]
    sequence=[list(range(4))]*4+[[0,1,2]]*3+[[1,2]]*3+[[3,0]]*3+[list(range(4))]*3
    for k,ids in enumerate(sequence):
        data=observation(ids,f);_,_,a=old.step(k/30,*data);_,_,b=new.step(k/30,*data)
        _,_,c=conservative.step(k/30,*data)
        rows.append(dict(t=k/30,ids=ids,old=a['relative_f2'],new=b['relative_f2'],slots_only=c['relative_f2']))
    assert all(abs(r['new']-1.)<1e-12 for r in rows[3:])
    assert all(r['slots_only'] is None or abs(r['slots_only']-1.)<1e-12 for r in rows[3:])
    false_old=max(abs(r['old']-1.) for r in rows[3:] if r['old'] is not None)
    for k in range(16,20):
        _,_,b=new.step(k/30,*observation([1,2],f*1.02))
    assert abs(b['relative_f2']-1.02**2)<1e-12
    # Reordering regions is not a motor swap.
    data=observation([2,1],f*1.02);_,_,permuted=new.step(20/30,*data)
    assert abs(permuted['relative_f2']-1.02**2)<1e-12
    # A hidden motor changing alone cannot be inferred from the visible ones.
    hidden=f*1.02;hidden[3]*=1.2
    _,_,hidden_result=new.step(21/30,*observation([1,2],hidden))
    assert hidden_result['hidden_frequencies_measured']==0 and hidden_result['partial_sum_requires_common_scale']
    # Axis-only view: arbitrary unlabelled region periods never enter the input.
    data=observation([0,1],f,'axis_only');data[0].update(observed_model_ids=[],model_centers=None)
    u,q,axis=new.step(22/30,*data)
    assert axis['source']=='pooled_axis_separate_gauge' and not axis['current_model_ids']
    missing=(dict(mode='axis_only',geometry_epoch=0,axis_xy=[0.,-1.]),[],[],dict(qualified=False))
    for k in range(23,30):
        u,q,empty=new.step(k/30,*missing);assert u is None
    data=observation([0,1,2],f);data[0]['geometry_epoch']=1
    u,q,reset=new.step(1.,*data);assert not reset['calibrated_model_slots']
    result=dict(synthetic_only=True,unequal_static_rotors_old_false_relative_jump=false_old,
        new_false_relative_jump=0.,common_frequency_scale_squared=1.02**2,
        region_reordering_preserved=True,hidden_motor_change_not_observed=True,
        side_uses_pool_not_blob_count=True,geometry_epoch_reset=True,missing_does_not_create_input=True,
        slot_relock_may_require_unmeasured_warmup_not_fake_period=True,
        Newton_force_or_real_RPM_truth=False)
    (out/'controls.json').write_text(json.dumps(result,indent=2),'utf-8')
    (out/'static_visibility_rows.json').write_text(json.dumps(rows,indent=2),'utf-8')
    print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1])
