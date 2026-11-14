"""Guard independent validation across short packets, gaps and duplicates."""
from pathlib import Path
import sys,json,copy,numpy as np
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from joint_phase_bank import JointPhaseBank
from joint_force_adapter import adapt
from rotor_cuda_worker import DTYPE

if __name__=='__main__':
    OUT=ROOT.parent/'output/guards245_20261108';OUT.mkdir(exist_ok=False)
    ellipse=dict(cx=300.,cy=300.,major=80.,minor=50.,ratio=.625,angle=0.,mode='ellipse',center_sigma_px=1.,axis_angle_sigma_deg=2.)
    valid=[True];sources=[]
    def fake(events,r,t,velocity=None,**kwargs):
        mid=(round(t*1e6)-33333+33333//2)/1e6
        assert kwargs['geometry_source_end_s']<=mid+1e-9
        assert kwargs['velocity_source_end_s'] is None or kwargs['velocity_source_end_s']<=mid+1e-9
        sources.append((t,kwargs['geometry_source_end_s']))
        g=dict(r,fov_clipped=False)
        return dict(control=dict(angular_harmonic=2),arms={'joint_pose':dict(qualified=valid[0],inferred_geometry=g,
            discovery_state_time_s=mid,author_state=dict(angular_velocity_rad_s=2*np.pi*600,alpha=0.))})
    e=np.empty(0,DTYPE);motion=dict(dt_s=1/30,current_observation=True,estimate=dict(dx=0.,dy=0.,score=1.))
    with patch('joint_phase_bank.observe_ring',fake):
        m=JointPhaseBank('not_loaded.dll');results=[]
        for t in [10.033333,10.066666,10.066766,10.099999,10.100099]:results.append(m.step(t,e,[ellipse],motion))
        assert results[1][0] is None and results[2][0] is None
        assert results[3][0] is not None and results[4][0] is not None
        assert results[3][2]['learning_input_current'] and not results[4][2]['learning_input_current']
        valid[0]=False;assert m.step(10.133332,e,[ellipse],motion)[0] is None
        valid[0]=True;assert m.step(10.166665,e,[ellipse],motion)[0] is None
        assert m.step(10.199998,e,[ellipse],motion)[0] is not None
        assert m.step(11.,e,[ellipse],motion)[0] is None
        m=JointPhaseBank('not_loaded.dll');rotors=[ellipse,ellipse,dict(ellipse,cx=450.)]
        for t in [1.,1.033333,1.066666]:last=m.step(t,e,rotors,motion)
        assert last[2]['qualified_current_rings']==1 and len(last[2]['ambiguous_candidates'])==2
    original=dict(structure=dict(axis_xy=[0.,-1.],mode='edge_on'),period_selection={'observed_components':{}},force={'input_u':None})
    u,meta,g=adapt(original,dict(u=None,meta=dict(records=[],ambiguous_candidates=[])),refine_normal=True)
    assert u is None and g==original['structure'] and original['structure']['axis_xy']==[0.,-1.]
    checks=dict(short_packet_does_not_leak_late_geometry_or_double_confirm=True,
        stale_frequency_not_measured_on_lost_frame=True,relock_requires_new_confirmations=True,
        duplicate_pair_does_not_discard_independent_third_ring=True,
        empty_side_branch_unchanged=True,hidden_frequencies_measured=0,NN_used=False,GPU_used=False,
        initializations_checked=len(sources),last_guard_only_affects_partial_final_packet_after_scored_origins=True)
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2),'utf-8');print(json.dumps(checks),flush=True)
