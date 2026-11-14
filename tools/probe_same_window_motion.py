"""Frozen body/geometry/period observations; causal motion-consistency change.

No RAW timing is inferred here. Annotations are opened after all predictions
are saved, and the legacy replay must first reproduce its original input run.
"""
from pathlib import Path
import os,sys,json,copy,time,collections
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from run_fast_structure import PhysicalPositionState
from live_structure_bridge import QualifiedCenter
from shared_image_motion import SharedImageMotion
from projected_unknown_force import ProjectedResidualForce
from normal_turn_forecast import NormalTurnForecast
from compare_final_pipeline import score

def replay(rows,enabled,curved=False,direct=False,gauge=False,curved_force=False,evidence_gain=False,memory=False,excitation_limit=False,endpoint=False,memory_gaps=False,weak_support=False,visible_support=False,shorter_memory=False,tangent_response=False,spatial_weak=False,commit_motion_velocity=False,calibrated_velocity_commit=False,position_history_rebase=False,contained_reanchor_motion=False,contiguous_motion_history=False,component_acceleration=False):
    state=PhysicalPositionState();center=QualifiedCenter();motion=SharedImageMotion(same_window_motion=enabled,curved_motion=curved,anchor_gauge=gauge,weak_motion_support=weak_support,visible_motion_support=visible_support,spatial_weak_motion=spatial_weak,position_history_rebase=position_history_rebase,contained_reanchor_motion=contained_reanchor_motion,contiguous_motion_history=contiguous_motion_history,component_acceleration=component_acceleration)
    force_class=ProjectedResidualForce
    if endpoint:
        from endpoint_force_balance import EndpointForceBalance
        force_class=EndpointForceBalance
    force=force_class(gain_signal_to_noise=2.);normal=NormalTurnForecast();out=[];cost=[]
    force.direct_balance=direct
    force.curved_balance=curved_force
    force.evidence_gain=evidence_gain
    force.excitation_gain_limit=excitation_limit
    from adaptive_force_memory import AdaptiveForceMemory
    memory_model=AdaptiveForceMemory(extend_calibrated=memory_gaps,allow_shorter=shorter_memory) if memory else None
    tangent_model=None
    if tangent_response:
        from tangential_force_response import TangentialForceResponse
        tangent_model=TangentialForceResponse()
    for r in rows:
        start=time.perf_counter();s=r['structure'];adjusted,d=center.step(r['body'],s,r['geometry'].get('rotors',[]))
        state.kf.noise_scale=d.get('noise_scale',1.)
        c,valid,md=state.step(r['t'],adjusted)
        c,m=motion.step(r['t'],c,state.kf.P[2:4,2:4] if c else np.eye(2),adjusted,r['tracking'],structure=s)
        commit_info=None
        if commit_motion_velocity:
            from shared_velocity_state import commit_velocity
            commit_info=commit_velocity(state.kf,c,m)
        force.input_learning_current=(r.get('axis_amplitude_bridge') or {}).get('learning_input_current',True)
        force.observe_flow(m['observation'],m['acceleration'].get('applied',False) and m['acceleration'].get('force_balance_vector_qualified',True));force.observe_axis_motion(s,m)
        p,f=force.step(r['t'],c,r.get('force_observed_position'),r['force']['input_u'],.35)
        if calibrated_velocity_commit and f.get('gain_ever_learned'):
            from axis_amplitude_bridge import current_axis_qualified
            if current_axis_qualified(s) and (f.get('active') or f.get('short_gap_acceleration_held')):
                from shared_velocity_state import commit_velocity
                commit_info=commit_velocity(state.kf,c,m)
        trend=normal.observe(r['t'],s,c,m);p,n=normal.correct(p,f,trend)
        memory_info=None
        if memory_model is not None:
            from axis_amplitude_bridge import current_axis_qualified
            p,memory_info=memory_model.step(r['t'],c,r.get('force_observed_position'),p,f,n,valid,
                m.get('acceleration',{}).get('applied',False),current_axis_qualified(s))
        tangent_info=None
        if tangent_model is not None:p,tangent_info=tangent_model.step(r['t'],c,p,f,s,m,n,memory_info)
        cost.append((time.perf_counter()-start)*1000)
        out.append(dict(r,coefficient=c,physical=p,force=f,image_motion=m,normal_turn_forecast=n,adaptive_force_memory=memory_info,tangential_force_response=tangent_info,shared_velocity_commit=commit_info,valid=valid))
    return out,cost

def controlled():
    from live_flow_acceleration import FlowAcceleration
    def sample(bad_position=False):
        old=FlowAcceleration('state');new=FlowAcceleration('same-window');last=None
        a=np.array([1200.,-600.]);v0=np.array([300.,30.]);dt=1/30
        for i in range(13):
            t=i*dt;p=np.array([400.,300.])+v0*t+.5*a*t*t
            if bad_position:p=np.array([400.,300.])+v0*t
            b=dict(detected=True,bbox=[p[0]-30,p[1]-20,p[0]+30,p[1]+20],center=p.tolist())
            c=dict(position=p.tolist(),velocity=(v0+a*t+np.array([150.,-100.])).tolist(),acceleration=[0.,0.])
            flow=dict(valid=i>0,midpoint_t=t-dt/2,velocity=(v0+a*(t-dt/2)).tolist(),velocity_covariance=np.eye(2).tolist())
            _,o=old.step(t,c,b,flow);_,n=new.step(t,c,b,flow);last=(o,n)
        return last
    o,n=sample();assert not o['applied'] and n['applied'],(o,n)
    _,bad=sample(True);assert not bad['applied'],bad
    return dict(lagged_state_velocity_does_not_reject_consistent_curve=True,
        inconsistent_body_curve_still_rejected=True,old_residual_gates_unchanged=True)

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);checks=controlled();base=ROOT.parent/'output'
    specs=[('side','disk_motion_30px_20261028/RAW_side_normal8s','dense_labels_side_110_1195'),
        ('train2','disk_motion_30px_20261028/RAW_train2_temporal6s_final','dense_labels_train2_40_46')]
    result=dict(checks=checks,clips={},RAW_FPS_measured=False,parameters_swept=False)
    for name,folder,label in specs:
        rows=[json.loads(s) for s in (base/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        variants={};costs={}
        for k,enabled,curved,direct,gauge in [('legacy',False,False,False,False),('anchor_gauge',False,False,False,True)]:
            v,cost=replay(rows,enabled,curved,direct,gauge);variants[k]=v;costs[k]=float(np.mean(cost))
            (out/(name+'_'+k+'.jsonl')).write_text('\n'.join(json.dumps(x) for x in v)+'\n','utf-8')
        delta=max(float(np.max(abs(np.asarray(a['physical'][h])-np.asarray(b['physical'][h]))))
            for a,b in zip(rows,variants['legacy']) for h in a['physical'])
        assert delta<1e-7,('Legacy replay mismatch',name,delta)
        truth=sorted(json.loads((base/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        result['clips'][name]=dict(legacy_max_prediction_difference=delta,
            scores={k:score(v,truth)['physical_future_error_px'] for k,v in variants.items()},
            qualified_acceleration={k:sum(r['image_motion']['acceleration'].get('applied',False) for r in v) for k,v in variants.items()},
            acceleration_reasons={k:dict(collections.Counter(r['image_motion']['acceleration'].get('reason') for r in v)) for k,v in variants.items()},
            motion_force_CPU_mean_ms=costs,same_source_and_prediction_slots=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
