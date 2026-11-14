"""Score frozen RAW outputs and a no-motion control on identical current inputs."""
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_fast_structure import PhysicalPositionState
from responsive_force import DampedResponsiveForce
from flow_force_balance import FlowBalancedForce
from compare_final_pipeline import score,reference_at


def main(run,label):
    path=Path(run);rows=[json.loads(s) for s in (path/'predictions.jsonl').read_text('utf-8').splitlines()]
    state=PhysicalPositionState();force=DampedResponsiveForce();motion_force=DampedResponsiveForce();balanced_force=FlowBalancedForce()
    control=[];motion_only=[];balanced_only=[]
    for r in rows:
        body=copy.deepcopy(r['body']);bridge=r['center_bridge']
        if bridge.get('applied'):body['center']=bridge['fused_position']
        state.kf.noise_scale=bridge.get('noise_scale',1.)
        c,valid,_=state.step(r['t'],body)
        assert valid==r['valid']
        if c is not None:assert np.allclose(c['position'],r['coefficient']['position'],atol=1e-8,rtol=0)
        current_input=(r.get('axis_amplitude_bridge') or {}).get('learning_input_current',True)
        force.input_learning_current=current_input;motion_force.input_learning_current=current_input;balanced_force.input_learning_current=current_input
        p,f=force.step(r['t'],c,r.get('force_observed_position'),r['force']['input_u'],.35)
        control.append(dict(r,coefficient=c,physical=p,force=f))
        p,f=motion_force.step(r['t'],r['coefficient'],r.get('force_observed_position'),r['force']['input_u'],.35)
        motion_only.append(dict(r,physical=p,force=f))
        image_motion=r.get('image_motion') or {}
        balanced_force.observe_flow(image_motion.get('observation',dict(valid=False)),image_motion.get('acceleration',{}).get('applied',False))
        p,f=balanced_force.step(r['t'],r['coefficient'],r.get('force_observed_position'),r['force']['input_u'],.35)
        balanced_only.append(dict(r,physical=p,force=f))
    # Freeze the control before opening independent manual positions.
    (path/'same_input_no_motion_control.jsonl').write_text('\n'.join(json.dumps(r) for r in control)+'\n','utf-8')
    (path/'same_input_balance_no_tangent.jsonl').write_text('\n'.join(json.dumps(r) for r in balanced_only)+'\n','utf-8')
    truth=sorted(json.loads(Path(label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    arms=[('same_input_no_motion',control),('same_input_motion_without_flow_balance',motion_only),('same_input_full_balance_without_tangent',balanced_only),('RAW_current_motion',rows)]
    scores={n:score(v,truth) for n,v in arms}
    for name,values in arms:
        errors=[];last=None;detail=[]
        for r in values:
            slot=int(np.floor(r['t']*5+1e-8))
            if not r['valid'] or slot==last:continue
            last=slot;target=reference_at(truth,r['t']+.4)
            if target is None:continue
            error=float(np.linalg.norm(np.asarray(r['physical']['0.4'])-target));errors.append(error)
            detail.append(dict(t=r['t'],target=target.tolist(),prediction=r['physical']['0.4'],error_px=error))
        scores[name]['within50_fraction']=None if not errors else float(np.mean(np.asarray(errors)<50))
        scores[name]['future04_detail']=detail
    last10=[r['cost']['completion_lag_ms'] for r in rows[-10:]]
    result=dict(scores=scores,current_motion_velocity_updates=sum(bool((r.get('image_motion') or {}).get('velocity',{}).get('applied')) for r in rows),
        current_motion_acceleration_updates=sum(bool((r.get('image_motion') or {}).get('acceleration',{}).get('applied')) for r in rows),
        tail10_completion_lag_ms=dict(mean=float(np.mean(last10)),min=min(last10),max=max(last10)),
        same_current_input_force_geometry_and_prediction_slots=True,
        control_is_replay_not_separate_RAW_timing=True,
        observed_full_track_scatter_used=True,labels_are_exposed_development_data=True)
    (path/'shared_motion_accuracy.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    for value in scores.values():value.pop('anchors');value.pop('future04_detail')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main(*sys.argv[1:])
