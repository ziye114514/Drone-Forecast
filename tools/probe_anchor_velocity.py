"""Use current track displacement through an independently strong NN reanchor."""
from pathlib import Path
import sys,json,time,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from shared_image_motion import SharedImageMotion
from run_fast_structure import PhysicalPositionState
from flow_force_balance import FlowBalancedForce
from compare_final_pipeline import score


def main(run,label,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in (Path(run)/'predictions.jsonl').read_text('utf-8').splitlines()]
    if 'projected_unknown_force' in rows[0]['force']:
        from projected_unknown_force import ProjectedResidualForce
        model=ProjectedResidualForce()
    else:model=FlowBalancedForce()
    motion=SharedImageMotion();state=PhysicalPositionState();values=[];times=[]
    for r in rows:
        body=copy.deepcopy(r['body']);bridge=r['center_bridge']
        if bridge.get('applied'):body['center']=bridge['fused_position']
        state.kf.noise_scale=bridge.get('noise_scale',1.);c,valid,_=state.step(r['t'],body)
        assert valid==r['valid']
        if c is not None:assert np.allclose(c['position'],r['coefficient']['position'],atol=1e-8,rtol=0)
        start=time.perf_counter();c,info=motion.step(r['t'],c,state.kf.P[2:4,2:4] if c else np.eye(2),body,r['tracking'])
        model.observe_flow(info['observation'],info['acceleration'].get('applied',False))
        if hasattr(model,'observe_axis_motion'):model.observe_axis_motion(r['structure'],info)
        model.input_learning_current=(r.get('axis_amplitude_bridge') or {}).get('learning_input_current',True)
        p,f=model.step(r['t'],c,r.get('force_observed_position'),r['force']['input_u'],.35)
        times.append((time.perf_counter()-start)*1000);values.append(dict(r,coefficient=c,image_motion=info,physical=p,force=f))
    (out/'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
    truth=sorted(json.loads(Path(label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    result=dict(position_and_validity_unchanged=True,current_tracks_used_for_velocity_not_NN_center_jump=True,
        RAW_FPS_not_measured=True,anchor_upgrades=sum((r['image_motion']['observation']).get('anchor_upgrade',False) for r in values),
        visibility_reanchors=sum((r['image_motion']['observation']).get('visibility_reanchor',False) for r in values),
        mean_motion_force_CPU_ms=float(np.mean(times)),scores={n:score(v,truth)['physical_future_error_px'] for n,v in [('control',rows),('anchor_velocity',values)]})
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(*sys.argv[1:])
