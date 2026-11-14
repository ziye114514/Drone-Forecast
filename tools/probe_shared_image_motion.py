"""Reuse frozen KLT deltas before the physical force learner; no new RAW/flow."""
from pathlib import Path
import sys,json,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from shared_image_motion import SharedImageMotion
from run_fast_structure import PhysicalPositionState
from responsive_force import DampedResponsiveForce
from compare_final_pipeline import score


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);outputs={};summary={}
    for name,path,label in [('side','goal50px30fps_20261027/side_axis_responsive','dense_labels_side_110_1195'),
        ('train2','shared_template_budget_20261027/train2_current_nn','dense_labels_train2_40_46')]:
        source=ROOT.parent/'output'/path;rows=[json.loads(s) for s in (source/'predictions.jsonl').read_text('utf-8').splitlines()]
        state=PhysicalPositionState();motion=SharedImageMotion();force=DampedResponsiveForce();values=[];times=[]
        max_state_delta=0.
        for r in rows:
            body=dict(r['body']);bridge=r['center_bridge']
            if bridge.get('applied'):body['center']=bridge['fused_position']
            state.kf.noise_scale=bridge.get('noise_scale',1.);c,valid,_=state.step(r['t'],body)
            if c is not None:max_state_delta=max(max_state_delta,float(abs(np.asarray(c['position'])-r['coefficient']['position']).max()))
            begin=time.perf_counter();c,info=motion.step(r['t'],c,state.kf.P[2:4,2:4] if c else np.eye(2),body,r['tracking'])
            p,f=force.step(r['t'],c,r.get('force_observed_position'),r['force']['input_u'],.35)
            times.append((time.perf_counter()-begin)*1000)
            values.append(dict(r,coefficient=c,physical=p,force=f,shared_image_motion=info))
        assert max_state_delta<1e-8,max_state_delta
        (out/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
        outputs[name]=(rows,values,label)
        summary[name]=dict(source_frames=len(rows),position_state_max_difference=max_state_delta,
            mean_CPU_ms=float(np.mean(times)),velocity_updates=sum(r['shared_image_motion']['velocity'].get('applied',False) for r in values),
            acceleration_updates=sum(r['shared_image_motion']['acceleration'].get('applied',False) for r in values))
    for name,(old,new,label) in outputs.items():
        truth=sorted(json.loads((ROOT.parent/'output'/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        summary[name]['scores']={n:score(r,truth)['physical_future_error_px'] for n,r in [('control',old),('shared_motion',new)]}
    result=dict(cases=summary,all_outputs_before_annotations=True,old_published_slots_retained=True,
        no_extra_flow_pass_or_position_KF=True,full_RAW_FPS_not_measured=True,
        caveat='Saved old KLT records lack scatter; use conservative2px proxy. Actual RAW must record measured scatter before promoting.')
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
