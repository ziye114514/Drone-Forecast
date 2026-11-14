"""One physical state-feedback mechanism on frozen current observations."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_fast_structure import PhysicalPositionState
from force_position_state import ForceDrivenNoiseKF
from responsive_force import DampedResponsiveForce
from compare_final_pipeline import score


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);outputs={}
    for name,path in [('side4','final_pipeline_speed_20261027/side_async_prefilter'),
        ('side3','three_continuation_20261026/raw_release_side'),('clear','three_continuation_20261026/raw_release_clear')]:
        rows=[json.loads(s) for s in (ROOT.parent/'output'/path/'predictions.jsonl').read_text('utf-8').splitlines()]
        state=PhysicalPositionState();state.kf=ForceDrivenNoiseKF();force=DampedResponsiveForce();result=[]
        for r in rows:
            body=dict(r['body']);diag=r['center_bridge']
            if diag.get('applied'):body['center']=diag['fused_position']
            state.kf.noise_scale=diag.get('noise_scale',1.)
            c,valid,info=state.step(r['t'],body)
            observed=body.get('center') if body.get('detected') else None
            pred,f=force.step(r['t'],c,observed,r['force']['input_u'],.35)
            state.kf.set_control(f['acceleration_px_s2'])
            result.append(dict(r,coefficient=c,valid=valid,physical=pred,force=f))
        outputs[name]=result
        (out/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in result)+'\n','utf-8')
    truth=sorted(json.loads((ROOT.parent/'output/dense_labels_side_110_1195/labels.json').read_text('utf-8'))['records'].values(),key=lambda z:z['t'])
    scores={n:score(r,truth)['physical_future_error_px'] for n,r in outputs.items() if n!='clear'}
    summary=dict(scores=scores,source_observations_frozen=True,all_causal_outputs_saved_before_labels=True,
        parameters_swept=False,force_also_controls_current_position_velocity_state=True,
        limitations='Previously exposed development references; no RAW speed/independent force/held-out generalization claim.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary))


if __name__=='__main__':run(sys.argv[1])
