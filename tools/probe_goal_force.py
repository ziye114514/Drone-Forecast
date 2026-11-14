"""One mechanism change on frozen causal observations; labels opened last."""
from pathlib import Path
import json,sys,time,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from fast_force_response import FastForceResponse
from responsive_force import ResponsiveForce,DampedResponsiveForce,CalibratedResponsiveForce,InnovationResidualForce,WindowResidualForce
from compare_final_pipeline import score


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    paths=dict(side4=ROOT.parent/'output/final_pipeline_speed_20261027/side_async_prefilter',
        side3=ROOT.parent/'output/three_continuation_20261026/raw_release_side',
        clear=ROOT.parent/'output/three_continuation_20261026/raw_release_clear')
    summaries={};outputs={};reconstruction=[]
    for name,path in paths.items():
        rows=[json.loads(s) for s in (path/'predictions.jsonl').read_text('utf-8').splitlines()]
        models=dict(reference=FastForceResponse(),strict_gain=FastForceResponse(gain_signal_to_noise=2.),recent_disturbance=ResponsiveForce(),damped_recent=DampedResponsiveForce(),calibrated=CalibratedResponsiveForce(),innovation_residual=InnovationResidualForce(),local_force_balance=WindowResidualForce())
        outputs[name]={n:[] for n in models};timings={n:[] for n in models}
        for row in rows:
            observed=row.get('force_observed_position')
            if observed is None and row['body'].get('detected'):
                observed=row['center_bridge'].get('fused_position',row['body']['center'])
            u=row['force']['input_u']
            for n,model in models.items():
                start=time.perf_counter();pred,info=model.step(row['t'],row['coefficient'],observed,u,.35)
                timings[n].append(1000*(time.perf_counter()-start))
                new=dict(row,physical=pred,force=info);outputs[name][n].append(new)
                if n=='reference' and row['physical']:
                    reconstruction.append(max(abs(np.asarray(pred[k])-v).max() for k,v in row['physical'].items()))
        summaries[name]=dict(frames=len(rows),replay_reference_max_error_px=max(reconstruction),
            cpu_mean_ms={n:float(np.mean(v)) for n,v in timings.items()},
            final_gains={n:m.gain for n,m in models.items()})
    assert max(reconstruction)<1e-8,max(reconstruction)
    # Save all causal outputs BEFORE accessing the external annotation file.
    for name,arms in outputs.items():
        for arm,rows in arms.items():
            (out/f'{name}_{arm}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n','utf-8')
    labels=ROOT.parent/'output/dense_labels_side_110_1195/labels.json'
    truth=sorted(json.loads(labels.read_text('utf-8'))['records'].values(),key=lambda z:z['t'])
    for name in ['side4','side3']:
        summaries[name]['scores']={n:score(rows,truth)['physical_future_error_px'] for n,rows in outputs[name].items()}
    result=dict(cases=summaries,source_inputs_frozen=True,outputs_saved_before_labels=True,parameters_swept=False,
        limitations='Previously exposed side development labels only; clear has no new precision score. Same source targets across arms. No calibrated force truth or held-out generalization.')
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1])
