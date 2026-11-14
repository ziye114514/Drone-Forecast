"""One missing-thrust mechanism on fixed observations, labels read last."""
from pathlib import Path
import sys,json,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from responsive_force import DampedResponsiveForce
from unknown_force_feedback import UnknownForceFeedback
from compare_final_pipeline import score


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);outputs={};summary={}
    for name,folder,label in [('side','side_axis_responsive','dense_labels_side_110_1195'),
        ('train2','train2_transfer_aligned','dense_labels_train2_40_46'),('clear','clear_axis_responsive',None)]:
        path=ROOT.parent/'output/goal50px30fps_20261027'/folder
        rows=[json.loads(s) for s in (path/'predictions.jsonl').read_text('utf-8').splitlines()]
        models=dict(control=DampedResponsiveForce(),unknown_feedback=UnknownForceFeedback());values={n:[] for n in models};cost={n:[] for n in models}
        error=0.
        for r in rows:
            for n,model in models.items():
                begin=time.perf_counter();pred,diag=model.step(r['t'],r['coefficient'],r.get('force_observed_position'),r['force']['input_u'],.35)
                cost[n].append((time.perf_counter()-begin)*1000);values[n].append(dict(r,physical=pred,force=diag))
                if n=='control' and pred:error=max(error,max(abs(np.asarray(pred[h])-p).max() for h,p in r['physical'].items()))
        assert error<1e-8,error
        for n,value in values.items():(out/f'{name}_{n}.jsonl').write_text('\n'.join(json.dumps(r) for r in value)+'\n','utf-8')
        summary[name]=dict(source_frames=len(rows),control_reconstruction_error=error,
            mean_CPU_ms={n:float(np.mean(v)) for n,v in cost.items()},
            unknown_seed_frames=sum(r['force']['total_unknown_force_seed_supplied'] for r in values['unknown_feedback']),
            event_thrust_inputs=sum(r['force']['active'] for r in values['unknown_feedback']))
        outputs[name]=(values,label)
    for name,(values,label) in outputs.items():
        if label is None:continue
        truth=sorted(json.loads((ROOT.parent/'output'/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        summary[name]['scores']={n:score(r,truth)['physical_future_error_px'] for n,r in values.items()}
    result=dict(cases=summary,all_causal_outputs_saved_before_labels=True,unchanged_position_velocity_and_event_inputs=True,
        full_RAW_FPS_not_measured=True,parameters_swept=False,independent_force_truth_available=False,
        scope='Previously exposed development boxes; mechanism benefit is unknown-motion feedback, not measured rotor thrust.')
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
