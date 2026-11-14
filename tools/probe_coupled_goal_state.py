"""Reuse existing force state model on fixed inputs; retain original score slots."""
from pathlib import Path
import sys,json,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from coupled_force_observer import CoupledForceObserver
from compare_final_pipeline import score


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);results={};summaries={}
    for name,path,label in [
        ('side','goal50px30fps_20261027/side_axis_responsive','dense_labels_side_110_1195'),
        ('train2','shared_template_budget_20261027/train2_current_nn','dense_labels_train2_40_46'),
        ('clear','shared_template_budget_20261027/clear_fused_goal',None)]:
        source=ROOT.parent/'output'/path;rows=[json.loads(s) for s in (source/'predictions.jsonl').read_text('utf-8').splitlines()]
        model=CoupledForceObserver();values=[];times=[]
        for r in rows:
            begin=time.perf_counter();c,pred,valid,diag=model.step(r['t'],r.get('force_observed_position'),r['force']['input_u'],.35,r['center_bridge'].get('noise_scale',1.))
            times.append((time.perf_counter()-begin)*1000)
            # Compare EVERY original valid slot even if the new observer's
            # diagnostic readiness differs; report availability separately.
            values.append(dict(r,coefficient=c,physical=pred,force=diag,new_valid=valid))
        (out/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
        results[name]=(rows,values,label)
        summaries[name]=dict(frames=len(rows),retained_valid=sum(r['valid'] for r in rows),new_valid=sum(r['new_valid'] for r in values),
            mean_CPU_ms=float(np.mean(times)),gain=model.gain,accepted_positions=sum(bool(r['force']['state_observer']['accepted_position']) for r in values))
    for name,(original,values,label) in results.items():
        if label is None:continue
        truth=sorted(json.loads((ROOT.parent/'output'/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        summaries[name]['scores']={n:score(rows,truth)['physical_future_error_px'] for n,rows in [('control',original),('joint_force_state',values)]}
    result=dict(cases=summaries,all_original_score_slots_retained=True,all_predictions_saved_before_labels=True,
        original_position_observations_event_inputs_unchanged=True,independent_test=False,parameters_swept=False,
        scope='Existing6state observer with current event control and Singer mean rollout; not a new physical thrust measurement. Timing is component only.')
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
