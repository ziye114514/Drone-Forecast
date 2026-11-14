"""Score fixed goal windows; old force receives the SAME current observations."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/goal50px30fps_20261027'
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from fast_force_response import FastForceResponse
from responsive_force import DampedResponsiveForce
from compare_final_pipeline import score,reference_at,stats


def run():
    cases={};frozen={}
    for name,folder,start,end,label in [
        ('side','side_axis_responsive',110.,114.,'dense_labels_side_110_1195'),
        ('context','side_context_burst_fixed',110.,114.,'dense_labels_side_110_1195'),
        ('train2','train2_transfer_aligned',40.,46.,'dense_labels_train2_40_46'),
        ('train2_association','train2_association',40.,46.,'dense_labels_train2_40_46'),
        ('clear','clear_axis_responsive',26.,28.,None)]:
        path=OUT/folder
        if not (path/'summary.json').is_file():continue
        rows=[json.loads(s) for s in (path/'predictions.jsonl').read_text('utf-8').splitlines()]
        model=FastForceResponse();check=DampedResponsiveForce();control=[];maximum=0.
        for row in rows:
            observed=row.get('force_observed_position')
            pred,diag=model.step(row['t'],row['coefficient'],observed,row['force']['input_u'],.35)
            reproduced,_=check.step(row['t'],row['coefficient'],observed,row['force']['input_u'],.35)
            if row['physical']:maximum=max(maximum,max(abs(np.asarray(reproduced[k])-v).max() for k,v in row['physical'].items()))
            control.append(dict(row,physical=pred,force=diag))
        assert maximum<1e-8,(name,maximum)
        # Baseline full causal outputs saved before opening references.
        (path/'same_observations_original_force.jsonl').write_text('\n'.join(json.dumps(r) for r in control)+'\n','utf-8')
        selected=[r for r in rows if start<=r['t']<end];old=[r for r in control if start<=r['t']<end]
        frozen[name]=(rows,selected,old,label,path)
        summary=json.loads((path/'summary.json').read_text('utf-8'))
        cases[name]=dict(run=str(path),source_window=[start,end],inference_start=rows[0]['t'],source_frames=len(rows),
            summary=summary,force_replay_max_error_px=maximum,
            last10_completion_lag_ms=stats([r['cost']['completion_lag_ms'] for r in rows[-10:]]),
            scheduled_axis_ticks=sum(r['geometry'].get('current_axis_budget_tick',False) for r in rows))
    for name,(rows,selected,old,label,path) in frozen.items():
        if label is None:continue
        truths=sorted(json.loads((WORK/'output'/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        metrics={n:score(r,truths) for n,r in [('new',selected),('original_force_same_observations',old)]}
        last=None;samples=[];lookup={r['t']:r for r in old}
        for row in selected:
            slot=int(np.floor(row['t']*5+1e-8))
            if not row['valid'] or slot==last:continue
            last=slot;target=reference_at(truths,row['t']+.4)
            if target is None:continue
            errors={n:float(np.linalg.norm(np.asarray(r['physical']['0.4'])-target)) for n,r in [('new',row),('original_force_same_observations',lookup[row['t']])]}
            samples.append(dict(t=row['t'],target_t=row['t']+.4,errors=errors))
        cases[name].update(scores={n:{k:v for k,v in s.items() if k!='anchors'} for n,s in metrics.items()},
            within50_fraction={n:float(np.mean([s['errors'][n]<=50. for s in samples])) if samples else None for n in metrics},
            paired_0_4s_samples=samples,
            mean50_target_met=bool(samples and metrics['new']['physical_future_error_px']['0.4']['mean']<50.))
        (path/'goal_scores.json').write_text(json.dumps(cases[name],indent=2),'utf-8')
    result=dict(cases=cases,goal=dict(horizon_s=.4,mean_error_px_lt=50.,full_pipeline_FPS_at_least=30.,
        also_report_P95_and_within50=True,source_frames_never_dropped=True),
        inference_never_reads_annotations=True,baseline_and_new_share_source_times_and_body_inputs=True,
        independent_heldout_test=False,both_goals_globally_achieved=False,
        limits='Existing development labels. Short noisy timings; async output sequence may differ from prior runs. Do not combine accuracy from one run with FPS of another; context includes a burst, all0.5s past events are processed.')
    (OUT/'goal_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({n:dict(fps=v['summary']['steady_capacity_fps'],fresh_Hz=v['summary']['paced_fresh_observation_rate'],
        old_new_0_4s_mean={k:s['physical_future_error_px']['0.4']['mean'] for k,s in v.get('scores',{}).items()},
        within50=v.get('within50_fraction'),overflows=v['summary'].get('decoder_overflow_packets')) for n,v in cases.items()},ensure_ascii=False))


if __name__=='__main__':run()
