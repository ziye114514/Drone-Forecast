"""Close a single short serial/overlap RAW comparison without retiming targets."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_scale_rollout import ScaleRolloutMotion,scale_predictions
from live_local_flow import velocity_coefficient
from score_live_labels import evaluate
from summarize_live_pilot import read_rows
OUT=WORK/'output/support_overlap_20261026';assert not (OUT/'actual_comparison.json').exists()
report={}
for mode,folder in [('serial','actual_serial'),('gpu-overlap','actual_overlap')]:
    run=OUT/folder;completion=json.loads((run/'completion.json').read_text('utf-8'));protocol=json.loads((run/'protocol.json').read_text('utf-8'))
    assert completion['sources_unchanged'] and completion['frame_descriptor_mismatches']==0
    assert completion['flow_preparation']==protocol['flow_preparation']==mode
    assert not completion['labels_read'] and not completion['output_caches_read']
    model=ScaleRolloutMotion(25.);states={};exact=0
    rows=read_rows(run/'tracking.jsonl')
    for row in rows:
        t=row['t'];c,valid,_=model.step(t,row['body']);assert c==row['position_only_coefficient'] and valid==row['valid']
        obs=row['diagnostic']['local_flow']['observation']
        updated,_=velocity_coefficient(c,model.kf.P[2:4,2:4] if c else None,obs,t);assert updated==row['coefficient'];exact+=1
        assert row['detector_frame_sha256']==row['gpu_frame_sha256']==row['producer_frame_sha256']
        assert isinstance(row['support_sha256'],str) and len(row['support_sha256'])==64
        states[round(t,6)]=updated
    for p in read_rows(run/'predictions.jsonl'):
        if p['observation_t'] is not None:
            predicted=scale_predictions(states[round(p['observation_t'],6)],p['observation_age_ms']/1000)[0]
            assert predicted==p['predictions']['live_motion']
    scores=evaluate(run,WORK/'output/dense_labels_side_110_1195/labels.json')
    (OUT/(folder+'_scores.json')).write_text(json.dumps(scores,indent=2),encoding='utf-8')
    fields=['tracking_updates','publication_slots','valid_publications','processing_median_ms','observation_age_p95_ms',
        'support_preparation_median_ms','NN_receive_wait_median_ms','NN_roundtrip_median_ms','local_flow_median_ms']
    report[mode]=dict(runtime={k:completion[k] for k in fields},exact_state_and_publication_replay_ticks=exact,
        actual04s=scores['tables']['0.4']['metrics'],frame_and_current_support_recorded=True)
baseline=report['serial']['runtime'];candidate=report['gpu-overlap']['runtime']
decision=dict(actual_processing_median_saved_ms=baseline['processing_median_ms']-candidate['processing_median_ms'],
    valid_publications_not_reduced=candidate['valid_publications']>=baseline['valid_publications'],
    different_actual_acceptance_and_publication_times=True,accuracy_is_descriptive_not_a_fixed_cohort_gain=True,
    whole_pipeline_replications=1,scope='One serial/overlap pair on4s side RAW; short engineering observation, not sustained performance or significance.')
result=dict(results=report,decision=decision)
(OUT/'actual_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
