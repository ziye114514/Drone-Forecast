"""Finalize reviewed target-absent external sequence; NEVER score board FDE.

All27 original full native snapshots on four519 pages were visually inspected
before this script. They show a person/calibration board, no drone. The pending
automatic board outlines remain preserved as invalid target references.
"""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    out=ROOT.parent/'output/frozen_external518_20261113';ref=out/'reference'
    assert (out/'both_predictions_frozen.json').exists()
    pending=json.loads((ref/'references_pending_review.json').read_text('utf8'))
    assert len(pending['records'])==27 and all((ref/f'review_page{i}.png').exists() for i in range(4))
    records={k:dict(t=r['t'],state='absent',box=None,weaker_outline_box=None,
        automatic_board_box_INVALID_target_reference=r.get('box'),
        visual_review='Full native frame contains person with calibration board, no visible drone.',
        forecast_outputs_used_for_annotation=False,reference_is_independent_human_GT=False,
        assistant_visual_review_pending=False) for k,r in pending['records'].items()}
    (ref/'references.json').write_text(json.dumps(dict(records=records,
        assistant_full_native_frame_review_completed=True,reviewed_pages=[f'review_page{i}.png' for i in range(4)],
        calibration_sequence_not_drone_flight=True,automatic_dense_board_boxes_are_NOT_drone_truth=True,
        predictor_outputs_not_used_for_annotation=True),indent=2),'utf8')
    cases={}
    for arm in ('baseline487','candidate517'):
        rows=[json.loads(s) for s in (out/arm/'predictions.jsonl').read_text('utf8').splitlines()];queries=[]
        for t in np.round(np.arange(10.8,15.6001,.2),1):
            r=[q for q in rows if q['t']<=t+1e-9][-1];fresh=t-r['t']<=.04
            queries.append(dict(t=float(t),source_t=r['t'],target_present=False,
                false_positive_current_detection=bool(fresh and r['body']['detected']),
                false_positive_valid_forecast=bool(fresh and r['valid']),
                period_input_supplied=bool(r.get('period_selection',{}).get('learning_input_current')),
                mode=r['structure']['mode']))
        cases[arm]=dict(reviewed_fixed_queries=25,current_detection_false_positives=sum(r['false_positive_current_detection'] for r in queries),
            valid_forecast_false_positives=sum(r['false_positive_valid_forecast'] for r in queries),records=queries,
            FDE_px=None,FDE_unavailable_reason='No drone trajectory exists in the fixed external interval.')
    protocol=json.loads((out/'protocol.json').read_text('utf8'))
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in protocol['source_sha256'].items())
    result=dict(cases=cases,new_drone_motion_accuracy_validated=False,FDE_px=None,eligible_positive_origins=0,
        fixed_interval_preserved=True,no_tuning_based_on_external=True,source_frozen_through_inference_and_review=True,
        calibration_negative_control_only=True,FPS_gate_met=False,
        plain_language_result='预留新录像解封后发现是人举标定板，27帧原生画面均未见无人机；该固定区间只能作阴性检查，不能验证新无人机录像的30px预测精度。未改区间、未以其结果调参，FPS验收门槛仍未满足。')
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf8')
    print(json.dumps({k:{a:b for a,b in s.items() if a!='records'} for k,s in cases.items()},indent=2))
