"""Close the completed batch with explicit unachieved accuracy/blind gates."""
from pathlib import Path
from datetime import datetime
import sys,subprocess,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent

if __name__=='__main__':
    subprocess.run([sys.executable,str(ROOT/'tools/document_target30p5_490.py')],check=True)
    out=WORK/'output/target30p5_20261112';s=json.loads((out/'summary.json').read_text('utf8'))
    assert s['best_stage']=='487' and not s['target30p5_achieved']
    assert not (WORK/'output/frozen_external483_20261112').exists()
    for folder in ('raw_local_contrast487_20261112','raw_contrast_symmetric488_20261112','raw_contrast_plane492_20261112'):
        d=json.loads((WORK/'output'/folder/'progress.json').read_text('utf8'));assert len(d)==6
        assert all(q['frames']==181 and q['native_packets_read']==181 and q['skipped_observations']==0 for q in d.values())
    doc=ROOT/'docs/HANDOFF_20261112_TARGET30P5.md';note=doc.read_text('utf8')
    note+='\n本批次478–489及492全部退出，无后台RAW/GPU任务。最新推荐是487＋原186教师模型；目标未达，不能标项目完成。\n'
    doc.write_text(note,'utf8');(out/'handoff.md').write_text(note,'utf8')
    current=s['best'];stamp=datetime.now().astimezone().isoformat()
    p=WORK/'output/checkpoints/latest_handoff.json';d=json.loads(p.read_text('utf8'))
    d.update(timestamp=stamp,in_progress=False,running_jobs=[],background_jobs=[],experiments_running=[],
        objective_completed=False,account_quota_exhaustion_verified=False,
        project_work_state='accuracy_batch_finished_target_not_achieved',
        current_user_stage='478-489 and492 complete; best487 actual6RAW original102 mean31.471473',
        status='All batch jobs exited.487 genuine development improvement31.954595->31.471473.30.5 target unmet; reserved blind kept until30 gate;FPS deferred.',
        current_status_document=str(doc),latest_report=str(out/'index.html'),candidate_entry=s['entry'],
        default_model_path=str(ROOT/'models/native_response_plane_v1.json'),
        current_precision_mean_px=current['mean'],aggregate_same102=dict(current,within30p5=66),
        candidate_scores={n:dict(n=q['n'],mean=q['candidate_mean']) for n,q in s['scores'].items()},
        current_target_FDE_px=30.5,next_target_FDE_px=30.5,milestone31_achieved=False,
        milestone30p5_achieved=False,blind_gate_mean_px=30.0,
        current_pipeline_uses_body_NN=False,raw_current_all6_complete=True,FPS_optimization_allowed=False,
        all6_exposed_development=True,no_recording_specific_inference_rules=True,
        best_experimental_mean_px=current['mean'],best_experimental_entry=s['entry'],
        current_turn_completed_trials={k:v['mean'] for k,v in s['trial_summaries'].items()},
        next_action=[
            'Read HANDOFF_20261112_TARGET30P5.md; preserve487+old model and473 control.',
            'One root: validate native translation against current time-separated evidence, after reading189/279-282/350 negatives. Do not feed compensated pixels into geometry.',
            'Do not redo affine/LK thresholds480/492, paired cost484/485, timestamp dewarp482, same13parameter retraining489 or axis covariance486 without new evidence.',
            'Keep reserved video undecoded until original30px gate, then fixed frozen test before FPS.'])
    # Retain an explicitly named historical score instead of a stale current P95.
    if 'current_precision_p95_px' in d:d['historical473_p95_px']=d.pop('current_precision_p95_px')
    d['failed_coverage482']=dict(original_available=86,original_missing=16,promoted=False,comparable_full102_mean=None)
    d['external483_execution']='Denied before CreateProcess; user chose preserve blind; no decode/predictions/imagery. Archived script unconditionally exits.'
    d['blind_video_user_instruction_20261112']='Retain blind; continue optimizing to gate. No early validation permission.'
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf8')
    p=WORK/'WORKLOG.md';text=p.read_text('utf8');a='<!-- target30p5-handoff:start -->';b='<!-- target30p5-handoff:end -->'
    if a in text:text=text[text.index(b)+len(b):].lstrip()
    header=f'''{a}
当前权威交接：[目标30.5](event_drone_forecast/docs/HANDOFF_20261112_TARGET30P5.md)；[图文结果](output/target30p5_20261112/index.html)。487同六源实际RAW、原102点均31.471473px，较473改善0.483121px，中位25.379331；4/6源改善、2/6回退，非泛化证明。入口run_rotor_local_contrast_candidate.py＋原模型。489重训34.855904、492局部平面32.452010，均未推广。本批全部作业退出，30.5未达。用户明确保留盲测，原30px＋新录像通过后的FPS门槛保持；不要提前解码。以下为历史。
{b}

'''
    text=header+text+'\n\n## 2026-11-12 478–493：局部事件能量与平移解耦\n\n'+note
    p.write_text(text,'utf8')
    p=ROOT/'README.md';text=p.read_text('utf8');a='<!-- target30p5-current:start -->';b='<!-- target30p5-current:end -->'
    if a in text:text=text[text.index(b)+len(b):].lstrip()
    header=f'''{a}
> **当前研究（2026-11-12）**：同原102起点0.4s均值31.955→**31.471px**，中位25.379；30.5目标未达。当前实验入口`run_rotor_local_contrast_candidate.py`＋原`models/native_response_plane_v1.json`。局部事件能量归一化改善桨叶位移匹配，纯ON/OFF＋五圆＋可测代理/未知合力主线保留，无NN/G；不是已测准真实推力。4/6开发源改善、2/6回退，新录像盲测仍保留，FPS后置。见[报告](../output/target30p5_20261112/index.html)与[交接](docs/HANDOFF_20261112_TARGET30P5.md)。以下为历史。
{b}

'''
    p.write_text(header+text,'utf8')
    print('FINAL SAVED',out/'index.html',s['best_stage'],current)
