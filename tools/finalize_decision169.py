"""Finalize bounded169 documentation/checkpoint; never rerun prediction fitting."""
from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
OUT = WORK / 'output/decision169_20261105'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value:
                self.links.append(value)


if __name__ == '__main__':
    # Correct an overly broad draft label; no predictions/metrics are recomputed.
    score_path = OUT / 'decision_scores.json'
    scores = json.loads(score_path.read_text('utf-8'))
    if 'manual_supervision_upper_bound_only' in scores['decision']:
        shutil.copy2(score_path, OUT / 'decision_scores_before_terminology_correction.json')
        scores['decision'].pop('manual_supervision_upper_bound_only')
        scores['decision'].update(manual_calibration_diagnostic=True, universal_upper_bound=False)
        score_path.write_text(json.dumps(scores, indent=2), 'utf-8')
    assert not scores['decision']['all_suffixes_beat_both_controls']
    assert not scores['decision']['more_parameter_search_allowed_in_this_batch']

    # Importing the report builder does not execute it. Run separately before this tool.
    expected = [
        OUT / 'index.html', OUT / 'geometry_value.png', OUT / 'DECISION_STATUS.json',
        OUT / 'final_source_hashes.json', ROOT / 'rotor_evidence.py',
        ROOT / 'docs/ROTOR_MODEL_V1.md', ROOT / 'docs/HANDOFF_20261105_DECISION169.md',
        OUT / 'exports/clear/rotor_model.jsonl', OUT / 'exports/pure_side/rotor_model.jsonl',
    ]
    assert all(p.is_file() for p in expected)
    links = Links()
    links.feed((OUT / 'index.html').read_text('utf-8'))
    missing = [s for s in links.links if not re.match(r'^[a-z]+:', s)
               and not (OUT / s.split('#')[0]).resolve().exists()]
    assert not missing, missing
    manifest = json.loads((OUT / 'final_source_hashes.json').read_text('utf-8'))
    assert all(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in manifest.items())
    lines = {}
    for name in ('clear', 'pure_side'):
        path = OUT / f'exports/{name}/rotor_model.jsonl'
        records = [json.loads(s) for s in path.read_text('utf-8').splitlines()]
        assert records
        lines[name] = len(records)
    assert lines == {'clear': 120, 'pure_side': 60}

    checkpoint_path = WORK / 'output/checkpoints/latest_handoff.json'
    backup = OUT / 'latest_handoff_before169.json'
    if not backup.exists():
        shutil.copy2(checkpoint_path, backup)
    checkpoint = json.loads(checkpoint_path.read_text('utf-8-sig'))
    layout = json.loads((OUT / 'layout_summary.json').read_text('utf-8'))
    now = datetime.now(timezone.utc).isoformat()
    report = str(OUT / 'index.html')
    handoff = str(ROOT / 'docs/HANDOFF_20261105_DECISION169.md')
    module_doc = str(ROOT / 'docs/ROTOR_MODEL_V1.md')
    checkpoint.update(
        utc=now, updated_local_date='2026-11-05', client_date='2026-11-05',
        execution_status='Stage169 bounded batch finished: visibility-conditioned rotor evidence/layout API delivered. Controlled missing-geometry consistency improved; frozen force response diagnostic failed. No claim of complete prediction improvement.',
        research_objective_status='incomplete', last_completed_stage=169, latest_numbered_stage=169,
        latest_diagnostic_stage=169, last_completed_named_batch='decision169_20261105',
        in_progress=False, experiments_running=False, running_jobs=0, pending_runs=[], running_RAW_sessions=[],
        completed_scope='Callable native-candidate geometry/evidence backend, controlled missing-observation consistency, one frozen diagnostic; not actual occlusion or30px/30FPS completion.',
        completion_scope='Finite169 batch only; force prediction remains experimental, default and published ZIPs unchanged.',
        default_promoted=False, stable_high_accuracy_realtime_achieved=False,
        stable_high_accuracy_realtime=False, event_RPM_advantage_proven=False,
        independent_RPM_gain_proven=False, realtime_proven=False,
        latest_report=report, previous_report=str(WORK / 'output/source_force168_20261105/index.html'),
        latest_research_report=report, latest_diagnostic_report=report,
        current_status_document=handoff, latest_instruction_handoff=handoff,
        current_findings_document=module_doc, latest_implementation_document=module_doc,
        handoff_priority_document=handoff, latest_research_document=handoff,
        latest_explanation_report=report, latest_named_report_is_authoritative=True,
        older_numbered_stage_fields_are_historical=True,
        latest_actual_RAW_stage=165, latest_cached_forecast_stage=168,
        latest_cached_input_force_diagnostic_stage=169,
        latest_artifact_checks=str(OUT / 'artifact_checks169.json'),
        latest_source_snapshot=str(OUT / 'final_source_hashes.json'),
        work_in_progress_output=str(OUT), latest_progress_checkpoint=str(OUT / 'DECISION_STATUS.json'),
        active_batch='decision169_20261105 (bounded batch finished)',
        current_research_route='Own native ON/OFF rotor support and visibility-conditioned five-circle/three/two/side geometry first; currently observed contribution plus unknown force remains research only. NN optional auxiliary ROI, not substituted trajectory contribution.',
        next_action='Read HANDOFF_20261105_DECISION169.md. Do not tune force head further on these suffix labels. Use callable geometry API on a specific natural visibility-change failure with independent annotations before claiming actual detection/forecast gain. Preserve baseline and negative results;169 does not achieve30px/30FPS.',
        critical_next_steps=[
            'Real natural occlusion/visibility-change center check against independent reference, not same-detector deleted observations.',
            'Then test whether corrected own geometry removes a specific velocity/forecast jump; no genericNN score improvement in place of rotor contribution.',
            'Leave force response experimental until independent source/response constraints exist; no more parameter search on169 suffixes.',
        ],
        current_work_state=dict(name='Bounded rotor evidence and geometry169', state='completed_local_batch',
                                jobs=[], default_model_replaced=False, old_public_ZIPs_frozen=True,
                                goal_04s_mean_px=30, goal_fps=30, goal_met_across_scenes=False,
                                next_design=handoff),
    )
    checkpoint['latest_stage169'] = dict(
        report=report, handoff=handoff, module_document=module_doc,
        bounded_batch_finished=True, project_goal_complete=False, active_jobs=[],
        new_files=['rotor_evidence.py', 'calibrated_force_head.py', 'tools/rotor_model169.py'],
        module_input='already extracted current native ON/OFF arcs/ellipses and period tests',
        new_pixel_detector_claimed=False, NN_changed=False, NN_loaded_by_exporter=False,
        clear_candidates_upstream_NN_ROI=True, pure_side_export_upstream_NN=False,
        exported_rows=lines, geometry_reference=layout['reference'], geometry_test_type=layout['test_type'],
        complete_frames=20, query_frames=16, related_controlled_cases=64,
        naive_visible_centroid=layout['baseline_common'], history_only=layout['history_only_common'],
        shared_translation=layout['shared_translation_common'], old_affine=layout['model_common'],
        api=layout['api_geometry_consistency'], geometry_is_trajectory_FDE=False,
        actual_occlusion_accuracy_proven=False, true_COM_or_RPM_or_thrust_proven=False,
        frozen_force_diagnostic={name:dict(baseline=v['uncalibrated_baseline'],
                                         **{a:b['error'] for a,b in v['arms'].items()})
                                 for name,v in scores.items() if name != 'decision'},
        force_response_status='experimental_not_validated', manual_calibration_used=True,
        universal_upper_bound=False, new_blind_test=False, more_force_search_on_current_suffixes=False,
        final_source_hashes=str(OUT / 'final_source_hashes.json'),
        no_new_RAW_or_FPS_measurement=True, latest_actual_RAW_stage=165,
        last_full_native_cache_run_stage=168, default_promoted=False, global_novelty_proven=False,
    )
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), 'utf-8')
    log = WORK / 'WORKLOG.md'
    marker = '<!-- decision169-results:start -->'
    if marker not in log.read_text('utf-8-sig'):
        with log.open('a', encoding='utf-8') as dest:
            dest.write('''

<!-- decision169-results:start -->
## 2026-11-05 阶段169：有限收口，独立桨叶证据／可见性几何模块

- 用户要求“让我的思路有价值……不要陷在无止境的实验中”。本轮固定一个几何交付、一次冻结后段诊断；不改NN、不训练、不上传，不继续在失败后段标签上扫参数。
- 新rotor_evidence.py的RotorEvidence拆分当前区域/周期资格、模板锁定与局部时间支持。四路当前合格周期无序求和，无需持久电机ID；三/二路只保留实测贡献，推断桨无频率。当前候选数不是物理电机数；当前数据尚无“无模板四周期和”的实录正例，合成API正例不当实际覆盖收益。
- VisibilityLayout四点给弱透视布局中心候选；三点最近.4s模板优先共享平移，必要时复用旧形状检查仿射；两点/侧线不强求完整中心，prior有时效且不是当前测量。几何中心不等于真实COM，方向不等于完整3D姿态。
- 复用166清晰原生候选：20个完整帧中16帧有近期历史，逐个删一个当前中心共64个相关受控几何案例。只用3当前点+过去布局，不读当前NN机体位置；上游ROI仍有NN辅助。相对同帧完整检测布局中心，可见3点平均21.258px、仅历史速度.837px、共享平移/API .325px、旧仿射.348px；64/64输出。原仿射隐藏点一致性1.010px。简单共享平移已足够，不把复杂度当创新。
- 必须保留范围：只删几何观测，原图未模拟遮挡，候选可能错且弧拟合受先验约束；参考不独立。以上不是实际遮挡AP、真实COM、0.4s轨迹误差或64段独立数据，不证明世界首次/整套推力创新成功。
- 新tools/rotor_model169.py导出可独立消费JSON/API，不加载NN/权重/坐标标签，输入是已有原生候选，依赖仓库几何工具。最终exports/clear120行（20完整/4近期三点，上游NN ROI）；exports/pure_side60行无NN（6二点/54侧线，完整当前中心null）。未替换实时默认预测器。
- 一次固定标定—后段诊断：最多非负k+二维未知偏置；对齐/置零/过去.4s错时序，各用相同人工标定段并冻结。后段均值基线/对齐/零/错时序：NN清晰25.659/32.766/35.048/35.048；NN侧视45.741/59.245/62.690/56.835（9/11可评分）；NN另一段72.568/117.403/117.403/117.403；纯侧视39.498/42.378/42.378/50.582；纯另一段134.861/167.545/157.479/157.479。未通过，纯侧视k=0，不再扩大推力预测收益主张，不继续扫这些标签。只是低维响应头诊断，不是所有模型上限或无需标定部署；早稿“监督上限”术语更正留原评分副本，预测/数值未改。
- k/d可相互解释，隐藏桨力可快变，当前周期仍代理、投影/定位参考可变；不能据此断言所有物理方法不可能，也不能凭偏差修正声称辨识真实力。力学分支保留experimental。
- 机制检查已过无模板四和/置换不变/三点不补RPM/缺周期拒绝/侧视不造中心/平移/形变回退；正式拟合禁读后段目标，结束源码守卫通过，最终哈希和本地链接检查落盘。新代码交付版本与运行快照分别保留，不改原快照。
- 报告output/decision169_20261105/index.html；交接event_drone_forecast/docs/HANDOFF_20261105_DECISION169.md；接口ROTOR_MODEL_V1.md。最终特征用evidence_v2，最终示例用exports；根初稿和standalone_*不混用。
- 没有新RAW/GPU/FPS实验；最新实际RAW仍165，168原生缓存稳态29.93排首包/P95 54ms/含首包15.79FPS，不是持续RAW30Hz。30px/30Hz目标未完成，旧默认/发布ZIP/两处yolov10/RAW/权重不动；无新依赖/代理，无后台作业。
- 下一步仅针对实际自然可见数变化故障，用独立参考验证几何中心，再检验是否减少速度/预测异常。不要继续用同检测器删点结果充当真实遮挡精度，不用通用NN刷分替代核心贡献，也不重启无休止推力拟合。
<!-- decision169-results:end -->
''')
    checks = dict(utc=now, local_html_links_checked=len(links.links), missing_links=missing,
                  exported_jsonl_rows=lines, final_source_hashes_match=True,
                  prediction_fitting_rerun=False, prediction_values_unchanged=True,
                  force_success_claim=False, FDE_or_actual_occlusion_claim_from_geometry=False,
                  checkpoint_backup=str(backup), handoff=handoff, default_promoted=False)
    (OUT / 'artifact_checks169.json').write_text(json.dumps(checks, indent=2), 'utf-8')
    print(json.dumps(checks, ensure_ascii=False))
