"""Save this completed batch and the next isolated phase-module investigation."""
from pathlib import Path
import json,datetime
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'output/three_continuation_20261026'
source=out/'author_phase_arc_init';log=source/'observations.jsonl'
rows=list(map(json.loads,log.read_text('utf-8').splitlines())) if log.exists() else []
failure=dict(status='isolated_module_process_access_violation',exit_code=-1073741819,
    saved_complete_rows=len(rows),last_saved_t=rows[-1]['t'] if rows else None,
    production_default_changed=False,
    next='Inspect author-core event bounds and concurrency. One serial-core diagnostic before any integration; no RPM/thrust evidence from this failure.')
(source/'failure.json').write_text(json.dumps(failure,indent=2),'utf-8')
path=ROOT/'output/checkpoints/latest_handoff.json';d=json.loads(path.read_text('utf-8'))
summary=json.loads((out/'summary.json').read_text('utf-8'))
d.update(latest_structure_report=str(out/'index.html'),latest_structure_artifact_checks=str(out/'artifact_checks.json'),
    latest_progress_checkpoint=str(out/'completed_handoff.json'),current_jobs=[],current_running_jobs=[],
    quota_pause_requested=False,current_work_state='active; completed accumulated report; next isolated author-phase bridge investigation',
    core_route_completion='Current short-arc normal constraints, qualified3 guidance, first-matured-row disturbance,120ms acceleration hold and exact CPU reductions integrated; physics still worse than offline kinematics.',
    three_continuation_progress=dict(completed=True,report=str(out/'index.html'),summary=str(out/'summary.json'),
        rate_definitions=summary['rate_definitions'],same_input_three_events=[0,2],three_span_s=.099999,
        normal_updates=7,current_same_input_side_13target04px=summary['current_side_accuracy']['scores']['0.4'],
        original_position_core7_unchanged=True,videos_decoded=[60,90],browser_preview='VP8/WebM',
        stable30_proven=False,physics_advantage_proven=False,
        strict_gain_quality='optional, no prediction change in this pilot',coupled_observer='negative, not deployed',
        author_phase_complete_init=0,author_phase_qualified_arc_init=failure,no_current_jobs=True),
    updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
path.write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf-8')
(out/'completed_handoff.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf-8')
with (ROOT/'WORKLOG.md').open('a',encoding='utf-8') as f:
    f.write('\n\n### 本轮累计改进完成与相位复用探针（2026-10-27）\n'
        '报告output/three_continuation_20261026/index.html与summary.json完成；60/90帧MP4与VP8网页预览逐帧解码通过，19Python语法/12本地链接检查通过，原位置核心7文件字节保持。README和docs/THREE_CONTINUATION.md已更新。当前clear/side计算能力30.784/36.078FPS，源更新28.871/29.845Hz，成功整机观测26.940/26.858Hz；仍非持续低延迟30FPS。\n'
        '最新side同13目标0.4s旧物理53.899→累计52.718px，median56.601→45.006，P9585.371→89.447变差；离线传统35.929仍好。不得与旧14目标绝对分数混比。2sigma增益代理资格本段预测不变，保留可选不默认；耦合观察器同12目标均54.470→56.268、P9590.833→146.525，实验保留不部署。\n'
        '复用现有本地HelixTrack作者核心做独立周期模块试验：严格完整椭圆接口60时刻零初始化，新四点实际来自合格残弧。显式残弧形状初始化臂发生access violation（exit -1073741819），failure.json和部分因果行已保存，未改作者C++核心、未进入默认预测。下一步先检查边界/串行核心稳定性，不能将失败结果算RPM/推力证据。全部已知作业terminal。\n'
        '任务保持继续状态，用户低于10%只看时钟的政策保留；00:11单次heartbeat已核验，触发时若本对话已有工作就接续，避免重复启动。\n')
print(json.dumps(failure))
