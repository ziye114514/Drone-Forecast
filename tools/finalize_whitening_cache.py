"""Record the bounded, exact v7 CPU improvement and current handoff."""
from pathlib import Path
from datetime import datetime, timezone
import json

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
OUT=WORK/'output/flow_acceleration_20261026'
result=json.loads((OUT/'whitening_cache/comparison.json').read_text('utf-8'))
assert result['all_fixed_sequence_outputs_exact']
assert sum(result['exact_coefficients_and_audits_by_case'].values())==481
note='''

### 累计v7小优化与交接澄清（2026-10-26）

按用户要求继续积累小收益，不要求每条录像均改善；完整组合相对当前累计版评价，原版仅作冻结对照。当前默认已经是v7，而非早期原算法。v2/v3/v6保留可选，v6重叠准备已随v7完整组合进入默认，不直接相加独立像素收益。

本批将每条合格光流观测的协方差白化矩阵计算后复用，避免每次窗口拟合重复分解求逆。保留外部三元组history兼容。与默认无视频v7源快照对照，side86、train2 99、headless95、test3 201共481时刻系数与完整诊断逐项完全相同；两方法非线性调用均35。三轮交替组件CPU测试各1443次，中位0.295300→0.248900ms，P95 0.621190→0.599700ms。此为约0.0464ms组件开销节省，不声称新增精度或整条流水线加速；没有启动RAW/GPU、没有标签读取。结果output/flow_acceleration_20261026/whitening_cache/comparison.json。保留当前v7版本名与全部基线。

交接旧顶层in_progress/default_promoted及quota_policy的早期暂停读数过时，已统一当前默认、当前额度和无运行任务；历史阶段结果保留。下一步改善变向时当前速度与加速度的联合估计，先检查光流中点时间补偿与不确定性，避免反复扫窗口和重跑整套验证。
'''
with (WORK/'WORKLOG.md').open('a',encoding='utf-8') as f:f.write(note)
with (ROOT/'docs/ACCUMULATED_MAINLINE.md').open('a',encoding='utf-8') as f:f.write(note)
page=OUT/'index.html'
html=page.read_text('utf-8')
html=html.replace('</html>','<h2>v7继续累积：白化矩阵复用</h2><p>481时刻系数与完整诊断精确一致，组件CPU中位0.295300→0.248900毫秒，约省0.0464毫秒。没有新增精度或整体耗时主张；没有启动RAW/GPU。</p><p><a href="whitening_cache/comparison.json">固定输入对照与组件耗时</a></p></html>')
page.write_text(html,encoding='utf-8')
p=WORK/'output/checkpoints/latest_handoff.json'
d=json.loads(p.read_text('utf-8-sig'))
d.update(utc=datetime.now(timezone.utc).isoformat(),
    execution_status='active goal; cumulative v7 includes exact whitening cache',
    in_progress='No jobs running; v7 cumulative default retained with exact CPU improvement',
    experiments_running=False,default_promoted=True,current_jobs=[],
    latest_live_revision='accumulated_v7_flow_acceleration',
    latest_component_optimization=str(OUT/'whitening_cache/comparison.json'),
    quota_remaining=dict(short=12,weekly=47),
    next_action='Check both quota windows before next short batch; save and pause at10% remaining. Continue from cumulative v7. Whitening cache is exact on481 observations. Consider joint current-velocity midpoint compensation only with proper uncertainty; no sweeps or full validation reruns. Original and prior recipes remain selectable.')
d['quota_policy'].update(last_observed_primary_used_percent=88,last_observed_primary_remaining_percent=12,
    last_observed_weekly_used_percent=53,resume_requires_scheduled_check=False,automation_id=None,
    reason_for_early_wind_down='Save handoff headroom near user-authorized10% threshold')
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(saved=True,exact_ticks=481,current_jobs=[],default=d['latest_live_revision'])))
