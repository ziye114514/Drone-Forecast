"""Save causal midpoint mechanism results without promoting incomplete uncertainty."""
from pathlib import Path
import json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
OUT=WORK/'output/midpoint_compensation_20261026'
p=OUT/'comparison.json';d=json.loads(p.read_text('utf-8'))
for value in d['results'].values():
    value.pop('current_p_v_validity_unchanged',None)
    value.update(current_position_validity_unchanged=True,midpoint_variant_velocity_changed=True)
p.write_text(json.dumps(d,indent=2),encoding='utf-8')
note='''

### 当前速度中点补偿固定机制试验（2026-10-26）

上一turn为progress：v7白化复用已合入。此批按原固定395观测/相同发布重建，先确认旧状态精确，再只在新a资格通过且CI选择flow端点时，用raw v(midpoint)+(t-midpoint)*a_new替换当前rollout速度。位置、有效性不变，不使用未来观测，不扫补偿权重。推理状态生成与标签评分分离，没有RAW/GPU。

固定0.4s开发宏平均72.485959→72.701450（略差0.215491px）；side16目标均56.724562→57.155544，但P95100.644786→96.736310（省3.908476px）；train2六目标保持88.247356。另test3十三变时域49.899394→49.866781，P95115.721467→115.043280，不混池，不称全局盲测。side57、test3 121次更新，未改变有效输出。

方案与代码保留为可复现实验。当前只是均值时间补偿，尚未传播raw v与拟合a的共同协方差，不将此半成品接默认；暂不默认的原因是联合不确定性缺失，不是收益太小。下一步在拟合中保留截距/斜率完整协方差及当前时刻预测协方差，处理共享观测相关性，再用相同固定方案做完整组合比较，不扫窗口/权重。默认仍累计v7含白化缓存。结果output/midpoint_compensation_20261026/comparison.json；过程无后台任务。
'''
with (WORK/'WORKLOG.md').open('a',encoding='utf-8') as f:f.write(note)
html='''<!doctype html><meta charset="utf-8"><title>速度中点补偿小试验</title><style>body{font:18px sans-serif;max-width:960px;margin:40px auto;line-height:1.7}table{border-collapse:collapse}td,th{padding:8px 16px;border:1px solid #ccc}</style><h1>速度中点补偿：保留的取舍方案</h1><p>以已通过位置一致性的新加速度，把原始光流中点速度换算到当前时刻。固定方案，无参数扫描，无新RAW/GPU。</p><table><tr><th>指标</th><th>当前v7机制</th><th>加新时间补偿</th></tr><tr><td>开发固定0.4秒等权均误差</td><td>72.486 px</td><td>72.701 px</td></tr><tr><td>侧视P95（16目标）</td><td>100.645 px</td><td>96.736 px</td></tr><tr><td>test3均误差（13变时域目标）</td><td>49.899 px</td><td>49.867 px</td></tr></table><p>位置及有效性不变，速度改变。方案保留，默认仍v7。尚未传播速度与加速度的共同协方差；不是完整部署算法或新实时精度证据。历史样本已暴露，不是全局盲测。</p><p><a href="comparison.json">完整结果</a> · <a href="../flow_acceleration_20261026/index.html">当前默认报告</a></p>'''
(OUT/'index.html').write_text(html,encoding='utf-8')
p=WORK/'output/checkpoints/latest_handoff.json';h=json.loads(p.read_text('utf-8-sig'))
h.update(utc=datetime.now(timezone.utc).isoformat(),latest_mechanism_probe=str(OUT/'index.html'),
    current_jobs=[],experiments_running=False,
    next_action='Check quota; pause at10% as authorized. Keep cumulative v7 default including cached whitening. Midpoint compensation retained as mean-only experimental tradeoff, not rejected for small gains: devmean+0.2155px, sideP95-3.9085px. Next implement full intercept/slope covariance and shared-observation velocity/acceleration uncertainty before runtime promotion; same fixed clips, no sweeps.',
    quota_remaining=dict(short=11,weekly=46))
h['quota_policy'].update(last_observed_primary_used_percent=89,last_observed_primary_remaining_percent=11,last_observed_weekly_used_percent=54)
p.write_text(json.dumps(h,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(saved=True,background_jobs=[],production_promoted=False)))
