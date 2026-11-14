"""Retain the measured overlap tradeoff without claiming an accuracy gain."""
from pathlib import Path
import argparse,ast,datetime,json
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/support_overlap_20261026'
parser=argparse.ArgumentParser();parser.add_argument('--short-remaining',type=int,required=True);parser.add_argument('--weekly-remaining',type=int,required=True);args=parser.parse_args()
assert not (OUT/'retention_decision.json').exists()
stage=json.loads((OUT/'comparison.json').read_text('utf-8'));actual=json.loads((OUT/'actual_comparison.json').read_text('utf-8'));common=json.loads((OUT/'common_origin_comparison.json').read_text('utf-8'))
path=ROOT/'configs/live_tracks.json';manifest=json.loads(path.read_text('utf-8'));assert manifest['recommended']=='accumulated_v5_fast_ci'
manifest['tracks']['accumulated_v6_support_overlap'].update(status='retained_measured_latency_accuracy_tradeoff_not_default',evidence=str(OUT/'retention_decision.json'))
path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
a,b=actual['results']['serial']['runtime'],actual['results']['gpu-overlap']['runtime'];c=common['tables']['0.4']
decision=dict(variant='accumulated_v6_support_overlap',retained_as_selectable_variant=True,recommended='accumulated_v5_fast_ci',
    automatic_common_origin_accuracy_pass=not common['guard_failures'],guard_failures_unchanged=common['guard_failures'],
    GPU_stage_median_saved_ms=stage['summary']['serial']['median_ms']-stage['summary']['gpu-overlap']['median_ms'],
    actual_processing_median_saved_ms=a['processing_median_ms']-b['processing_median_ms'],
    actual_observation_age_p95_saved_ms=a['observation_age_p95_ms']-b['observation_age_p95_ms'],
    same_valid_publications=True,new_accuracy_gain_claimed=False,
    reason='Keep useful phase-latency tradeoff; one4s pair processing median flat and common-origin mean slightly worse, so do not claim an overall prediction improvement or replace v5.')
(OUT/'retention_decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(11,3.7))
metrics=[('Matched-image stage',stage['summary']['serial']['median_ms'],stage['summary']['gpu-overlap']['median_ms']),
    ('Actual processing median',a['processing_median_ms'],b['processing_median_ms']),('Actual observation age P95',a['observation_age_p95_ms'],b['observation_age_p95_ms'])]
for ax,(title,x,y) in zip(axes,metrics):
    ax.bar([0,1],[x,y],color=['#377eb8','#e17c05']);ax.set_xticks([0,1],['Serial','Overlap']);ax.set_title(title,fontsize=10);ax.set_ylabel('ms');ax.set_ylim(0,max(x,y)*1.22)
    for i,v in enumerate([x,y]):ax.text(i,v+max(x,y)*.025,f'{v:.2f}',ha='center')
fig.tight_layout();fig.savefig(OUT/'latency_comparison.png',dpi=160);plt.close(fig)
diagnostic=json.loads((OUT/'next_accuracy_diagnostic.json').read_text('utf-8'))
rows=''.join(f'<tr><td>{name}</td><td>{s["valid_detected_ticks"]}</td><td>{s["position_acceleration_eligible"]}</td><td>{s["nonzero_acceleration"]}</td></tr>' for name,s in diagnostic.items())
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>支持图与GPU重叠：可切换的时序取舍</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 20px;color:#243040}}h1,h2{{line-height:1.3}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #d4dee8;padding:9px}}.note{{background:#fff3d9;padding:15px}}code{{word-break:break-all}}</style>
<h1>支持图准备与GPU推理重叠</h1><p class="note">实现已接入，可用 <code>--live-revision accumulated_v6_support_overlap</code> 切换。阶段等待减少、这组RAW的观测年龄P95更低；整条处理耗时中位基本持平，共同起点的平均误差略增。收益与取舍完整保留，默认继续累计v5。</p>
<h2>改动很直接</h2><p>发送当前GPU请求后，主进程先生成该帧纯ON/OFF支持图，再接收GPU回复。两个进程只读同一不可变当前画面，没有下一帧、延迟补标或新预测公式。增加 <code>--flow-preparation serial|gpu-overlap</code> 明确记录顺序；关闭局部位移时自动回到串行准备，原版入口保留。</p>
<h2>短试验的真实结果</h2><img src="latency_comparison.png" alt="阶段中位等待降低，但整条状态处理的中位耗时持平；本组观测年龄P95下降">
<p>9张原生画面、3轮交替配对，共54次GPU请求：检测框及支持图哈希全部一致，阶段中位15.636 → 12.248ms。随后仅做一组同片段4秒RAW对照，均97次跟踪、20次发布/18次有效。整体处理36.916 → 37.104ms，没有测出整体中位加速；观测年龄P95122.695 → 84.204ms。它们是单组实测值，不是长期性能或显著性证明。</p>
<h2>没有隐藏平均误差回退</h2><p>两次实际执行接收的观测和发布时刻不同。按相同发布槽位，将已发布状态对齐到两者较晚的源时间，再预测完全相同的未来目标；所用状态均已在共同起点前可用，没有选择更新或未来观测。这是虚拟共同起点公平对比，不冒充两次原始实时截止时间。</p>
<table><tr><th>固定0.4s，16个目标</th><th>平均 px</th><th>P95 px</th></tr><tr><td>串行累计v5</td><td>{c['serial']['mean']:.3f}</td><td>{c['serial']['p95']:.3f}</td></tr><tr><td>重叠版本</td><td>{c['overlap']['mean']:.3f}</td><td>{c['overlap']['p95']:.3f}</td></tr></table>
<p>平均增加 {c['overlap']['mean']-c['serial']['mean']:.3f}px，P95减少 {c['serial']['p95']-c['overlap']['p95']:.3f}px。原工程判据略超均值允许线，失败项保留原值，没有改阈值凑通过。保留为有用时序取舍版本，不因处理均值持平而删除，也不称为新的精度提升。</p>
<h2>下一处精度卡点</h2><p>检查此前累计v4的实际当前观测：侧视与train2的加速度经常为零。局部位移已经补充了速度，下一步优先利用原始光流速度在过去时段内的变化，估计有不确定性约束的加速度，再与当前只依靠框中心变化的加速度比较。先复用现有鲁棒拟合和2.1MB精确支持缓存，不再扫并行顺序或权重参数。</p>
<table><tr><th>片段</th><th>有效且当前检出的时刻</th><th>位置加速度资格通过</th><th>实际非零加速度</th></tr>{rows}</table>
<p>这仍是图像空间的运动量，不等同于推力或真实三维加速度。</p>
<p>证据：<a href="comparison.json">同画面阶段测量</a> · <a href="actual_comparison.json">实际流水线与逐步回放</a> · <a href="common_origin_comparison.json">共同起点完整评分</a> · <a href="retention_decision.json">保留决定与失败项</a> · <a href="next_accuracy_diagnostic.json">下一步精度诊断</a> · <a href="../fast_ci_20261026/index.html">当前默认v5</a></p></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
files=['forecast.py','run_live_forecast.py','configs/live_tracks.json','tools/probe_support_overlap.py','tools/register_support_overlap.py','tools/summarize_support_overlap.py','tools/score_overlap_common_origins.py','tools/record_support_overlap.py']
for name in files:
    source=ROOT/name;data=source.read_bytes()
    if source.suffix=='.py':ast.parse(data)
    dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
hpath=WORK/'output/checkpoints/latest_handoff.json';h=json.loads(hpath.read_text('utf-8'))
h.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_status='active goal; v5 recommended, v6 overlap retained as latency/accuracy tradeoff',
    latest_report=str(OUT/'index.html'),latest_live_revision='accumulated_v5_fast_ci',available_tradeoff_revision='accumulated_v6_support_overlap',
    current_jobs=[],current_goal_turn_classification='progress',quota_remaining=dict(short=args.short_remaining,weekly=args.weekly_remaining),
    next_action='Focus next on accuracy, not more overlap timing. Use cached spatial_flow_20261026 observations and original current bodies/publications. Raw flow velocity at midpoint (before acceleration compensation) is new first-derivative evidence. Inspect/reuse existing robust fitter; fixed .35s window/min6/span.16 conventions, no window or lambda sweep. Fit velocity derivative with uncertainty and compare a position-consistency-qualified variant, preserving v5 p/v and eligibility. Side valid86 only34 nonzero a; train2 valid32 only1. v6 is selectable, not default; common04 mean56.944→59.819 (gate false) but tail115.473→106.957 and ageP95122.695→84.204. All known jobs terminal; do not repeat these trials. Check quota and pause at10%.')
hpath.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')
(WORK/'output/latest_live_forecast.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=support_overlap_20261026/index.html"><a href="support_overlap_20261026/index.html">最新优化报告</a>',encoding='utf-8')
with (ROOT/'docs/ACCUMULATED_MAINLINE.md').open('a',encoding='utf-8') as f:
    f.write('\n\n## 可选v6：支持图与GPU重叠\n\n`--live-revision accumulated_v6_support_overlap`。固定同画面阶段15.636→12.248ms，NN框/支持哈希相同。4秒实际串行/重叠各97跟踪、18/20有效；处理36.916→37.104ms持平，ageP95122.695→84.204。共同起点.4s均56.944→59.819、P95115.473→106.957，均值工程线略超保留false。保留为时序取舍，默认v5。\n')
with (WORK/'WORKLOG.md').open('a',encoding='utf-8') as f:
    f.write(f'\n\n### 支持图与GPU重叠试验完成（2026-10-26）\n\n前一turn为progress（v5等价CI加速与缓存）。本轮新增--flow-preparation串行/重叠，将支持图放在GPU请求flush后、receive前；只读当前不可变画面。9原生图×3交替配对，54请求，NN框/支持哈希全相同，阶段15.636→12.248ms。\n\n一组4秒RAW对照都97tracking、20pub/18valid；处理中位36.916→37.104ms持平，ageP95122.695→84.204；两模式均使用相同不可变校验开销。所有GPU/原生/producer哈希匹配，97+97状态和发布逐步精确回放，无缓存标签进入推理。\n\n另做共同起点公平评分：各槽用已经发布的快照，在两次较晚源时间为共同起点，不用更新/未来观测；.4s16目标均56.944201→59.818684，P95115.472895→106.956816。均值增2.874482略超2.847210工程允许线，失败原值保留不改门槛。v6保留为可切换时序/尾误差取舍，默认仍累计v5，不称新精度收益。\n\n精度下一卡点：side有效且当前检出86时刻，57位置a资格/34非零a，中位a0；train2 32中3资格/1非零a，中位0；test3 181中174资格/116非零a。下一步用未补偿raw flow v(midpoint)拟合一阶导数a及不确定性，复用原有.35s/6点/.16s标准，不扫窗口/权重；与位置一致性资格组合比较，保持p/v。直接用缓存，不再重复解码/测速。\n\n会话22306(配对GPU)、95235(串行RAW)、14129(重叠RAW)均exit0；当前无运行作业，goal active。报告output/support_overlap_20261026/index.html；quota余短{args.short_remaining}%/周{args.weekly_remaining}%。\n')
print(json.dumps(decision,ensure_ascii=False),flush=True)
