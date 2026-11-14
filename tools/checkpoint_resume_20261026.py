"""Save the resumed session's measured evidence without replacing history."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, html, json

ROOT=Path(__file__).resolve().parents[1]; WORK=ROOT.parent; OUTPUT=WORK/'output'
OUT=OUTPUT/'optimization_resume_20261026'; OUT.mkdir(exist_ok=True)
def read(name): return json.loads((OUTPUT/name).read_text('utf-8'))
def write(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
ca=read('live_CA_velocity_20261026/comparison.json')
scale=read('live_scale_rollout_20261026/comparison.json')
raw=read('live_scale_rollout_RAW_20261026/scores.json')
pixel=read('fixed_pixel_periodicity_sdk_20261026/comparison.json')
live={name:read(f'fixed_pixel_periodicity_RAW_20261026/{name}/completion.json') for name in ['full','quarter']}
checks={}
for name,c in live.items():
    directory=OUTPUT/'fixed_pixel_periodicity_RAW_20261026'/name
    rows=[json.loads(s) for s in (directory/'tracking.jsonl').read_text('utf-8').splitlines()]
    pubs=[json.loads(s) for s in (directory/'predictions.jsonl').read_text('utf-8').splitlines()]
    assert len(rows)==c['tracking_updates'] and len(pubs)==c['publication_slots']
    assert sum(r['valid'] for r in pubs)==c['valid_publications']
    assert c['sources_unchanged'] and c['frame_descriptor_mismatches']==0
    for r in rows:
        assert r['producer_frame_sha256']==r['detector_frame_sha256']==r['gpu_frame_sha256']
        assert r['max_event_t'] is None or r['max_event_t']<round(r['t']*1e6)
    checks[name]=dict(source_unchanged=True,three_frame_hashes_equal=True,native_cutoffs_pass=True,
        periodic_qualified=sum(bool((r.get('periodic_axis') or {}).get('qualified')) for r in rows),
        unsigned_axis_qualified=sum(bool((r.get('periodic_axis') or {}).get('axis_valid')) for r in rows))
assert live['full']['native_events']==live['quarter']['native_events']==16981921
assert live['full']['native_batches']==live['quarter']['native_batches']==65
summary=dict(utc=datetime.now(timezone.utc).isoformat(),
    scope='User-authorized single scheduled resume after account reset; known clips, small mechanism experiments.',
    goal_controller='Last verified paused; user asked to resume with goal UI. Current scheduled turn is working, not a resumed controller claim.',
    quota_last_read=dict(primary_remaining_percent=61,weekly_remaining_percent=69),
    CA_velocity=dict(adopted=False,reason='Both primary .4s comparisons worsened',evidence='../live_CA_velocity_20261026/comparison.json'),
    scale_rollout=dict(optional=True,default_promoted=False,frozen_gate_passed=scale['provisional_gate_passed'],
        actual_RAW_scores=raw['results'],evidence='../live_scale_rollout_RAW_20261026/scores.json'),
    PVTpp=dict(commit='d71cad0ed0bfede979469088704f6ba1d5a6f38c',model_executed=False,
        official_motion_weights='MEGA API -16 EBlocked; no bypass or random-weight reproduction',local_reference_only=True),
    reader_index=dict(optional=True,default_changed=False,all205_native_slice_fields_exact=True,steady_speedup_proven=False),
    stale_work_skip=dict(enabled_live=True,maximum_age_s=.16,skips_only_frame_generation_and_transport=True,
        all_native_noise_and_periodic_updates_preserved=True,observed_full_run_skips=live['full']['stale_before_generation']),
    periodic_pixel_sampling=dict(optional=True,default_promoted=False,SDK_mask_reused=True,
        mask_sha256=pixel['mask_sha256'],selected_events=pixel['selected_events'],
        exact_selected_frequency_records=pixel['exact_selected_frequency_records'],
        feed_ms=pixel['feed_ms'],actual_live=live,checks=checks,
        causal_forecast_feature=False,mechanical_RPM_or_force_estimated=False,
        timing_limit='Single ordered runs; different accepted input cohorts, not a controlled precision or hardware speed comparison.'),
    original_position_core_changed=False,old_archives_are_historical=True,
    stable_high_accuracy_real_time_proven=False,running_jobs=[],
    next_action='Probe current periodic evidence for short NN gaps; require past-only calibration and fresh support, no prediction-as-measurement. Stop at either quota remaining<=10%.')
write(OUT/'summary.json',summary)
rows=[]
for name in ['side_v2','train2']:
    r=raw['results'][name]; s=r['scores']['0.4']; m=s['metrics']
    rows.append(f"<tr><td>{name}</td><td>{m['unmodified_motion']['n']}</td><td>{m['unmodified_motion']['mean']:.3f}</td><td>{m['live_motion']['mean']:.3f}</td><td>{s['changed']}</td></tr>")
sampling=[]
for name,c in live.items():
    sampling.append(f"<tr><td>{name}</td><td>{c['tracking_updates']}</td><td>{c['valid_publications']}/{c['publication_slots']}</td><td>{c['observation_age_p95_ms']:.1f}</td><td>{c['producer_profile']['periodic']['p95_ms']:.2f}</td></tr>")
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>恢复优化：尺度修正与周期观测减负</title><style>body{max-width:1050px;margin:40px auto;padding:0 22px;font:17px/1.7 system-ui;color:#1b263b}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4df;padding:8px;text-align:left}aside{background:#eef4fa;padding:16px}a{color:#1166ac}</style><h1>恢复优化：已经改了什么</h1><aside>已按授权在额度重置后恢复当前工作。以下是已完成的小实验；原位置核心、研究基线保留。新模型还不能估计真实推力，不能称持续高精度实时预测。</aside><h2>尺度变化：可选的条件修正</h2><p>仅在历史框的宽、高变化一致且观测完整时，根据透视尺度变化修正高阶位移。保持当前的位置、速度、加速度和输出资格不变。固定形状是假设；姿态变化也会改变框尺寸。</p><p>实际 RAW、同一个有效发布时间、同一个目标时刻，0.4 秒平均中心误差（像素）：</p><table><tr><th>片段</th><th>共同目标数</th><th>原方法</th><th>条件尺度修正</th><th>改变的目标数</th></tr>'''+''.join(rows)+'''</table><p>侧视有小收益，train2 未触发修正。样本少且片段已用于开发，不代表泛化结果。首次侧视运行发生约 3 秒中途读取停顿，有效输出仅 5/20，已保留，不能用后一次 18/20 全归因于算法。</p><h2>周期观测：保留选中像素的全部时间信息</h2><p>复用 SDK RoiMaskAlgorithm，只对固定约四分之一原生像素运行周期算法；检测图像仍接收完整事件。16,981,921 原生事件中选中 4,227,827 个，759,255 个选中周期记录与完整流逐字段一致。空间覆盖减少，因此整幅图的可观测性不能称相同。</p><p>SDK 探针中周期处理 P95 从 17.93 降到 12.00 ms。NumPy 筛选版本反而更慢，不采用。实际 RAW 短试：</p><table><tr><th>周期输入</th><th>跟踪次数</th><th>有效/发布</th><th>全部观测年龄P95 ms</th><th>周期处理P95 ms</th></tr>'''+''.join(sampling)+'''</table><p>两次各 1.2 秒，原生事件总数相同，但实时接受的时刻集合不同；这是接线与负载证据，不能把 3→4 次有效输出归为稳定精度收益。周期支路仍只是观测，不控制运动或推力。</p><h2>减少已经过期的无用工作</h2><p>每个原生批次仍更新噪声和周期状态；若其时间已经超过原有 160 ms 新鲜度上限，在生成图像和进程传输前停止该项工作。实际完整周期试验跳过 9 次过期生成。SDK 原生索引关闭选项的 205 切片逐字段一致，但未证明稳态速度收益，默认不变。</p><h2>明确停止的方向</h2><p>常加速度状态只替换速度：侧视 0.4 秒 51.089→55.007 px，train2 73.856→76.746 px，拒绝接入，不扫参数。PVT++ 已查作者源及固定提交，但官方运动权重链接返回 MEGA -16（封禁），未运行模型。EVtracker 的公开仓库目前只有 README，没有可直接复用的实现。</p><h2>接下来处理漏检空档</h2><p>尝试用当前周期事件作为短期弱观测。必须先用过去的 NN 框标定事件中心偏差，漏检时仍要求新事件支持，不把惯性推算当检测结果。不在没有独立证据时假定四桨、转速或推力。</p><p><a href="summary.json">机器摘要</a> · <a href="../live_scale_rollout_RAW_20261026/scores.json">RAW配对误差</a> · <a href="../fixed_pixel_periodicity_sdk_20261026/comparison.json">原生掩码证据</a> · <a href="../live_CA_velocity_20261026/comparison.json">拒绝的速度候选</a></p><p>旧 0.3.1 wheel 仅为位置核心；以前的源码 ZIP 均是历史快照，不包含本轮全部修改。新的实际周期运行保存精确 source_snapshot。没有上传 GitHub。</p><p>参考：<a href="https://github.com/Jaraxxus-Me/PVT_pp">PVT++ 作者仓库</a>；<a href="https://github.com/yexuezhaoge/EVtracker">EVtracker 作者仓库</a>；<a href="https://docs.prophesee.ai/stable/api/python/core/bindings.html">SDK像素掩码</a>；<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html">Theil–Sen估计</a>。</p></html>'''
(OUT/'index.html').write_text(page,encoding='utf-8')
handoff=OUTPUT/'checkpoints/latest_handoff.json'; data=json.loads(handoff.read_text('utf-8'))
data.update(utc=summary['utc'],execution_status='Current authorized scheduled resume working; controller last verified paused, user resume requested',
    in_progress=summary['next_action'],experiments_running=False,latest_report=str(OUT/'index.html'),next_action=summary['next_action'],
    resume_20261026=summary)
write(handoff,data)
log=WORK/'WORKLOG.md'; marker='### 额度恢复后的实际优化（2026-10-26）'
if marker not in log.read_text('utf-8'):
    with log.open('a',encoding='utf-8') as f:
        f.write('\n\n'+marker+'\n\n账户重置已核验，当前授权自动化turn继续工作；目标控件最后仍paused，已请用户手动恢复，不伪称控制器active。新报告output/optimization_resume_20261026/index.html及summary.json。每短批查额度，任一窗口余量<=10%保存后停。\n\n常加速度仅换速度固定候选失败：主独立时钟side .4s51.089→55.007、train2 73.856→76.746，拒绝且不追加RAW。PVT++已固定作者提交并核验源文件，官方运动权重MEGA -16封禁，未运行，不绕过、不假称复现。EVtracker公开仓库仅README。\n\n新增可选scale-rollout compatible，保持当前状态和资格，宽高变化兼容才给高阶透视修正。实际RAW side_v2同发布时间.4s n16 49.930→48.982px，train2 n6 88.684不变；已暴露的小样本，不升默认。首次side中途读取停顿约3s、5/20有效保留；side_v2 18/20不能全归因代码。\n\nreader-index disabled复用SDK公开HAL配置，205切片x/y/p/t全部一致，但未证明稳态速度收益。live生产器在保留全部原生噪声/周期更新后，跳过已超过160ms的图像生成/传输，实际full周期运行跳9次。没有改SDK。\n\n新增可选periodic-sampling fixed-quarter复用SDK RoiMaskAlgorithm，固定seed141000，选230233像素；16981921原始事件选4227827，759255选中周期记录逐字段等于完整流。NumPy筛选较慢不用；SDK探针P95 17.928→11.997ms。实际full/quarter两次1.2s：20/30跟踪、6/6发布、3/4有效、周期P9520.528/12.014ms。时刻集合不同，非受控测速或精度证明。周期仍只观测，无RPM/推力反馈。\n\n原7文件位置核心未改；新周期RAW保存source_snapshot。旧wheel及各ZIP是历史，未重打包或上传。全部已知RAW会话终止，无后台任务。下一步只试当前事件证据短暂补NN漏检，需过去偏差标定且当前新支持，不能用惯性预测冒充观测。\n')
old=OUTPUT/'optimization_run_20261025/index.html'
oldtext=old.read_text('utf-8'); link='optimization_resume_20261026/index.html'
if link not in oldtext: old.write_text(oldtext.replace('<body>','<body><aside style="padding:16px;background:#e9f4ff">额度恢复后新增：<a href="../'+link+'">尺度修正、周期观测减负及当前结果</a></aside>',1),encoding='utf-8')
print(json.dumps(dict(report=str(OUT/'index.html'),checks=checks,checkpoint_updated=True),ensure_ascii=False))
