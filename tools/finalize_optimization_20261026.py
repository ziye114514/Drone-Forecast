"""Measured report and handoff for the one-time quota-resume session."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,html,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUTPUT=WORK/'output'
parser=argparse.ArgumentParser();parser.add_argument('--primary-remaining',type=int,required=True)
parser.add_argument('--weekly-remaining',type=int,required=True);parser.add_argument('--quota-paused',action='store_true');args=parser.parse_args()
OUT=OUTPUT/'optimization_resume_20261026';OUT.mkdir(exist_ok=True)
def read(name):return json.loads((OUTPUT/name).read_text('utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
single=read('single_model_optimized_RAW_20261026/scores.json')
extension=read('heldout_motion_RAW_20261026/scores.json')
merged=read('equal_weight_detector_20261026/export.json')
fine=read('roi_augmented_detector_20261026/export.json')
transfer=read('equal_weight_detector_20261026/transfer_v2/scores.json')
half=read('mixed_fp16_detector_v3_20261026/comparison.json')
packet=read('periodic_packet_20261026/comparison.json')
dual=read('center_ensemble_RAW_20261026/scores.json')
train=read('roi_augmented_detector_20261026/train_completion.json')
assert not single['default_promoted'] and not extension['default_promoted']
cfg=json.loads((ROOT/'configs/local.json').read_text('utf-8'))
assert cfg['detector_sha256']=='e2531556ad0431d904fb0d1d432727a2e2dd9eaed9e098a6ed3d9e55f474db0b'
core=ROOT/'src/event_drone_forecast/position';old=OUTPUT/'live_runtime_source_20261025_startup/src/event_drone_forecast/position'
assert len(list(core.glob('*.py')))==7
assert all((old/p.name).read_bytes()==p.read_bytes() for p in core.glob('*.py'))
status='quota_paused' if args.quota_paused else 'completed_batches_checkpoint; current authorized turn working'
summary=dict(utc=datetime.now(timezone.utc).isoformat(),execution_status=status,research_objective_complete=False,
    usage_last_verified=dict(primary_remaining_percent=args.primary_remaining,weekly_remaining_percent=args.weekly_remaining),
    goal_controller_last_verified='paused; API has no resume operation; authorized heartbeat work ran in this turn',
    automation=dict(id='automation',one_time_resume_executed=True,deleted=True,tool_delete_status='deleted'),
    running_jobs=[],baseline=dict(original_configuration_unchanged=True,all7_position_core_files_exact=True,
        research_default='138+140+144, optional156 retained',live_default_model_sha256=cfg['detector_sha256']),
    candidate=dict(kind='Single equal-weight merged detector with conditional scale rollout',model=merged,
        inference_observations='RAW only; static model/config allowed; no saved detections, trajectories or labels fed to actual inference',
        actual_results=single['results'],heldout_extension=extension,
        default_promoted=False,reason='Small development mean gains, inconsistent tail errors, heldout motion mean slightly worse'),
    detector_transfer=transfer,training=train,
    mixed_precision=dict(model=half['model'],sha256=half['sha256'],bytes=half['mixed_bytes'],source_bytes=half['source_bytes'],
        maximum_center_shift_px=half['maximum_center_shift_px'],same_box_counts=half['same_box_counts'],forward=half['forward'],
        probe_only=True,full_RAW_or_future_accuracy_verified=False,cuDNN_frontend_errors_logged=True,default_promoted=False),
    periodic_transport=dict(quarter_sampling_optional=True,compaction_optional=True,
        semantic_equality_observations=packet['compared_current_observations'],packet_byte_reduction=packet['reduction_fraction'],
        actual_full_bytes=6929712,actual_compacted_bytes=1040616,actual_latency_gain_proven=False,force_feedback=False),
    rejected=['CA velocity worsened both primary cases','Periodic line gap recovery0 centers','60ms detector input0 extra detections',
        'Direct turn fit0 supported rates','Direct cropped centers worsened forecast','Full fine-tune failed short-horizon gate',
        'Dual detector incurred extra inference time; no standalone whole-system superiority proven'],
    official_reference_limitations=dict(PVTpp='Fixed author sources; official motion-weight link MEGA -16 blocked; model not executed',
        EVtracker='Public repository only README at inspection',DroneStalker='Prediction source reviewed; motion weights not downloaded or executed'),
    annotation=dict(directory=str(OUTPUT/'dense_labels_train3_transfer_20261026'),native_frames=80,fps=10,window_us=5000,
        prediction_prefill=False,labels_created=False,GUI_not_launched=True,optional_user_task=True),
    scoring_tool=str(ROOT/'tools/score_live_labels.py'),
    archive_scope='Old42-file source ZIP and0.3.1 position wheel are historical. No new archive, wheel or public upload.',
    limitations=['2D image coordinates only','No calibrated RPM/thrust/3D/wind/mass truth','Sparse/previously exposed labels',
        '5Hz publications include invalid heartbeats','Original comparison in single-model tests is offline at matching current frames, not a separately timed competitor'],
    next_action='Do not repeat alpha/Q/threshold sweeps. Use optional80-frame independent test3 annotations for fixed-horizon scoring; next optimization should target temporal localization errors rather than IoU alone. Research reusable temporal detection-consistency methods before implementation. Keep original default.',
    source_metadata_after_last_RAW='Only worker precision-scope logging and CLI runtime override/dry-run support changed after the latest RAW snapshots; no new full-chain claim for those metadata-only edits.')
write(OUT/'summary.json',summary)
progress=OUT/'progress.json'
if progress.exists() and not (OUT/'progress_while_training.json').exists():(OUT/'progress_while_training.json').write_bytes(progress.read_bytes())
write(progress,dict(utc=summary['utc'],execution_status=status,running_jobs=[],all_training_and_RAW_jobs_terminal=True,
    latest_report=str(OUT/'index.html'),summary='summary.json',automation_deleted=True))

forecast_rows=[];timing_rows=[]
for case,r in single['results'].items():
    for h,s in r['scores'].items():
        m=s['metrics'];a=m['original_model_replay'];b=m['merged_scale']
        forecast_rows.append(f'<tr><td>{case}</td><td>{h}</td><td>{a["n"]}</td><td>{a["mean"]:.2f}</td><td>{b["mean"]:.2f}</td><td>{a["median"]:.2f} → {b["median"]:.2f}</td><td>{a["p95"]:.2f} → {b["p95"]:.2f}</td></tr>')
    c=r['completion'];timing_rows.append(f'<tr><td>{case}</td><td>{c["tracking_updates"]}</td><td>{c["valid_publications"]}/{c["publication_slots"]}</td><td>{c["processing_median_ms"]:.2f}</td><td>{c["observation_age_p95_ms"]:.2f}</td></tr>')
c=extension['completion'];timing_rows.append(f'<tr><td>test3，8秒</td><td>{c["tracking_updates"]}</td><td>{c["valid_publications"]}/{c["publication_slots"]}</td><td>{c["processing_median_ms"]:.2f}</td><td>{c["observation_age_p95_ms"]:.2f}</td></tr>')
e=extension['metrics'];state_text='已按额度保留线暂停，无本轮后台任务。' if args.quota_paused else '当前批次已完成并保存；报告中的所有训练和RAW任务均已结束。'
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>无人机预测：本轮优化结果</title>
<style>body{{max-width:1120px;margin:35px auto;padding:0 20px;font:17px/1.75 system-ui;color:#203047;background:#fafcff}}h1,h2{{line-height:1.4}}h2{{margin-top:38px}}table{{width:100%;border-collapse:collapse;font-size:15px}}td,th{{padding:9px;border:1px solid #ccd5e0;text-align:left}}th{{background:#edf3fa}}aside{{padding:17px 22px;background:#eaf1f9;border-left:5px solid #3d78b7}}video,img{{max-width:100%;height:auto}}a{{color:#165b9e}}code{{background:#eff2f6;padding:2px 5px}}.warn{{background:#fff2dc;border-left-color:#c08a22}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}@media(max-width:800px){{.grid{{display:block}}table{{font-size:13px}}}}small{{color:#5b6777}}</style>
<h1>这轮做了实质优化，收益仍有边界</h1><aside><b>结论：原默认保留，新版本作为可选候选。</b>新训练改善了框定位，单模型候选在原来两段的未来位置平均误差有小幅收益；换到原定留出录像后，未来位置平均误差略差。因此不能称普遍更准。<br>{state_text}<br>最近核验：短窗口剩余 {args.primary_remaining}%、周窗口剩余 {args.weekly_remaining}%。一次性恢复自动化已经删除。</aside>
<h2>实际改动</h2><p>用原有训练图片做了一次固定10轮微调，加入局部放大样本；没有新增网络或反复挑权重。直接把放大后的中心用于预测会变差，已拒绝。随后保留原模型与新模型的匹配中心做等权结合，发现第二次推理会吃掉一部分收益；最终用已有 WiSE-FT / Model Soups 思路固定等权平均模型参数，回到一次推理，并接入先前的条件尺度修正。</p><p>原模型、7个位置核心文件、原研究流程都保留。新候选没有把桨叶事件密度当成已知推力，仍是二维位置通道。周期诊断另外做了SDK固定像素采样和传输去重：只减少重复数据，未证明新的推力或运动收益。</p>
<h2>两段实际RAW：同目标位置误差</h2><p>单位为像素。原模型参考是在<b>同一张原生画面、同一组接收时刻、同一个实际发布时间+h</b>下离线重放的状态，并非另外实测的实时竞争系统。候选的RAW处理与延迟是真实测量。标注是既有稀疏可见框中心，间隔≤150毫秒时插值；片段已参与开发。</p><table><tr><th>片段</th><th>预测秒数</th><th>共同目标数</th><th>原模型参考均值</th><th>单模型+尺度均值</th><th>中位数</th><th>P95</th></tr>{''.join(forecast_rows)}</table><p>0.4秒均值改善约1.21 / 3.45像素，尾部误差没有统一改善，不能只展示均值。候选与参考没有通过少报难例来获得这些配对分数。</p>
<div class="grid"><div><h3>侧视：0.4秒对比</h3><video controls preload="metadata" src="../single_model_optimized_RAW_20261026/side/comparison_04s.mp4"></video></div><div><h3>另一段：0.4秒对比</h3><video controls preload="metadata" src="../single_model_optimized_RAW_20261026/train2/comparison_04s.mp4"></video></div></div><p><small>蓝色为实际候选预测；橙色为同目标原模型离线参考；粉色是真实未来画面的人工框中心参考，仅在评估后叠加，不进入推理。底部说明单独扩展画布，完整保留原生720像素画面。</small></p>
<h2>留出录像：没有迁移成普遍收益</h2><aside class="warn">预先固定录像3的0—8秒，得到13个共同目标。由于既有标注稀疏，实际预测时域是 {extension['actual_horizon_range_s'][0]:.3f}—{extension['actual_horizon_range_s'][1]:.3f} 秒，<b>不是固定0.4秒成绩</b>。平均误差：原模型参考 {e['original_reference']['mean']:.2f}，合并模型 {e['merged_curved']['mean']:.2f}，加尺度后 {e['merged_scale']['mean']:.2f}。新版本均值略差，P95有所下降，保留这个负结果。</aside>
<p>这说明框IoU提升不自动等于未来位置更准；中心的时序噪声、变向响应和条件尺度假设仍需要改。原定测试录像3的107张有机弱标注图中，合并模型检出107张，原模型106张；IoU中位0.743→0.784，中心均值7.14→6.69像素；11张无机图各误检2张。另7张空场图均零检出。这些是检测结果，不能直接写成未来预测成绩或前沿基准。</p>
<h2>低频发布的实际可用性</h2><table><tr><th>片段</th><th>跟踪更新</th><th>有效/发布</th><th>处理中位ms</th><th>全部观测年龄P95 ms</th></tr>{''.join(timing_rows)}</table><p>5Hz是发布节拍，其中包含明确无效的心跳。侧视和另一段各4秒，留出段8秒；不是实时相机输入，也不是稳定连续有效5Hz的保证。train2在40.88—42.60秒有44次缺框，但这段没有人工可见标注；能核查的36个当前时刻原模型全部检出，不能把整段空档都称为检测失败。</p>
<h2>周期诊断与推理减负</h2><p>固定SDK像素掩码保留选中像素全部原生时间事件，检测图仍用完整事件。周期记录在选中像素逐字段一致；空间覆盖会减少。只对141无状态观测器提前去重，30次摘要、线、点全部一致。实际短RAW的管道数据6.93→1.04MB，约减少85%；不同运行的有效时刻与延迟不完全相同，未证明稳定端到端提速。</p><p>混合FP16仅做固定19图探针：文件10.45→5.33MB，GPU前向中位5.11→3.92ms，最大已选中心差0.382像素，框数量一致；cuDNN前端仍记录了执行计划错误信息，推理最终完成。<b>未进行完整RAW/未来轨迹检验，不接默认，不把它当全链提速。</b></p>
<h2>哪些方向已停止</h2><p>常加速度只换速度在两段均变差；周期线补位恢复0个缺框；60ms积累没有找回额外检测；直接匀转弯拟合没有可信当前转率；局部放大中心让轨迹变差。均保留结果，不继续扫门槛和权重。PVT++官方运动权重链接被封禁，只有核验过的作者源，未执行模型；EVtracker公开仓库当时只有README。没有把“有论文/有链接”写成“已经复现”。</p>
<h2>下一步已准备好</h2><p>复用了已有标注工具，准备录像3的80张5毫秒原生ON/OFF帧，10Hz，<b>没有预测框预填、没有自动制造标签</b>。不需要飞控日志或重新录制。你方便时可在标注目录双击 <a href="../dense_labels_train3_transfer_20261026/start_annotation.cmd">start_annotation.cmd</a>；这是可选任务，现在没有打开GUI。</p><p>后续优化优先解决“定位时序一致性”而非只提高IoU：应利用真实运动差分或变换等变约束，避免直接把真实转弯平滑掉。先查已有时序检测方法再动代码。原默认、未知/退化状态和真实时间语义继续保留。没有独立转速/姿态/推力真值，仍不能声称物理分支已带来轨迹优势。</p>
<h2>文件与可复用入口</h2><p><a href="summary.json">机器摘要</a> · <a href="../single_model_optimized_RAW_20261026/scores.json">实际单模型数据</a> · <a href="../heldout_motion_RAW_20261026/scores.json">留出录像负结果</a> · <a href="../equal_weight_detector_20261026/transfer_v2/scores.json">检测迁移对照</a> · <a href="../mixed_fp16_detector_v3_20261026/comparison.json">半精度探针</a> · <a href="../periodic_packet_20261026/comparison.json">周期包等价证据</a></p><p>源码入口仍为 <code>forecast.py run --profile live</code>。新增模型路径/哈希命令行覆盖，不必改默认配置。原0.3.1 wheel仅包含位置核心，旧源码ZIP均为历史快照，本轮没有重新打包或上传GitHub。各实际RAW目录保留当时的精确source_snapshot。</p><p>参考：<a href="https://github.com/obss/sahi">SAHI</a>，<a href="https://github.com/mlfoundations/wise-ft/blob/master/src/wise_ft.py">WiSE-FT权重平均</a>，<a href="https://proceedings.mlr.press/v162/wortsman22a.html">Model Soups</a>，<a href="https://onnxruntime.ai/docs/performance/model-optimizations/float16.html">ORT半精度</a>，<a href="https://docs.prophesee.ai/stable/api/python/core/bindings.html">原生SDK</a>。本轮是相关方法在本项目中的工程复用，不主张新理论、首创或完整论文复现。</p></html>'''
if (OUT/'mean_errors.png').exists():
    page=page.replace('<div class="grid">','<p><img src="mean_errors.png" alt="同目标的平均位置误差，小幅收益；留出运动未改善"></p><div class="grid">',1)
page_path=OUT/'index.html'
if page_path.exists() and not (OUT/'index_before_training_summary.html').exists():(OUT/'index_before_training_summary.html').write_bytes(page_path.read_bytes())
page_path.write_text(page,encoding='utf-8')

card=f'''# 2026-10-26模型候选：默认仍保留原模型

这轮完成了固定10轮局部放大增强训练（355+199张训练、180+93张验证，原定测试3/空场5不参与），以及固定0.5参数平均。标准方法复用，不是新网络。全部权重仅在本地，未公开上传。

|模型|ONNX SHA256|状态|
|---|---|---|
|原模型|{cfg['detector_sha256']}|默认保留|
|放大增强微调|{fine['sha256']}|检测迁移改善；短时预测门槛未通过|
|固定等权参数合并|{merged['sha256']}|可选候选；原两段小收益、留出运动均值略差|
|混合FP16|{half['sha256']}|仅19图数值/前向探针，未跑完整RAW，未默认|

单模型+尺度实际RAW：侧视固定0.4s n16，同目标原模型离线参考52.583→51.368px；train2 n6 88.629→85.181px。P95略差，不作为普遍升级证据。原定留出3号0—8s的13个变时域(.317—.451s)目标，60.994→61.770px；不可与固定0.4s跨口径比较。

两个模型同时推理的等权中心结合保留为可选诊断，命令参数`--center-ensemble-weights`和`--center-ensemble-sha256`。保留原检测资格、框尺寸和测量噪声，第二模型不能初始化或延长缺框。额外推理有真实耗时，不能只引用冻结时刻的较好分数。

原生周期观测的`--periodic-sampling fixed-quarter`与`--periodic-packet latest-pixel`仍为可选。后者只适合无状态141观测，不能作为状态化频率/相位引擎输入。都没有变成推力测量。

从项目根目录运行候选（需本机现有SDK/GPU环境；这是源码命令，不是新wheel）：

```powershell
.\\.venv-event311\\Scripts\\python.exe event_drone_forecast\\forecast.py run `
  --config event_drone_forecast\\configs\\local.json --profile live `
  --raw E:\\dataset\\train\\6\\media\\event.raw --start 110 --duration 4 `
  --output output\\my_single_model_candidate `
  --detector-weights output\\equal_weight_detector_20261026\\merged.onnx `
  --detector-sha256 {merged['sha256']} `
  --frame-transport frames --startup-sync reader-ready --scale-rollout compatible --preview
```

以上不会改变默认配置。`--dry-run`可只核验参数，不启动推理。命令行模型路径相对当前目录；配置文件内路径仍相对配置文件。

密集标签评分：`python tools/score_live_labels.py --run-dir <完成的live目录> --labels <GUI生成labels.json> --output <新分数.json>`。按实际发布时间+h，不用旧stage15评分器冒充live时间语义。留出3号80帧在`output/dense_labels_train3_transfer_20261026`，尚无新标签，没有自动启动GUI。

最后实际源快照在各RAW目录。其后仅增worker精度说明、CLI模型覆盖和评分工具，未声称再次完成整个RAW验证。旧42文件源码ZIP和0.3.1位置wheel均不含本轮全部改动，不能沿用旧包名宣称已更新。
'''
(ROOT/'docs/DETECTOR_CANDIDATES_20261026.md').write_text(card,encoding='utf-8')
(ROOT/'docs/CURRENT_OPTIMIZATION_20261026.md').write_text('# 当前优化状态\n\n'+
    '最新完整报告：output/optimization_resume_20261026/index.html。机器交接：output/checkpoints/latest_handoff.json。\n\n'+
    '实际完成：原生周期观测减负、条件尺度修正、一次固定训练、双模型中心对照、单模型权重平均及留出运动核查。原默认与7个位置核心文件保留。\n\n'+
    '结论：原两段未来位置均值有小收益，留出3号均值略差，未升级默认。不重复Q、alpha或门槛搜索。下一步解决检测中心的时序噪声，先调研可复用时序一致性方法；80张原生图已为可选独立密集标注准备，不能制造GT。\n\n'+
    '详见 [模型候选说明](DETECTOR_CANDIDATES_20261026.md)。'+state_text+'一次恢复自动化已删除；当前无本轮后台任务。\n',encoding='utf-8')

handoff_path=OUTPUT/'checkpoints/latest_handoff.json';handoff=json.loads(handoff_path.read_text('utf-8'))
handoff.update(utc=summary['utc'],execution_status=status,experiments_running=False,current_running_jobs=[],
    in_progress='No jobs running; optimization checkpoint saved; original default retained',
    latest_report=str(page_path),latest_live_revision='20261026-single-model-conditional-scale',
    next_action=summary['next_action'],resume_20261026=summary,
    quota_pause_requested=args.quota_paused,automation_resume=dict(id='automation',deleted=True,reason='Single authorized resume executed'),
    live_thrust_input_enabled=False,stable_high_accuracy_realtime_achieved=False)
write(handoff_path,handoff)
marker='### 恢复任务最终阶段：检测训练、单模型候选与留出负结果（2026-10-26）'
log=WORK/'WORKLOG.md'
if marker not in log.read_text('utf-8'):
    with log.open('a',encoding='utf-8') as f:
        f.write('\n\n'+marker+'\n\n'+
        '实际固定10epochs训练285.344s完成，旧权重不改；原定train1/2/4、val6、test3/空场5。355+199训练、180+93验证，前10层梯度冻结，AdamW1e-4、rect=True，单次验证fitness选best。期间Ultralytics额外下载Arial.ttf到本项目目录，未安装新模型库。新ONNX326f23...。\n\n'+
        '直接新全图模型侧视.4s49.930→49.094，但.1s8.730→10.415，门槛失败；ROI精定位95次仍把侧视.4s推到77.971，不接默认。固定等权中心组合保留原检测/框/测量噪声，冻结side49.930→47.652、train2 88.684→85.332；真实双模型side54.576→54.121、train2 91.992→88.051，包含第二模型成本32.345/27.581ms，不能当独立单模型实时胜利。\n\n'+
        '按已有WiSE-FT/Model Soups数学定义，用torch.lerp固定alpha0.5平均浮点state，整数计数保留原值；不扫alpha，不主张作者代码/完整论文复现。作者API源码固定下载遇TLS失败，公开代码/论文已浏览；未执行作者训练。合并ONNX SHA256 ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4。\n\n'+
        '单模型+条件尺度真实RAW各4s205片，side100次跟踪、18/20有效，处理23.444ms/年龄P95121.133ms；train2 101次、6/20有效，18.979ms/42.385ms。精确当前图原模型离线参考、同一实际publication+h：.4s side n16 52.583→51.368，train2 n6 88.629→85.181，P95都略差。不是独立计时基线，更不是前沿/泛化证明。视频未来GT只后处理，修正说明遮挡底部无人机的问题，扩展为820高画布；原遮挡版本保留_clipped_caption_v1。\n\n'+
        '原定留出3号固定0–8s已跑：400原生片/15905506事件、201跟踪、40发布36有效、处理21.407ms/年龄P9581.380ms。原模型201张精确当前图离线参考全SHA一致。既有稀疏框只能得到13个.317–.451s变时域目标：原60.994、合并61.422、加尺度61.770，均值略差，P95144.394→137.177；不能写成固定0.4s，也不升默认。\n\n'+
        '检测测试3:107有机图，合并107/原106、IoU中位.784/.743、中心均值6.694/7.139，11无机图各2误检；空场5七图均0。这不等于预测精度提升。\n\n'+
        '混合FP16仅固定19图探针：10.45→5.33MB，前向中位5.111→3.919ms，同框数、最大中心差.382px；cuDNN前端执行计划错误日志保留。初版图未拓扑排序、v2评分6列坐标切片错误，v3修正完成，未调模型/判据，未跑完整RAW、不默认。worker随后补精度范围为FP32 I/O，内部以哈希图为准；CLI新增SDK/GPU/模型/哈希/cache覆盖，dry-run无RAW/GPU和新输出目录，原7文件核心与配置均未改。\n\n'+
        '复用既有GUI工具准备3号80张5ms原生ON/OFF、10Hz、零预测框预填，没有自动标签或启动GUI；start_annotation.cmd可由用户自愿双击。新tools/score_live_labels.py按真实publication+h评分，与已保存side51.368px完全一致，不用旧时间语义。\n\n'+
        '当前所有训练/RAW/离线探针会话均已terminal，无本轮后台任务。报告output/optimization_resume_20261026/index.html和summary.json，候选说明docs/DETECTOR_CANDIDATES_20261026.md。原0.3.1 wheel/42文件ZIP历史，不重打包、不上传。一次恢复automation已确认deleteStatus=deleted，不会每天重唤醒；goal工具最后仍paused，实际由本授权heartbeat完成工作，未伪称控制器已恢复。\n')
if args.quota_paused:
    write(OUT/'quota_pause.json',dict(utc=summary['utc'],primary_remaining=args.primary_remaining,weekly_remaining=args.weekly_remaining,
        user_authorized_reserve_percent=10,no_jobs_running=True,goal_not_complete=True,automation_deleted=True))
alias=OUTPUT/'latest_live_forecast.html'
alias.write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=optimization_resume_20261026/index.html"><a href="optimization_resume_20261026/index.html">当前优化报告</a>',encoding='utf-8')
print(json.dumps(dict(report=str(page_path),status=status,core7_exact=True,default_unchanged=True,no_running_jobs=True),ensure_ascii=False))
