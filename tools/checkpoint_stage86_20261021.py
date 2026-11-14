"""Verify79–86, refresh handoff/docs, and save recoverable sources/evidence."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import datetime,hashlib,json,re,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;BASE=PROJECT/'output'
for n in [79,80,81,82,83,84,85]:sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/f'research/stage{n}'))
import study79,study80,study81,study82,study83,study84,study85
MODULES=[study79,study80,study81,study82,study83,study84,study85]
load=study79.load;save=study79.save;sha=study79.sha;OUT=study85.OUT
class Links(HTMLParser):
    def __init__(self):super().__init__();self.local=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and v and not urlsplit(v).scheme and not v.startswith('#'):self.local.append(unquote(v.split('#')[0]))
def run():
    for module in MODULES:module.verify()
    checks={int(m.__name__.removeprefix('study')):load(m.OUT/('checks'+m.__name__.removeprefix('study')+'.json')) for m in MODULES}
    for n in [79,80,82,84]:assert checks[n]['parent']==2185 and checks[n]['native10']==4370
    for n in [81,83,85]:assert checks[n]['parent']==835 and checks[n]['native10']==1670 and checks[n]['video_frames_decoded']==600
    assert checks[85]['all49_exact78'] and checks[84]['all49_and_current_exact78'] and checks[80]['all49_and_current_exact78']
    assert load(study84.OUT/'protocol.json')['controls']['differing_numeric_fields']==0
    for m,n in [(study80,80),(study82,82),(study84,84)]:assert (m.OUT/f'benchmark{n}.json').is_file()
    auditdir=BASE/'stage86_latency_deadlines_20261021';auditp=load(auditdir/'protocol.json');assert auditp['source_sha256']==sha(ROOT/'research/stage86/latency86.py')
    for key,value in auditp['input_hashes'].items():
        stage,case=key.split(':');folder={78:BASE/'stage78_RAW_separated_forecast_20261021',81:study81.OUT,83:study83.OUT,85:OUT}[int(stage)]
        assert sha(folder/case/'observations_forecasts.jsonl')==value
    inventory=load(OUT/'process_inventory_20261021.json');assert inventory['matching_project_python_count']==0
    delivery=load(OUT/'delivery_data85.json');assert delivery['helper_sha256']==sha(ROOT/'tools/deliver_stages79_86.py')
    assert load(study79.OUT/'delivery79.json')['helper_sha256']==sha(ROOT/'tools/deliver_stage79.py')
    reports=[Path(delivery['report']),Path(load(study79.OUT/'delivery79.json')['report'])];links=0
    for report in reports:
        parser=Links();parser.feed(report.read_text('utf-8'))
        for name in parser.local:assert (report.parent/name).resolve().is_file(),name
        links+=len(parser.local)
    prior=load(BASE/'checkpoints/latest_handoff.json');assert sha(prior['archive'])==prior['archive_sha256']
    failed=BASE/'stage83_RAW_streamed_window_20261021';failedp=load(failed/'protocol.json')
    for original,snapshot in [('run_stream_forecast.py','failed_source_run_stream_forecast.py'),('research/stage83/study83.py','failed_source_study83.py')]:
        assert sha(failed/snapshot)==failedp['source_hashes'][original]
    assert load(failed/'failure_postmortem83.json')['numeric_current_and49_exact78']
    for vendor in ['filterpy_imm_upstream','filterpy_mmae_upstream']:
        directory=ROOT/'external'/vendor;snap=load(directory/'UPSTREAM_SNAPSHOT.json')
        for item in snap['files']:
            raw=(directory/item['path']).read_bytes();assert sha(directory/item['path'])==item['sha256']
            if item.get('git_blob_sha1'):assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==item['git_blob_sha1']
    timing={(r['stage'],r['case']):r for r in delivery['timing_table']};metrics=[]
    for case in ['clear6','side6','train3','extra2']:
        a=timing[78,case];b=timing[85,case];metrics.append(dict(case=case,old78_ms=a['median_processing_ms'],new85_ms=b['median_processing_ms'],
            optimistic_no_decode_processing_fps=b['optimistic_no_decode_processing_fps'],deadlines=b['lead_deadlines']))
    speed='；'.join(f"{r['case']} {r['old78_ms']:.2f}→{r['new85_ms']:.2f}ms" for r in metrics)
    report=reports[0];relreport=report.relative_to(BASE).as_posix()
    update=f'''\n**最新79–86完成：输入诊断与保持49路的计算优化。** [完整报告/视频](</{report.as_posix()}>)，新RAW入口 `event_drone_forecast/run_batched_forecast.py`（85），默认54保持。\n\n79实际SDK频率未普遍补齐缺失，未接入推力；记录24B/16B视图适配和首次越频带失败。80预测映射隔离复制、82全局取样提前拼接、84批量26时距未来投影，数学/参数/49路/模式学习均保持78。80/84各2185父严格全49回归，82同量当前活动精确；81/83v2/85每轮835父/1670原生、600视频帧，两真实RAW前缀通过。上述多轮是重复上下文，不能相加为独立数据。\n\n完整同源前缀单步中位记录：{speed}。为分次记录，控制模块基准另存80/82/84，不能把整链路比值当随机交错保证。86按记录估算50Hz FIFO截止时效，非实机摄像头延迟测量；排队仍增长，当前不实时。预测值保持，所以沿用78误差/失败，未提高准确率或证明RPM独立增益。\n\n83首版包装收尾失败（100帧数值已完成）保留原源/协议/事后检查，修复后v2全量重跑。下一步优先有界队列/异步观测/时间戳更新的实际因果测试，以及独立变向标注与有用周期/姿态覆盖；不能再只堆同稀疏标签上的专家。以下旧状态均为历史。\n'''
    today=datetime.datetime.now().strftime('%Y%m%d');date_label=datetime.datetime.now().strftime('%Y-%m-%d')
    status=PROJECT/f'PROJECT_STATUS_{today}.md'
    text=status.read_text('utf-8') if status.exists() else f'# 当前继续状态：{date_label}\n\n此前状态见[2026-10-21交接](PROJECT_STATUS_20261021.md)，逐轮经过见WORKLOG.md。\n'
    heading=text.splitlines()[0]
    if '**最新79–86完成' not in text:text=text.replace(heading,heading+'\n'+update,1)
    status.write_text(text,encoding='utf-8')
    readme=ROOT/'README.md';history=ROOT/'docs/README_HISTORY_THROUGH_STAGE78_20261021.md'
    if not history.exists():history.parent.mkdir(exist_ok=True);history.write_bytes(readme.read_bytes())
    speedrows='\n'.join(f"| {r['case']} | {r['old78_ms']:.2f} | {r['new85_ms']:.2f} |" for r in metrics)
    readme.write_text(f'''# Event Drone Forecast

只输入事件相机RAW的二维无人机检测、桨叶几何/周期观测与未来位置预测研究工程。当前输出是图像像素坐标；尚未实现稳定、高精度、端到端实时，也未独立验证机械RPM、姿态或真实质量/风速。

{date_label}最新79–86完成：[实验报告、科学图和预测视频](</{report.as_posix()}>)。最新完整实验入口是[run_batched_forecast.py](run_batched_forecast.py)（85），保留全部49路研究对照。原[稳定默认入口54](run_deploy_forecast.py)和旧实验代码保留；85没有改变78预测数值或提高其准确率。

## 运行

本机已配置SDK Python3.11、固定本地检测权重和独立GPU worker环境；示例针对本机1280×720 RAW。输出目录必须不存在，E盘只读。

```powershell
& 'D:\\drone forecast\\.venv-event311\\Scripts\\python.exe' 'D:\\drone forecast\\event_drone_forecast\\run_batched_forecast.py' --raw 'E:\\dataset\\train\\6\\media\\event.raw' --output 'D:\\drone forecast\\output\\my_batched_trial' --start 108 --duration 5 --preview --verify-frames
```

只使用原生ON/OFF，不使用G灰度重建。视频蓝CV、紫旧IMM、黄mmae_state_fresh；无当前物理支持时黄线回历史。`forecasts.mp4`按源50fps播放，不代表运行吞吐量。`observations_forecasts.jsonl`保存当前观测和全部预测，`protocol.json`/`completion.json`保存输入、源哈希与实际耗时。按独立CLI进程使用，常驻服务需先隔离审计钩子生命周期，见[85说明](research/stage85/README.md)。

## 方法与本轮改动

当前整机检测与桨盘/线段支持共同提供位置和方向，局部ON/OFF周期/活动作为有效推力代理，延迟位置反馈更新图像域参数。CV、历史运动及物理分支并行；MMAE保护独立CV状态。物理缺失时仍发布历史预测，不能靠少输出改善成绩。

79实际复用SDK逐像素频率，但未普遍填补缺失，未进入推力。80减少预测字典复制并缓存同窗重复背景tone；82保持原全局stride/跨包相位，在拼接前取样；84批量计算原26个未来时距的均值/协方差。没有改阈值、时间窗、滤波学习或减少49路。

## 验证与限制

80/84各2185父/4370原生RAW模块检查，全49、当前观测、模式/学习状态严格等于78；82同量当前活动字段一致。81/83v2/85每轮完整835父/1670原生，600视频帧全解码，两实际RAW截断前缀通过。这些多轮重复同一上下文，不能相加为独立数据。83首版包装收尾失败及原源码保留，v2重新完整验证。

相同源时间前缀的完整单步中位耗时记录（ms）：

| 片段 | 原78 | 新85 |
|---|---:|---:|
{speedrows}

这是分次记录；受worker/温度等影响，不能视为交错整链路保证。独立交错模块基准见报告。86仅按保存耗时做乐观FIFO截止审计，不是实机摄像头延迟测量。排队仍增长，完整链路未达到20ms预算。

预测保持78，所以原稀疏框误差及失败仍成立：额外2号弱于CV、额外4号弱于强历史，匹配常量经常同样或更好。未证明RPM独立收益、未知物体泛化或超过前沿论文；这些工程优化也不应当作新算法创新。

## 继续研究与交接

- [当前状态](../PROJECT_STATUS_{today}.md)、[完整工作日志](../WORKLOG.md)、[机器可读交接](../output/checkpoints/latest_handoff.json)。
- [85入口与边界](research/stage85/README.md)、[86延迟观测相关工作/下一步](research/stage86/RELATED_WORK.md)。下一步优先有界队列、异步观测和源时间戳更新的实际因果验证，以及独立变向标注和可靠周期/姿态覆盖。
- [原78精度与对照](../output/stage78_RAW_separated_forecast_20261021/第七十六至七十八阶段状态隔离与预测修正报告.html)、[79SDK完整失败/控制](../output/stage79_native_frequency_20261021/第七十九阶段逐像素频率与SDK接口诊断.html)。
- [截至78的原README历史](docs/README_HISTORY_THROUGH_STAGE78_20261021.md)。旧阶段“最新/未完成”均是当时记录，以当前状态与实际checks文件为准。

可恢复快照含源代码、协议、控制和报告；RAW、模型权重、完整JSONL/视频保留原路径。作者FilterPy源与MIT许可保留，SDK二进制未分发；没有读写受保护的yolov10-main。
''',encoding='utf-8')
    worklog=PROJECT/'WORKLOG.md';text=worklog.read_text('utf-8')
    if '## 115. 79–86本轮完成与交接' not in text:
        text+=f'''\n## 115. 79–86本轮完成与交接\n\n- 79已完成SDK模块/人工控制，80/82/84各2185父/4370原生模块检查；80/84全49严格相等78，82全部activity字段精确，前缀通过。79不启用，预测数学与精度保持，不按记录挑模型。\n- 81/83v2/85各835父/1670原生、600视频帧实际完整RAW/NN/桨叶worker结束、全部49/当前/状态精确78，两实际截断前缀通过。83首版包装失败原源/协议/100帧数值事后检查保留；修正版独立重跑。\n- 当前完整相同前缀中位：{speed}。独立模块交错基准分别保存，环境/温度/worker差异不当整链路随机对照。86完成记录截止时效审计，FIFO排队只是乐观推算，非实机实时测量，所有限制见报告。\n- tools/deliver_stages79_86.py报告/科学图/固定帧生成并核验；每步/视频源帧不等于实时吞吐量。无新机械RPM/姿态/未知参数识别或稳定高精度主张，默认54保持。\n- 最后进程盘点为0，所有本轮批次结束，可恢复快照含旧78清单和当前源/报告/控制/元数据；RAW/权重/完整JSONL/视频保留原路径，不写入小快照。完整目标尚未达成。\n'''
    worklog.write_text(text,encoding='utf-8')
    overview=BASE/'research_overview_20261018/index.html';page=overview.read_text('utf-8')
    lead='<div class="lead"><b>79–86完成：保持全部49预测，优化计算，实时目标仍未达成。</b><p>新RAW入口run_batched_forecast.py；逐像素SDK输入未普遍改善，未接推力。80/82/84模块严格一致，85真实RAW/前缀/600帧视频通过；报告区分模块加速、实际耗时与FIFO时效。默认54保持。</p></div>'
    page,n=re.subn(r'<div class="lead">.*?</div>(?=<div class="grid">)',lambda _:lead,page,count=1,flags=re.S);assert n==1
    url='../'+relreport
    if url not in page:page=page.replace('<div class="grid">','<div class="grid"><article><h2><a href="'+url+'">79–86 · 周期输入、精确计算优化与发布时效</a></h2><p>SDK失败、三项优化、实际RAW视频和时延边界。</p></article>',1)
    overview.write_text(page,encoding='utf-8')
    summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_status='checkpoint_ready',research_objective_status='incomplete',
        last_completed_stage=86,completed=list(range(19,87)),latest_full_forecast_stage=85,latest_cached_forecast_stage=84,in_progress=[],experiments_running=[],
        completed_scope='79 native SDK input diagnosis;80/82/84 exact computation controls;81/83v2/85 complete RAW deployment;86 recorded deadline audit.',
        default_promoted=False,new_thrust_input_enabled=True,new_thrust_input_scope='Existing78 event branches only; new79 native SDK candidates disabled.',
        latest_report=str(report),previous_report=prior.get('latest_report'),local_report_links_checked=links,
        latest_verified_baseline_entry=str(ROOT/'core/run_deploy_forecast.py'),experimental_RAW_entry=str(ROOT/'core/run_batched_forecast.py'),experimental_RAW_stage=85,
        previous_separated_RAW_entry=str(ROOT/'core/run_separated_forecast.py'),previous_streamed_RAW_entry=str(ROOT/'core/run_stream_forecast.py'),
        counts_by_stage=delivery['checks'],all49_exact78=True,accuracy_changed=False,
        stable_high_accuracy_realtime_achieved=False,event_RPM_advantage_proven=False,
        same_prefix_recorded_timing=metrics,no_G_or_yolov10_main_access=True,new_dependencies_installed=False,
        inventory_file=str(OUT/'process_inventory_20261021.json'),current_status_document=str(status),
        retained_failed_attempts=prior.get('retained_failed_attempts',[])+[
            '79 Python frequency buffer24B native/16B numpy view adapter needed; first synthetic range assertion failed before RAW',
            '79 strict SDK frequency zero coverage on train3/extra1/extra2, not deployed',
            '83 original wrapper after audit namespace clear failed postprocessing; original sources preserved;v2 rerun completed'],
        next_action='Keep85 exact numerical forecast as an experimental faster entry. Prioritise bounded queues, '
            'async current observations and timestamp-aware updates under actual dropped/delayed RAW packets. '
            'Keep source-time accuracy separate from publication deadlines; add independent maneuver truth '
            'and reliable periodic/attitude coverage. Do not claim mechanical RPM or tune more experts on same sparse labels.',
        archive_scope='Prior78 source/evidence manifest plus current79–86 sources/docs/small protocols/checks/reports/plots; '
            'RAW/weights/full JSONL/video remain original referenced paths. No SDK binaries redistributed.',
        prior_archive=prior['archive'],prior_archive_sha256=prior['archive_sha256'])
    files=set()
    with zipfile.ZipFile(prior['archive']) as zipped:
        manifest=json.loads(zipped.read('SNAPSHOT_MANIFEST.json'))
        for name,digest in manifest.items():
            assert hashlib.sha256(zipped.read(name)).hexdigest()==digest
            path=PROJECT/name
            if path.is_file():files.add(path)
    for parent in [ROOT/'src',ROOT/'research',ROOT/'tools',ROOT/'docs']:
        for pattern in ['*.py','*.md']:files.update(p for p in parent.rglob(pattern) if '__pycache__' not in p.parts)
    files.update(ROOT.glob('*.py'));files.update(ROOT.glob('*.md'));files.update(PROJECT.glob('PROJECT_STATUS*.md'));files.add(worklog)
    for n in range(79,87):
        for folder in BASE.glob('stage'+str(n)+'_*'):
            for pattern in ['*.json','*.md','*.html','*.png','*.py','*.txt']:
                files.update(p for p in folder.rglob(pattern) if p.stat().st_size<2_000_000 and p.name!='analysis.json' and not p.name.startswith('regression_failure'))
    files.update(overview.parent.glob('*.html'))
    archive=BASE/'checkpoints'/('research_through_stage86_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.zip')
    manifest={p.relative_to(PROJECT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(files):zipped.write(p,p.relative_to(PROJECT).as_posix())
        zipped.writestr('SNAPSHOT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2));zipped.writestr('HANDOFF_CHECKPOINT.json',json.dumps(summary,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        for name,value in manifest.items():assert hashlib.sha256(zipped.read(name)).hexdigest()==value
    summary.update(archive=str(archive),archive_sha256=sha(archive),source_summary_files=len(files),zip_verified=True)
    save(BASE/'checkpoints/latest_handoff.json',summary);save(BASE/'checkpoints'/f'latest_progress_{today}.json',summary)
    print(json.dumps(dict(last_stage=86,files=len(files),archive=str(archive),verified=True),ensure_ascii=False),flush=True)
if __name__=='__main__':run()
