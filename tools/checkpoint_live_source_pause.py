"""Save prepared source and paused-workload status; standard library only."""
from pathlib import Path
import datetime
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
delivery=WORK/'output/live_runtime_source_20261025_prepared_delivery.json'
prepared=json.loads(delivery.read_text('utf-8'))
contracts=WORK/'output/checkpoints/relocated_live_contracts_20261025_prepared.json'
check=json.loads(contracts.read_text('utf-8'))
assert not check['GPU_started'] and not check['RAW_events_read'] and not check['timing_measured']
manifest=json.loads((Path(prepared['directory'])/'manifest.json').read_text('utf-8'))
for row in manifest['source_files']:
    assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
historical=WORK/'output/optimization_run_20261025/source_candidate.zip'
with zipfile.ZipFile(historical) as z:
    members=[n for n in z.namelist() if n.endswith('/research/stage47/worker47.py') or n=='research/stage47/worker47.py']
    assert len(members)==1,members
    assert z.read(members[0])==(ROOT/'research/stage47/worker47.py').read_bytes()
prepared.update(contracts=str(contracts),status='prepared source, pending resumed RAW/CUDA smoke',
    full_clean_machine_install_verified=False,position_accuracy_gain_proven=False,
    original_research_worker_preserved=True,current_source_hashes_match_bundle=True,unverified_changes=[
        'New live CUDA worker end-to-end wiring',
        'Single SDK engine ownership in process mode',
        'Current-source frame transport regression after extraction'])

p=WORK/'output/checkpoints/latest_handoff.json';h=json.loads(p.read_text('utf-8'))
assert h['benchmark_pause']['requested_by_user']
assert h['benchmark_pause']['resume_requires_explicit_user_message']
h['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
h['in_progress']='Prepared live-only source bundle and lightweight source/CLI contracts; RAW/CUDA verification deferred by user pause'
h['next_action']='Wait for explicit workload resume before RAW/CUDA/timing. Then one short relocated live smoke, then controlled events/frames comparison; no physics-weight sweep.'
h['execution_status']='Benchmark, CUDA and RAW workloads remain paused by user; lightweight source/documentation checkpoint saved'
h['experiments_running']=False;h['running_jobs']=[]
h['prepared_live_runtime_source_20261025']=prepared
h['archive_scope']='Full841-file ZIP is historical through position-core extraction; not current shared transport/live cleanup. New small live-only source candidate has separate SHA and no RAW verification.'
h['latest_optimization_20261025']['next_action']=h['next_action']
p.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')

report_dir=WORK/'output/optimization_run_20261025'
summary_path=report_dir/'summary.json';summary=json.loads(summary_path.read_text('utf-8'))
summary.update(workload_pause=dict(active=True,resume_requires_explicit_user_message=True,
    no_new_RAW_GPU_or_timing_in_this_checkpoint=True),prepared_live_runtime_source=prepared,
    shared_frame_transport=dict(default='events',experimental='frames',
        exact_detector_frames=31,exact_preview_frames=31,
        equivalence=str(WORK/'output/shared_frame_transport_20261025/equivalence.json'),
        stable_end_to_end_speed_gain_proven=False),
    latest_negative_mechanisms=['time-coverage support: no scored publication change',
        'polynomial current velocity: train2 improves but side worsens',
        'existing146 physical expert with current phase: no material forecast gain'])
summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
html_path=report_dir/'index.html';html=html_path.read_text('utf-8')
banner='''<section id="gaming-pause-checkpoint"><h2>当前：用户游戏期间，负载测试暂停</h2>
<p>测速、CUDA/GPU试跑与RAW重放均保持暂停，等用户明确恢复。本轮只整理轻量源码、文档与启动参数，没有新的精度或速度成绩，也没有停止其他应用。</p>
<p>新增独立live检测进程与包内后端，隔离历史测速链；独立SDK进程拥有噪声/周期状态，父进程复用当前结果。源码AST与启动参数检查通过，<strong>新接线仍待恢复后做短RAW/CUDA检查</strong>。</p>
<p><a href="../live_runtime_source_20261025_prepared.zip">精简live源码候选ZIP</a>（约76KB，41文件；非wheel，需另配SDK、GPU环境与权重） · <a href="../live_runtime_source_20261025_prepared/manifest.json">文件清单</a> · <a href="../checkpoints/relocated_live_contracts_20261025_prepared.json">轻量检查结果</a> · <a href="../../event_drone_forecast/docs/LIVE_RUNTIME_SOURCE.md">使用与验证边界</a></p>
<p>默认传输仍events，共享画面frames未升默认。先前全部31对检测画面和31对预览逐字节一致，但非受控硬件时序差异不能作为加速结论。多项式当前速度使侧视变差，原146物理专家接入新相位也无实质预测增益，均不采用。</p></section>
'''
if 'id="gaming-pause-checkpoint"' not in html:
    anchor='<section><h2>这次实际改了什么</h2>'
    assert anchor in html
    html=html.replace(anchor,banner+anchor,1)
old='下一步应把合格旋转信号的增量与纯位置偏差校正做同起点比较；若没有超出位置分支的收益，不提高物理融合权重。进入缺桨段时先使用可信部分证据与显式缺失，禁止凑四桨强行算总推力。'
new='同起点的旋转增量与纯偏差对照已完成，没有一致预测增益；复用原146专家也没有实质帮助，停止继续扫权重。恢复负载实验后先核对精简live接线与当前画面传输，确认观测年龄及有效输出收益。缺桨时保留显式缺失，不凑四桨算总推力。'
html=html.replace(old,new)
html_path.write_text(html,encoding='utf-8')

marker='### 游戏期间轻量源码整理：精简live闭环（2026-10-25）'
log=WORK/'WORKLOG.md'
if marker not in log.read_text('utf-8'):
    with log.open('a',encoding='utf-8') as f:
        f.write('\n\n'+marker+'\n\n')
        f.write('用户暂停测速仍有效：本轮没有RAW重放、CUDA/GPU运行、测速或负载实验。只做轻量源码/文档，未终止其他应用。优化目标保持active，不能自行恢复负载。\n\n')
        f.write('新live_worker与detector_backend隔离历史45/44/41测速依赖；旧研究worker47保留。预处理/NMS/参数AST一致，ORT仅要求显式模型路径；GPU请求循环除模块别名外AST一致。独立SDK生产进程持有唯一噪声/周期引擎，父进程直接测量当前q；新接线待恢复后实际确认。额外记录可选三方帧SHA，启动前拒绝频率超界和thread+frames。位置公式/门槛/模型均不调。\n\n')
        f.write('准备精简实时源码：'+prepared['directory']+'；ZIP SHA256 '+prepared['sha256']+'。'+str(prepared['files'])+'文件，'+str(prepared['archive_bytes'])+'字节；白名单+逐文件SHA，无RAW/模型/SDK/HelixTrack/私有配置。源码与移出原目录的CLI合约通过，不导入数值/GPU/SDK库、不读RAW内容、无子进程。它不是wheel或完整部署验证，不能沿用旧实跑分数当当前版本成绩。v1/v2为中间候选，以prepared交接为准。\n\n')
        f.write('WORKLOG/机器handoff/当前HTML报告已同步。旧841文件完整ZIP及0.3.1位置wheel是历史产物。下一步仅在用户明确恢复后：一小段精简目录RAW接线，再同片段events/frames受控对照；不继续物理权重扫参。\n')

print(json.dumps(dict(saved=True,prepared=prepared['archive'],workloads_remain_paused=True,
    report=str(report_dir/'index.html')),ensure_ascii=False,indent=2))
