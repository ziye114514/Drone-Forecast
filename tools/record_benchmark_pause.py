"""Record the user's partial pause; no benchmarks or GPU operations."""
from pathlib import Path
import datetime,json
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
p=WORK/'output/checkpoints/latest_handoff.json';h=json.loads(p.read_text(encoding='utf-8'))
h['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
h['execution_status']='User paused benchmarking/GPU trials while gaming; lightweight code/docs only'
h['experiments_running']=False;h['running_jobs']=[]
h['benchmark_pause']=dict(requested_by_user=True,scope='All timing benchmarks and CUDA/GPU trial runs',
    reason='User is playing a game',resume_requires_explicit_user_message=True,
    allowed_work='Lightweight source, documentation and handoff editing only; no RAW replay/load tests',
    do_not_stop_other_applications=True)
h['in_progress']='Lightweight source/documentation consolidation; standalone live-runtime packaging is only planned, not implemented'
h['next_action']='Do not run benchmarks, RAW replays or CUDA until user explicitly resumes. Can prepare code/docs without executing workloads.'
h['shared_frame_transport_20261025']=dict(status='optional experimental, default remains events',
    report=str(WORK/'output/shared_frame_transport_20261025/equivalence.json'),
    exact_detector_frames=31,exact_preview_frames=31,native_events=16981921,
    frame_generation_and_GPU_hash_checks_passed=True,
    shared_run_publications=6,shared_run_valid_publications=3,
    pipe_transport_p95_ms=1.5771999969729222,
    preceding_events_profile_transport_p95_ms=29.54795000187005,
    comparison_is_not_controlled_GPU_benchmark=True,
    external_GPU_contention_observed_after_runs=True,
    current_source_archive_is_prior_to_shared_transport=True)
h['latest_CPU_mechanisms_20261025']=dict(
    time_support='Only1 unscored acceleration changed; publication coordinates unchanged, not adopted',
    polynomial_velocity='Train2 modest improvement but side worse; not adopted',
    reused146_with_phase='Proper existing delayed physical expert still no material gain;0.4s gate inactive, not adopted')
p.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')
marker='### 用户暂缓测速（2026-10-25，游戏期间）'
log=WORK/'WORKLOG.md'
if marker not in log.read_text(encoding='utf-8'):
    with log.open('a',encoding='utf-8') as f:
        f.write('\n\n'+marker+'''\n\n用户明确要求“我在玩游戏，暂缓测速吧”。立即暂停全部测速、CUDA/GPU试跑与RAW重放负载，等用户明确恢复；不停止游戏或其他应用。只允许轻量代码、文档与交接整理。当前本轮没有运行中的实验子进程。整个优化目标没有被用户取消，但测速这一部分已暂停。

暂停前新增可选共享画面传输：live_frames.py三槽共享内存，槽锁+代际编号，过期描述符拒绝；live_process.py保留全部原生事件SDK更新，传输可选events/frames，默认仍events。31张20ms检测画面与31张5ms预览逐字节相同，16,981,921原生事件/65片一致，无未来包。实际frames1.2秒6发布/3有效，传输P95约1.58ms；此前events约29.55ms，但GPU外部负载高，不能将非受控差异当稳定整链速度提升。frames未升默认，当前841文件源码ZIP早于这些改动，不能称已打包新传输。

另外三项CPU小机制未采用：按实际时间支持度替代帧数只影响一个未评分加速度时刻，已发布预测不变；同一稳健多项式求当前速度在train2略好但side变差；把新的相位验证特征喂给原146物理专家，0.4秒仍无门控权重、无实质增益。均不接默认、不继续扫权重。

后续原本考虑将当前live RAW运行所需的小型闭环打进可安装包，目前只是计划，尚未实现。已有0.3.1只完成position运动核心，完整RAW仍需源码目录、SDK与兼容权重；不要把计划写成安装成功。恢复测速前不要运行新的RAW/CUDA或负载实验。
''')
print('Recorded benchmark pause; no workload started',flush=True)
