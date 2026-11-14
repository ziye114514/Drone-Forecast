"""Check final phase artifacts and save the actual active-project handoff."""
from pathlib import Path
import ast,json,re,datetime,urllib.parse
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/shared_angular_phase_20261027'
def rows(folder):return list(map(json.loads,(OUT/folder/'observations.jsonl').read_text('utf-8').splitlines()))
a=rows('raw_clear_pairmin');b=rows('raw_directional')
for x,y in zip(a,b):
    x.pop('processing_ms');y.pop('processing_ms')
    for q in y['rotors']:q.pop('signed_power_directionality',None)
    assert x==y
links=[]
for u in re.findall(r'(?:href|src)="([^"]+)"',(OUT/'index.html').read_text('utf-8')):
    if urllib.parse.urlsplit(u).scheme or u.startswith('#'):continue
    assert (OUT/urllib.parse.unquote(u)).resolve().exists(),u
    links.append(u)
files=['shared_phase_fft.py','tools/check_shared_phase_fft.py','tools/probe_shared_phase_fft.py',
    'tools/audit_shared_phase.py','tools/report_shared_phase.py','tools/probe_five_circle_author_phase.py']
for f in files:ast.parse((ROOT/'event_drone_forecast'/f).read_text('utf-8'))
check=dict(current_RAW_phase_outputs_unchanged_after_directional_guard=True,source_ticks=60,
    local_links=len(links),all_local_links_exist=True,python_syntax_files=len(files),
    plotted_phase_tracks_inspected=True,full_HTML_render_not_claimed=True,
    all_current_jobs_terminal=True,production_default_changed=False)
(OUT/'artifact_checks.json').write_text(json.dumps(check,indent=2),'utf-8')
p=ROOT/'output/checkpoints/latest_handoff.json';d=json.loads(p.read_text('utf-8'))
d.update(current_jobs=[],current_running_jobs=[],quota_pause_requested=False,
    current_work_state='active project; two completed optimization batches; next visibility-invariant rotor proxy',
    latest_phase_report=str(OUT/'index.html'),latest_phase_artifact_checks=str(OUT/'artifact_checks.json'),
    latest_progress_checkpoint=str(OUT/'handoff.json'),
    shared_angular_phase=dict(report=str(OUT/'index.html'),production_default_changed=False,
        current_qualified_estimates=47,four_supported_ticks=6,source_ticks=60,
        same_pixel_period_comparisons=47,mean_relative_difference=.0109687900384146,
        independent_RPM_truth=False,position_forecast_advantage_proven=False,
        common_flicker_false_accepts_before_after=[4,0],real_RAW_qualified_estimates_preserved=True,
        fixed_body_shared_wobble_control=True,component_mean_ms=16.113613333315396,
        component_p95_ms=46.94774000236066,production30FPS_not_proven=True,
        original_author_core='parallel qualified-arc pilot crashed; serial completed but344ms; quarter/no-pose87ms. Default main pipeline unaffected.',
        next_action='Per-model-slot baseline normalization when visibility changes4/3/2. Direct mean of visible f^2 changes when unequal-speed rotors disappear. Map only genuinely accepted current conics to stable model IDs, keep hidden values unmeasured; pooled side proxy needs a separate continuity gauge. Keep original runtime control.'),
    automation_resume=dict(id='00-11',deleted=True,reason='00:11 quota restoration checked while work already active; one-time recovery fulfilled, duplicate wake removed'),
    updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
policy=d.setdefault('low_consumption_policy',{})
policy.update(single_0011_recovery_fulfilled=True,low_mode_entered=False,
    reason_not_entered='Both windows stayed above10% before the00:11 reset',task_is_not_paused=True,
    next_short_reset_utc='2026-10-26T21:10:39Z',
    do_not_reuse_expired_0011_deadline=True)
p.write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf-8')
(OUT/'handoff.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),'utf-8')
with (ROOT/'WORKLOG.md').open('a',encoding='utf-8') as f:
    f.write('\n\n### 00:11恢复后：共享几何批量空间相位完成（2026-10-27）\n'
        '00:11窗口实际恢复已核验，恢复前未低于10%，未触发低耗等待、未调用goal暂停。单次heartbeat00-11在本对话继续工作时删除，避免重复唤醒。账户最后检查短余82%/周74%；未来不得复用已过去的00:11时刻，当前API下次短重置为北京时间05:10:39。\n'
        '作者核心串行残弧初始化完成60时刻，5次四路支持但均344ms；25%固定像素/关闭姿态求解仍87ms、2次四路支持，不接默认。并发崩溃根因未确诊，独立试验进程结束。新shared_phase_fft.py为自写NumPy/SciPy实现，不依赖作者DLL；共享五圆平面仿射矫正后累计ON/OFF复数角矩，100us箱/160ms窗FFT，隐藏槽不补频率、轴线退化拒绝空间相位。不是灰度/G通道重建。\n'
        '实际同2秒RAW/冻结因果几何，60源时刻有6次四路合格、共47当前逐桨估计；全部与匹配模型槽SDK像素周期同倍频10%内，平均差1.10%，不是独立真实RPM误差。合成100Hz约99.977，机体不动/共同平面摆动仍99.977–99.980，随机时序和隐藏槽不伪造。新增正反频谱3:1门控，修复静态非均匀圆环共同闪烁旧门槛误接纳4→0，真实RAW47结果逐项保留。\n'
        '组件不同负载均9.81/16.77ms，最新带方向门控16.11ms/P9546.95ms；不是端到端FPS，160ms窗也可能滞后。移除四项排序，15交替48k数组操作3.389→2.069ms，60RAW输出逐项相同，不冒充整机加速。报告output/shared_angular_phase_20261027/index.html、方法docs/SHARED_ANGULAR_PHASE.md、README更新、相位图已看、合成控制及本地链接/语法通过。默认预测仍为上一批已完整RAW验证的三桨/二力/等价加速版，原位置核心保持。\n'
        '下一步优先：稳定模型槽对应的转速基准归一化，去掉4→3可见集合改变导致的假推力跳变；这是直接作用物理输入且计算便宜的路径，不先盲目给未证明预测价值的FFT堆GPU。当前无后台作业，项目不设置暂停。\n')
print(json.dumps(check))
