"""Summarize the completed small RAW checks without rerunning any workload."""
from pathlib import Path
import datetime
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
OUT=WORK/'output/live_startup_transport_20261025'
NAMES=['events_legacy','events_reader_ready','frames_reader_ready','relocated_frames_reader_ready','axis_reader_ready']


def load(p):return json.loads(p.read_text('utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text('utf-8').splitlines() if s]


def numeric_difference(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        return max((numeric_difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list):
        assert len(a)==len(b)
        return max((numeric_difference(x,y) for x,y in zip(a,b)),default=0.)
    if isinstance(a,(float,int)) and not isinstance(a,bool):return abs(a-b)
    assert a==b,(a,b)
    return 0.


data={};tracking={};predictions={}
for name in NAMES:
    d=OUT/name;s=load(d/'completion.json');protocol=load(d/'protocol.json')
    t=rows(d/'tracking.jsonl');p=rows(d/'predictions.jsonl')
    assert s['sources_unchanged'] and s['publication_slots']==len(p) and s['tracking_updates']==len(t)
    assert s['valid_publications']==sum(r['valid'] for r in p)
    assert s['frame_descriptor_mismatches']==0 and s['frame_pool_busy_skipped']==0
    for r in t:
        assert r['max_event_t'] is None or r['max_event_t']<round(r['t']*1e6)
        assert r['detector_frame_sha256']==r['gpu_frame_sha256']
        if r['producer_frame_sha256'] is not None:assert r['producer_frame_sha256']==r['detector_frame_sha256']
    for r in p:
        assert r['estimate_available_t'] is None or r['estimate_available_t']<=r['t']+1e-8
        assert r['observation_t'] is None or r['observation_t']<=r['t']+1e-8
        if r['valid']:assert r['observation_age_ms']<=160.00001 and r['body_age_ms']<=160.00001
    if name!='axis_reader_ready':
        assert s['native_batches']==65 and s['native_events']==16981921
    data[name]=dict(completion=s,periodicity_qualified=sum(bool(r.get('periodic_axis') and r['periodic_axis'].get('qualified')) for r in t),
        axis_qualified=sum(bool(r.get('periodic_axis') and r['periodic_axis'].get('axis_valid')) for r in t),
        source_files=len(protocol['sources']))
    tracking[name]={round(r['t'],8):r for r in t};predictions[name]=p

pairs=[]
for a,b in [('events_legacy','events_reader_ready'),('events_reader_ready','frames_reader_ready'),('frames_reader_ready','relocated_frames_reader_ready')]:
    shared=sorted(set(tracking[a])&set(tracking[b]));max_boxes=0.
    for t in shared:
        x,y=tracking[a][t],tracking[b][t]
        assert x['detector_frame_sha256']==y['detector_frame_sha256']
        max_boxes=max(max_boxes,numeric_difference(x['detector_boxes'],y['detector_boxes']))
    same_observations=set(tracking[a])==set(tracking[b])
    max_motion=None
    if same_observations:
        max_motion=max((numeric_difference(tracking[a][t]['coefficient'],tracking[b][t]['coefficient']) for t in shared),default=0.)
    assert max_boxes<1e-5
    if same_observations:assert max_motion<1e-9
    pairs.append(dict(a=a,b=b,shared_current_frames=len(shared),all_common_frame_hashes_exact=True,
        detector_max_difference=max_boxes,same_complete_tracking_times=same_observations,
        motion_max_difference_when_complete_inputs_match=max_motion))

report=dict(scope='Same known side110--111.2s, sequential small checks; no new labels or accuracy claim',
    startup_sync_default='none',transport_default='events',defaults_promoted=False,
    game_closed_by_user=True,controlled_hardware_or_cache_order=False,
    causal_time_and_frame_checks_passed=True,cases=data,pair_checks=pairs,
    accuracy_gain_proven=False,clean_machine_or_live_camera_verified=False,
    optional_axis_is_separate0p4s_wiring_only=True)
(OUT/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

columns=['events_legacy','events_reader_ready','frames_reader_ready','relocated_frames_reader_ready']
labels=['events原启动','events同步启动','frames同步启动','移出目录frames同步']
table=''.join('<tr><td>'+label+'</td>'+''.join('<td>'+str(data[n]['completion'][k])+'</td>' for k in ['tracking_updates','publication_slots','valid_publications'])+
    ''.join('<td>'+f"{data[n]['completion'][k]:.1f}"+'</td>' for k in ['observation_age_p95_ms','transport_lag_p95_ms'])+'</tr>' for n,label in zip(columns,labels))
delivery=load(WORK/'output/live_runtime_source_20261025_startup_delivery.json')
assert hashlib.sha256(Path(delivery['archive']).read_bytes()).hexdigest()==delivery['sha256']
page='''<!doctype html><meta charset="utf-8"><title>启动同步与共享画面短实验</title>
<style>body{font:17px/1.7 system-ui;max-width:1100px;margin:35px auto;padding:0 24px;color:#19232d}table{border-collapse:collapse}td,th{border:1px solid #ccd4dd;padding:8px 12px}video{max-width:100%}a{color:#1768ae}</style>
<h1>启动同步与共享画面：恢复后的短实验</h1>
<p>同一已知侧视RAW110—111.2秒，纯ON/OFF，逐次运行。用户关闭游戏后恢复。没有新增人工标签，不是精度实验或受控硬件基准。</p>
<table><tr><th>方式</th><th>跟踪</th><th>发布</th><th>有效</th><th>观测年龄P95(ms)</th><th>传输P95(ms)</th></tr>'''+table+'''</table>
<p>各1.2秒请求均处理65片（含0.1秒预热）、16,981,921事件。共同帧的原生画面SHA与检测输出一致；完整观测时间相同时运动系数也一致。新检测进程、启动同步与移出源码目录运行已完成此范围接线。</p>
<p>共享传输第一次得到5/6有效，移出目录的相同配置重复为4/6；尚无稳定有效数增益。不同运行文件缓存、温度和调度并未严格控制，不能把冷启动时间差或单个P95差归因于算法，也不能称连续有效5Hz。默认仍events、startup-sync none，改动均可选。</p>
<h2>周期分支的负结果</h2><p>另跑0.4秒周期接线：SDK与当前轴测量运行通过，但周期处理和频率数组传输增加负载，8个过旧输入被跳过，仅3次跟踪、2次发布且0次有效。此段与1.2秒主对照分开，不用于横比精度。周期诊断仍默认关闭，不为了输出推力拖慢位置通道。</p>
<h2>预测结果示例</h2><video controls preload="metadata" src="frames_reader_ready/predictions.mp4"></video>
<p>紫线是位置分支未来0.1/0.2/0.4秒预测，明确无效时不画预测；不是物理推力预测。视频固定发布频率，准确时间以JSON为准。</p>
<p><a href="comparison.json">原始对照数据</a> · <a href="../live_runtime_source_20261025_startup.zip">42文件精简源码候选</a> · <a href="../../WORKLOG.md">交接</a></p>'''
(OUT/'index.html').write_text(page,encoding='utf-8')

hpath=WORK/'output/checkpoints/latest_handoff.json';h=load(hpath)
h['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();h['experiments_running']=False;h['running_jobs']=[]
h['benchmark_pause']['active']=False;h['benchmark_pause']['requested_by_user']=False
h['execution_status']='User resumed workloads; short startup/transport/relocated RAW runs complete. Quota10% remaining stop policy active.'
h['in_progress']='Next optimization should target low-frequency maneuver errors; current live source relocation and startup/transport wiring complete in narrow smoke scope'
h['next_action']='Check quota before next batch. If remaining<=10%, checkpoint and pause goal under user instruction. Otherwise optimize maneuver/velocity mechanism; no more packaging expansion or physics-weight sweep.'
h['latest_startup_transport_20261025']=dict(comparison=str(OUT/'comparison.json'),report=str(OUT/'index.html'),
    defaults_promoted=False,RAW_runs_completed=len(NAMES),no_accuracy_gain_claim=True)
h['prepared_live_runtime_source_20261025']=dict(**delivery,narrow_relocated_RAW_smoke_completed=True,
    relocated_completion=str(OUT/'relocated_frames_reader_ready/completion.json'),
    full_clean_machine_install_verified=False,all_runtime_profiles_verified=False)
h['latest_optimization_20261025']['next_action']=h['next_action']
hpath.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')

main_dir=WORK/'output/optimization_run_20261025';summary=load(main_dir/'summary.json')
summary['workload_pause']=dict(active=False,resumed_by_user='游戏关掉了')
summary['quota_policy']=h.get('quota_policy')
summary['latest_startup_transport']=h['latest_startup_transport_20261025']
summary['prepared_live_runtime_source']=h['prepared_live_runtime_source_20261025']
(main_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
p=main_dir/'index.html';s=p.read_text('utf-8')
banner='''<section id="gaming-pause-checkpoint"><h2>当前：已恢复短实验，保留最后10%额度</h2>
<p>用户已关闭游戏并恢复RAW/GPU运行，同时要求剩余额度10%时保存暂停，10月26日01:08:30定时继续。</p>
<p>新live入口与移出目录的精简源码已完成窄范围RAW接线。1.2秒主对照有效发布4/6、4/6、5/6，相同配置移出目录重复为4/6，尚无稳定有效数增益，运动公式未变；周期诊断负载导致另一个0.4秒试验0次有效，继续默认关闭。不宣称精度提升或稳定整链加速。</p>
<p><a href="../live_startup_transport_20261025/index.html">查看启动/传输实测与预测视频</a> · <a href="../live_runtime_source_20261025_startup.zip">最新精简源码42文件</a> · <a href="../../WORKLOG.md">交接记录</a></p></section>'''
s,n=re.subn(r'<section id="gaming-pause-checkpoint">.*?</section>',lambda _:banner,s,count=1,flags=re.S)
assert n==1
p.write_text(s,encoding='utf-8')

marker='### 关闭游戏后恢复：启动同步、传输与移出目录RAW（2026-10-25）'
log=WORK/'WORKLOG.md'
if marker not in log.read_text('utf-8'):
    with log.open('a',encoding='utf-8') as f:
        f.write('\n\n'+marker+'\n\n')
        f.write('用户明确“游戏关掉了”，已撤销负载暂停。GPU检查当时0%利用率、约5.5W。新增可选startup-sync reader-ready：SDK准备后等待→完整检测预热→输出/发布器打开→放行源时钟。默认none，位置数学7文件仍逐字节等于已验证0.3.1核心；原研究worker不改。\n\n')
        for n,label in zip(columns,labels):
            c=data[n]['completion'];f.write(label+f"：1.2s，{c['tracking_updates']}跟踪、6发布/{c['valid_publications']}有效，观测年龄P95 {c['observation_age_p95_ms']:.3f}ms，传输P95 {c['transport_lag_p95_ms']:.3f}ms。\n\n")
        f.write('每段65切片/16,981,921原生事件；同帧SHA/NN输出核对通过，完整输入集相同时运动系数一致，因果时间/源码未变检查通过。单次顺序试验、文件缓存/硬件状态未严格控制；默认仍events+none，不称稳态提速、精度提升或连续有效5Hz。周期0.4s支路运行通过但8过旧输入跳过，3跟踪、2发布/0有效；继续默认关闭。\n\n')
        f.write('最新精简源码'+delivery['archive']+'，42文件/'+str(delivery['archive_bytes'])+'字节，SHA256 '+delivery['sha256']+'。移出目录RAW已跑通这一小段；复用本机SDK/权重/依赖，非干净机器、非wheel全链、非所有模式验证。旧prepared/v1/v2和841文件源码ZIP均为历史。\n\n')
        f.write('额度新约束：按“剩余10%”暂停，不耗尽。每短批次前查primary/secondary，达到阈值保存交接后调用update_goal paused。已创建一次thread heartbeat automation，ID automation，计划北京时间2026-10-26 01:08:30恢复，账户重置预计01:08:17；不使用重置券。后续优先变向速度/时域误差，停止扩展打包和物理权重扫参。\n')

print(json.dumps(dict(report=str(OUT/'index.html'),pair_checks=pairs,
    relocated_source_bytes=delivery['archive_bytes'],RAW_jobs_completed=len(NAMES),quota_policy=h.get('quota_policy')),ensure_ascii=False,indent=2))
