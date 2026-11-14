"""Update human/machine handoff after the completed151--156 work."""
from pathlib import Path
from datetime import datetime,timezone
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import json,shutil

ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent
OUT=WORK/'output/stage156_parallel_geometry_20261024'
report=OUT/'分层预测优化与并行加速.html'
backup=WORK/'output/checkpoints/handoff_before156_20261025';backup.mkdir(exist_ok=True)
for source in [WORK/'WORKLOG.md',WORK/'PROJECT_STATUS_20261022.md',WORK/'output/checkpoints/latest_handoff.json',ROOT/'README.md']:
    target=backup/source.name
    if not target.exists():shutil.copy2(source,target)

entry='''

## 156. 交接审计与151—156实际优化（2026-10-24—25）

用户要求持续按痛点优化，避免大量无意义验证。本轮先核对149/150与其他AI的结论，再做固定机制的小探针；有用路线进入真实RAW，无用路线立即停用。最终报告：[分层预测优化与并行加速](<D:/drone forecast/output/stage156_parallel_geometry_20261024/分层预测优化与并行加速.html>)；[运行与避免重复](<D:/drone forecast/event_drone_forecast/research/stage156/README.md>)。

**交接纠正**：worklog已到§155，但状态文档和latest_handoff仍停148。本轮确认149四段RAW完成，早期handoff_verification149里RAW=0仅是当时记录；150是负结果分组分析，不是新预测算法。77.9%/61.0%是全时段有框比例，不是有完整存在性标注的严格召回率；已标注有目标时刻为74/75、36/36。两套框合计111帧，0.4s旧口径96对，不是96段独立运动。五帧门槛属于默认结构模型，不是146—149物理支路的统一门槛；149侧视仍有36个条件特征时刻，但融合与纯偏差校正完全相同。不能从corr0.05推断真实姿态/RPM必然无用，也不能把6px单帧定位误差直接除以50px外推误差解释占比。

**实际新增路径**：
- 151复用已装Prophesee SparseOpticalFlow FastObjects，按当前ROI、每native轨迹一票、空间分布/一致性形成独立事件速度，再复用107固定FilterPy更新。775个实际RAW观测探针使用已有框作ROI，明确不是新增NN/端到端。side0.4s 50.027→57.916、train2 59.301→62.877：未采用、不接主入口，不继续调权重。
- 152以同样的138目标函数/当前周界/两形状/80预算做上一帧初始化。400时刻/800模型调用：仅省约2.6%中位，资格改变10次、中心最大变23.7px；不采用。
- 153把固定结构的无约束H8问题交给现成SciPy MINPACK LM，不成功回退原TRF，保留两形状假设、当前观测与原门槛/预算。同输入交替：模型中位10.85→8.75ms（约19%）；共同接受中心最大差0.00108px。27.78s原TRF达到80次上限未成功，LM收敛，RMS同为2.187px；不是放宽门槛。
- 154保持同一位置/速度/加速度及144阻尼，把法向加速度变成转弯率、切向加速度变成速度变化，积分曲线。速度<30px/s或外推>90°回原方法。全状态应用时clear0.4s自洽25.595→25.814略退，因此不普遍应用；156仅在尚未学成结构模型的回退状态使用。该资格不等于可靠正/侧视分类，也不把结构记忆当当前姿态。
- 155只比较既有16线程和单线程，两个同输入ROI单线程略慢，未改线程配置。
- 156把count/occupancy两套独立几何安全并行，去掉进程全局检测函数切换，改成实例固定检测器；各臂独立恢复/身份/结构状态，只共享只读计数，同刻join后再预测。100步同输入交替几何中位64.18→60.32ms，总量5.647→5.134s，观测和中心完全一致。并行的frontend/model字段是臂耗时之和，不能与geometry墙钟耗时相加。
- 当前NN无框时跳过不能参与位置更新的前端提取，模型仍传播与重置；GPU收包允许分离非JSON日志，首轮失败目录保留。旧接收器未保存出错行，具体来源不能断言。

**真实RAW主链**：154三段1175、156最终三段1175，共2350成功时刻，是相同录像的两轮，不算独立数据。首次clear通信失败不纳入，保留failed_clear_protocol。最终clear400、side475、train2 300：所有run的sources_unchanged、事件cutoff、禁读标签/缓存通过；检测框逐项同149；并行对照154三段原基线坐标最大差0。对149，side/train2原基线差0，clear最大0.00426px（LM一次收敛改善）。视频只做首/中/尾抽查，不冒称全解码。

| 窗口 | 0.4s历史同起点原→新px | 有效/缺失 | 严格已发布同起点原→新px | 有效/缺失 | 处理中位 |
|---|---|---|---|---|---|
| side | 50.027→49.216 | 66/缺9 | 46.430→45.594 | 64/缺11 | 59.84ms |
| train2 | 59.301→58.930 | 30/缺6 | 同左 | 30/缺6 | 49.45ms |
| clear（仅检测器自洽）| 25.595→25.595 | 377/缺20 | 25.393→25.393 | 373/缺24 | 98.75ms |

side0.1/0.2s历史口径9.887→9.844、18.399→18.213；train2 9.462→9.407、20.423→20.126。改善小，不能称已证明泛化或物理创新。严格已发布与历史系数可用口径完整保存，**禁止拿旧50.027和新45.594交叉比较**。评分器144增加可选--published-only：未发布计缺失，保留旧默认以便复算历史。人工框不是COM/RPM/姿态真值。

**侧视模糊补做**：149的diagnose149现已实际运行，4个固定时刻20/10/5/2ms原生PNG、压缩对照、55帧同期视频在stage149/blur。118.12s的20→5ms ROI占据像素3998→2413，平移拖影估计3.71→0.93px；短窗更稀疏，并不能恢复投影重合。5ms仍只改变显示；检测/周期保留20ms。不能由本试验确定原录制是否失焦。

**当前状态与后续**：默认138＋140＋144不变；可运行研究组合--pose-solver lm --parallel-geometry --skip-unobserved-geometry --curved-rollout --preview-window-ms 5。推荐输出键curved_rollout，preferred_forecast明确选择；新曲线是运动学回退，不是转速推力。仍非20ms实时，约11–14ms序列化/预览另计，不将跨次总耗时差全部归因于代码。下一步优先低频几何/高频位置与最新观测调度，量化实际发布延迟；运动侧围绕当前可观测性、真实机动，不重做已失败的裸框斜率守门/加速度模长分组。桨盘研究先限缩为姿态变化能否提前提示机动，不继续将未经独立验证的频率代理当总推力。

本轮无G、无新依赖、未访问/修改yolov10-main，E盘RAW只读。三个最终子进程均已join，主入口已执行资源清理；无本轮运行中的RAW/GPU作业。机器交接与状态文档现同步156；历史归档zip仍未包含143之后，不能声称旧zip已覆盖本轮。
'''
p=WORK/'WORKLOG.md';s=p.read_text(encoding='utf-8')
if '## 156. 交接审计与151—156' not in s:p.write_text(s+entry,encoding='utf-8')

header='''## 156：检测分层审计、曲线回退与并行加速（2026-10-25）

[最新报告与三段预测视频](<D:/drone forecast/output/stage156_parallel_geometry_20261024/分层预测优化与并行加速.html>) · [运行与交接](<D:/drone forecast/event_drone_forecast/research/stage156/README.md>)。详见WORKLOG§156，以下旧状态为历史。

149惯性/尺度模型已完成四段RAW，150阻尼分组无收益；旧机器交接停148已修正。151事件光流、152上一帧初始化、155单线程均未采用；153 LM求解和156安全并行保留。曲线回退只用于尚未学成结构的状态，清晰已建模路径保持原预测；不是可靠正/侧分类器，也没有RPM推力增益。

最终实际RAW1175时刻：side0.4s历史同起点50.027→49.216px（66/缺9），train2 59.301→58.930（30/缺6），clear检测器自洽25.595保持。严格仅已发布口径另列：side46.430→45.594（64/缺11），禁止与旧50.027跨口径比较。当前可选研究组合为`--pose-solver lm --parallel-geometry --skip-unobserved-geometry --curved-rollout --preview-window-ms 5`；无开关默认仍138＋140＋144。

处理中的中位clear98.75、side59.84、train2 49.45ms，另11–14ms序列化/预览；仍非20ms实时。下一步降低前端和实际发布延迟，采用低频几何/高频位置及最新观测调度；不堆物理权重或重复已失败的速度平滑。无本轮RAW/GPU后台。

'''
for p in [WORK/'PROJECT_STATUS_20261022.md',ROOT/'README.md']:
    s=p.read_text(encoding='utf-8')
    if not s.startswith('## 156：'):p.write_text(header+s,encoding='utf-8')

checkpoint=WORK/'output/checkpoints/latest_handoff.json'
d=json.loads(checkpoint.read_text(encoding='utf-8'))
scores=json.loads((OUT/'scores_actual156.json').read_text(encoding='utf-8'))
d.update(utc=datetime.now(timezone.utc).isoformat(),execution_status='checkpoint_ready',
    research_objective_status='incomplete',last_completed_stage=156,latest_full_forecast_stage=156,
    experimental_RAW_stage=156,latest_RAW_geometry_stage=156,latest_diagnostic_stage=156,
    latest_cached_forecast_stage=154,algorithm_stage=144,default_promoted=False,
    latest_report=str(report),latest_full_forecast_report=str(report),
    previous_report=str(WORK/'output/stage154_curved_rollout_20261024/检测分层审计与曲线预测优化.html'),
    latest_checks=str(OUT/'final_checks156.json'),current_status_document=str(WORK/'PROJECT_STATUS_20261022.md'),
    next_action='保留144默认；156研究配方LM/独立几何并行/无框跳前端/未建模曲线回退。优先低频几何与高频位置、最新观测队列及发布延迟；桨盘只先验证姿态机动提示，不把当前代理叫总推力。',
    completed_scope='151 event-flow RAW observation probe775 with saved boxes, rejected;152 warm-start replay rejected;153 LM model solve faster;154 full RAW1175;155 threading probe negative;156 full RAW1175, parallel geometry and unmodeled curved fallback.',
    completion_scope='本轮成功端到端2350时刻，两次相同3录像，不是独立数据。最终156共1175；小幅运动学收益，无RPM/力/姿态真值或实时性证明。',
    experiments_running=[],running_jobs=[],in_progress=[],
    optional_rotor_inertia_stage=149,optional_pose_solver_stage=153,optional_curved_rollout_stage=154,
    optional_parallel_geometry_stage=156,recommended_forecast_research_stage=156,
    current_optimized_RPM_input_enabled=False,event_RPM_advantage_proven=False,independent_RPM_gain_proven=False,
    realtime_proven=False,stable_high_accuracy_realtime_achieved=False,
    optimization_this_stage_actual_RAW_frames=2350,optimization_current_full_frames=1175,
    side_improvement_type='Small kinematic curved fallback on known-development manual bbox anchors; not measured thrust',
    side_physical_prediction_proven=False,
    current_score_conventions='Primary: origin valid=true. Legacy all-coefficient scores retained. Do not compare across conventions.',
    latest_stage156=dict(scores_file=str(OUT/'scores_actual156.json'),
        final_parameters=dict(pose_solver='lm',parallel_geometry=True,skip_unobserved_geometry=True,curved_rollout=True,preview_window_ms=5),
        processing_median_ms={n:v['completion']['processing_median_ms'] for n,v in scores.items()},
        physical_gain_proven=False,generalization_proven=False,default_promoted=False),
    archive_scope='Existing through142 ZIP is historical;143--156 live workspace files and156 source snapshot are not claimed inside that ZIP.')
checkpoint.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ['href','src'])
s=report.read_text(encoding='utf-8').replace('2026-10-24。151—156','2026-10-24—25。151—156')
s=s.replace('</html>','<p>复用：<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html">SciPy LM</a>、<a href="https://docs.prophesee.ai/stable/algorithms/optical_flow.html">Prophesee光流</a>。曲线运动与并行本身不作为新颖性声明。</p></html>')
report.write_text(s,encoding='utf-8');links=Links();links.feed(s)
missing=[];checked=0
for link in links.links:
    if urlparse(link).scheme or link.startswith('#'):continue
    checked+=1
    if not (report.parent/unquote(link.split('#')[0])).exists():missing.append(link)
assert not missing,missing
(OUT/'handoff156.json').write_text(json.dumps(dict(local_links_checked=checked,missing=missing,
    default_stage=144,latest_stage=156,worklog_section=156,report=str(report),
    dated='2026-10-25',running_RAW_jobs=[]),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(worklog_updated=True,status_updated=True,checkpoint_stage=156,local_links_checked=checked,report=str(report)),ensure_ascii=False))
