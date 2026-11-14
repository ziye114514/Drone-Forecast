"""Save this ongoing optimization milestone without marking the goal complete."""
from pathlib import Path
import datetime,json

ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/optimization_run_20261025'
p=WORK/'output/checkpoints/latest_handoff.json';h=json.loads(p.read_text(encoding='utf-8'))
h['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
h['execution_status']='active optimization; all current RAW/NN/render jobs completed, continuing portable motion core'
h['in_progress']='Extract installable live motion core without historical research imports; preserve actual frozen states exactly'
h['experiments_running']=False;h['running_jobs']=[]
h['latest_report']=str(OUT/'index.html')
h['latest_optimization_20261025'].update(report=str(OUT/'index.html'),summary=str(OUT/'summary.json'),
    phase_RAW_main_ticks=1810,phase_RAW_scope='150 initial +4x400 same8s +60 side; repeated development recording, not independent data',
    four_phase_supported_same400=dict(single=156,whole_reset=145,partial_reset=149,parallel_recovery=161),
    parallel_primary_states_max_difference=0.,all_motion_forecasts_unchanged=True,
    side_native_ticks=60,side_periodicity=49,side_periodic_axis=21,side_four_phase=0,
    incremental_forecast_result='Same detector self-consistency only;0.1s tiny extra,0.2s worse,0.4s inactive; not integrated',
    phase_core_local_only=True,next_action='Installable low-latency motion core; no further physics weight sweeps')
h['next_action']='Extract pure live motion core and pinned licensed math into src package; actual-state equality and relocated wheel check; then consider remaining accuracy bottlenecks.'
p.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')
marker='### 持续优化补充：相位旁路落地与复锁结果（2026-10-25）'
log=WORK/'WORKLOG.md'
if marker not in log.read_text(encoding='utf-8'):
    with log.open('a',encoding='utf-8') as f:
        f.write('\n\n'+marker+'''\n\n本轮仍在继续，报告output/optimization_run_20261025/index.html，summary.json及event_drone_forecast/docs/OPTIMIZATION_JOURNAL_20261025.md可交接。

已直接编译运行未修改HelixTrack作者核心。正反初始相位假设、移动采样ROI、当前SDK像素周期交叉核验已落地。解析ROI预筛选与四桨并行保持480个相位状态最大差0，开发探针323.824→97.028ms。实际RAW主入口新增可选只读event_observer旁路，从当下NN及四个有界独立椭圆自动初始化，不读取旧框/旧几何/标签。3秒150时刻在22.86秒初始化，后105时刻98次四桨当前支持。

8秒同录像四轮各400时刻：单次初始化156次四桨支持；整套重置145、局部重置149，两者不采用；保留原轨迹并行恢复候选161。原轨迹相位与有效性逐项保持，支持增加5次，不等四电机正确率。四轮位置预测与156对应400帧完全相同。并行模块中位100.913ms、整链208.311ms，仍离线研究旁路，没证明轨迹精度或真实RPM/推力收益。

复用141原线轴算法，提取无状态measure_current，共享SDK包。真实侧视110—111.2秒60时刻：四桨0，整机当前周期49，周期投影线轴21；原位置预测逐项保持，整链73.233ms、观测模块5.685ms。法线无向，像素频率不是机械RPM，不能虚构四电机总推力。侧视/清晰视频与原预测叠图已保存且抽查。

新旋转特征因果重放在同一161个合格起点用当前同一NN框作反馈，属于检测器自洽：0.1s原/同门控纯偏差/旋转增量4.395/4.294/4.247px；0.2s7.454/7.468/7.527，退化；0.4s均16.650且不启用。无一致收益，不接主入口、不继续扫权重，不能与人工侧视或全400帧分数跨口径比较。

作者代码与DLL只限本地，research/helix_reference已被源码打包器明确排除；没有改作者仓库、没有G或yolov10-main访问/修改，E盘RAW只读。原运动默认138+140+144及156可选保持。当前所有RAW/NN/渲染子进程均完成。下一步正在提取可独立安装的live运动核心，摆脱历史research传递导入；旧source_candidate.zip仍是上一轮快照。
''')
print('Milestone checkpoint saved; goal remains active',flush=True)
