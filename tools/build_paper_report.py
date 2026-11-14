"""Build a self-contained local Chinese evidence report and scientific plots."""
from pathlib import Path
import csv,html,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/paper_reproduction_v030'
sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.motion import extrapolate
from paper_experiment import csvsave,readlines

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
def metric(data,sequence,subset,h,method):
    return next(r for r in data['metrics'] if r['sequence']==sequence and r['subset']==subset and r['horizon_s']==h and r['method']==method)

data=load(OUT/'evaluation.json');timing=load(OUT/'single_process_timing.json');controlled=load(OUT/'controlled_flight/result.json')
causality=load(OUT/'causality_validation.json');protocol=load(OUT/'experiment_protocol.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
(OUT/'figures').mkdir(exist_ok=True)
names=['cv','paper_nominal','paper_adaptive','paper_fused'];labels=['CV','Nominal thrust','Adaptive thrust','Gated fusion'];colors=['#278ab0','#af48a0','#a8a020','#eb8b25']
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,subset,title in zip(axes,['all','four_ellipses_physics'],['All available forecast targets','Four observed ellipses + valid physics']):
    for method,label,color in zip(names,labels,colors):
        rows=[metric(data,'all',subset,h,method) for h in (.1,.2,.4)]
        ax.plot([.1,.2,.4],[r['mean_error_px'] for r in rows],'-o',label=label,color=color)
    ax.set(title=title,xlabel='Future horizon (s)',ylabel='Mean endpoint error (px)');ax.grid(alpha=.2)
    counts=[metric(data,'all',subset,h,'cv')['available'] for h in (.1,.2,.4)]
    ax.text(.02,.98,'n = '+', '.join(map(str,counts)),transform=ax.transAxes,va='top',fontsize=9)
axes[0].legend(fontsize=8,loc='lower right');fig.suptitle('Real RAW regression: no future labels enter inference; conditional subsets differ')
fig.tight_layout();fig.savefig(OUT/'figures/endpoint_errors.png',dpi=160);plt.close(fig)

obs=readlines(OUT/'runs/train6/observations.jsonl');traces=readlines(OUT/'runs/train6/forecasts.jsonl')
diag=[]
for o,q in zip(obs,traces):
    if not 22<=o['t']<=28:continue
    for r in o.get('rotors',[]):
        d=r.get('frequency_diagnostics',{})
        diag.append(dict(t=o['t'],rotor_id=r['rotor_id'],observed=r['observed_now'],frequency_valid=r['frequency_valid'],
            blade_pass_hz=r['blade_pass_hz'],phase_coherence=d.get('phase_coherence'),harmonic_ambiguous=d.get('harmonic_ambiguous'),
            axis_ratio=r['ratio'],geometry_quality=r['geometry_quality']))
csvsave(OUT/'four_rotor_signal_diagnostics.csv',diag)
fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
for identifier in sorted(set(r['rotor_id'] for r in diag)):
    rows=[r for r in diag if r['rotor_id']==identifier and r['frequency_valid']]
    if len(rows)<5:continue
    axes[0].plot([r['t'] for r in rows],[r['blade_pass_hz'] for r in rows],'.',ms=2,label=f'visual ID {identifier}')
axes[0].set(ylabel='Blade-pass Hz',title='Real four-rotor interval: frequencies are provisional, not RPM ground truth');axes[0].legend(ncol=4,fontsize=8)
selected=[q for q in traces if 22<=q['t']<=28 and q.get('active')]
for method,label in [('paper_nominal','nominal'),('paper_adaptive','adaptive')]:
    axes[1].plot([q['t'] for q in selected],[q.get('thrust_ratio',1) for q in selected],label='sum f squared / reference' if method=='paper_nominal' else '_nolegend_') if method=='paper_nominal' else None
    axes[2].plot([q['t'] for q in selected],[q['physical_parameters'][method]['thrust_gain_px_s2'] for q in selected],label=label+' thrust gain')
axes[1].set(ylabel='Relative thrust cue');axes[2].set(ylabel='Pixel gain (px/s²)',xlabel='RAW time (s)');axes[1].legend(fontsize=8);axes[2].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'figures/rotor_signals.png',dpi=160);plt.close(fig)

# Actual center errors paired at the same sparse target timestamps.
with (OUT/'all_endpoint_errors.csv').open(encoding='utf-8-sig') as f:errors=list(csv.DictReader(f))
lookup={(r['sequence'],r['target_time_s'],r['horizon_s'],r['method']):r for r in errors if r.get('error_px')}
paired=[]
for r in errors:
    if r['method']!='paper_fused' or r['horizon_s']!='0.4' or not r.get('error_px'):continue
    cv=lookup[(r['sequence'],r['target_time_s'],r['horizon_s'],'cv')]
    paired.append(dict(sequence=r['sequence'],target_time_s=float(r['target_time_s']),anchor_time_s=float(r['anchor_time_s']),
        cv_error_px=float(cv['error_px']),fused_error_px=float(r['error_px']),four_ellipses=r['four_ellipses'],physics_active=r['physics_active']))
csvsave(OUT/'paired_04s_errors.csv',paired)
fig,ax=plt.subplots(figsize=(6,5))
for four,color,label in [('False','#a8afb4','Other available targets'),('True','#e18c24','Four-ellipse targets')]:
    points=[r for r in paired if r['four_ellipses']==four]
    ax.scatter([r['cv_error_px'] for r in points],[r['fused_error_px'] for r in points],s=18,alpha=.7,label=label,color=color)
lim=max(max(r['cv_error_px'],r['fused_error_px']) for r in paired)*1.03;ax.plot([0,lim],[0,lim],'k--',lw=1)
ax.set(xlabel='CV endpoint error (px)',ylabel='Fused endpoint error (px)',title='Paired real 0.4s errors: below diagonal favors fusion');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'figures/paired_errors.png',dpi=160);plt.close(fig)

# Public benchmark: report unrounded estimator values as well as author output.
public=[]
for name in ('velcro_front','velcro_side'):
    r=load(OUT/f'eeppr_benchmark/{name}_result.json');f=np.load(OUT/f'eeppr_benchmark/{name}_window_estimates.npy');estimate=float(np.median(f[f>0]))
    r.update(unrounded_median_hz=estimate,unrounded_absolute_error_hz=abs(estimate-r['gt_freq_hz']),
             unrounded_relative_error_percent=100*abs(estimate/r['gt_freq_hz']-1))
    public.append(r)
(OUT/'eeppr_benchmark/combined_results.json').write_text(json.dumps(public,indent=2),encoding='utf-8')

stats=data['run_statistics'];total_events=sum(r['event_count'] for r in stats);total_frames=sum(r['frames'] for r in stats)
subset_rows=[]
for h in (.1,.2,.4):
    for subset,description in [('all','全部可预测'),('physics_active','物理观测有效'),('four_ellipses_physics','四椭圆+物理有效')]:
        rows=[metric(data,'all',subset,h,m) for m in names];base=rows[0]
        subset_rows.append([h,description,f"{base['available']}/{base['targets']}",*[fmt(r['mean_error_px']) for r in rows]])
all04=metric(data,'all','all',.4,'paper_fused');cv04=metric(data,'all','all',.4,'cv');four04=metric(data,'all','four_ellipses_physics',.4,'paper_fused');fourcv=metric(data,'all','four_ellipses_physics',.4,'cv')
four_improvement=100*(1-four04['mean_error_px']/fourcv['mean_error_px'])
with (OUT/'common_targets_previous_versions.csv').open(encoding='utf-8-sig') as f:common=list(csv.DictReader(f))
common_table=[[r['previous'],r['subset'],r['n'],fmt(float(r['previous_mean_error_px'])),fmt(float(r['current_mean_error_px']))]
    for r in common if r['horizon_s']=='0.4' and r['method']=='paper_fused' and r['n']!='0']
sequence_table=[]
for r in stats:
    score=metric(data,r['sequence'],'all',.4,'paper_fused')
    sequence_table.append([r['sequence'],f"{r['event_count']:,}",r['frames'],r['four_ellipse_frames'],r['physics_active_frames'],
        r['arc_observation_frames'],r['whole_recording_physical_updates'],f"{score['available']}/{score['targets']}",fmt(score['mean_error_px'])])

# Fixed intervals are diagnosed without selecting on future prediction errors.
interval_rows=[]
for title,start,end in [('四桨清晰',22,28),('近侧视',109,114)]:
    current=[o for o in obs if start<o['t']<=end];qs=[q for q in traces if start<q['t']<=end]
    interval_rows.append([title,f'{start}–{end}s',len(current),sum(o.get('four_ellipses',False) for o in current),sum(q.get('physics_active',False) for q in qs)])
content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>事件相机无人机预测 · v0.3 论文复现实验</title><style>
body{{font-family:"Microsoft YaHei",sans-serif;margin:0;background:#f4f6f8;color:#172533;line-height:1.75}}main{{max-width:1160px;margin:auto;padding:32px}}
h1{{font-size:30px;margin:0}}h2{{margin-top:34px;font-size:22px}}p{{max-width:1050px}}a{{color:#126b99}}.card{{background:white;border:1px solid #dce3e9;padding:22px 26px;margin:18px 0;border-radius:10px}}
.note{{border-left:4px solid #d99b32;background:#fff8e7;padding:13px 18px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #dce3e9;padding:8px;text-align:left}}th{{background:#eaf0f4}}
.scroll{{overflow-x:auto}}img,video{{max-width:100%;height:auto}}code{{background:#e8edf2;padding:2px 5px}}small{{color:#586777}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}main{{padding:16px}}}}
</style><main><h1>事件相机无人机预测：v0.3 论文结构实验</h1>
<p>先完成“桨盘→推力轴、周期→相对推力、两滤波器→未来位置”的可运行链路，再用数据判断效果。2026-10-16 · 输入仅 RAW · 10ms 发布步长 · 预测 0.1/0.2/0.4 秒。</p>
<div class="note"><b>当前结论：</b>四椭圆且物理有效的 0.4 秒子集中，融合误差 {four04['mean_error_px']:.2f}px，CV 为 {fourcv['mean_error_px']:.2f}px，降低 {four_improvement:.1f}%。但只有 {four04['available']} 个稀疏目标时刻，全部来自已用于开发的 train6。全部可预测样本上没有改善：{all04['mean_error_px']:.2f}px 对 {cv04['mean_error_px']:.2f}px。尚不能称超过论文，也未验证真实转速/姿态准确性。</div>
<p><a href="PAPER_REPRODUCTION.md">逐项论文对照与运行说明</a> · <a href="RELATED_WORK_AUDIT.md">论文/GitHub 核查</a> · <a href="实验报告.md">完整文字报告</a> · <a href="metrics.csv">全部指标 CSV</a> · <a href="all_endpoint_errors.csv">每个未来端点评分</a></p>
<section class="card"><h2>看实际预测</h2><p>左侧：在当前时刻已经发布的预测；右侧：0.4 秒之后的实际 RAW。右侧只用于展示，没有反馈进当时的预测。蓝：CV；粉：名义推力模型；橙：因果融合。频率标签为叶片通过 Hz，机械 RPM 尚未实测验证。</p>
<div class="grid"><div><h3>四桨清晰：train6 22–28 秒</h3><video controls preload="metadata" poster="visuals/four_rotors/comparison_3.png" src="visuals/four_rotors/current_vs_future.mp4"></video><a href="visuals/four_rotors/current_vs_future.gif">查看动图</a></div>
<div><h3>近侧视：train6 109–114 秒</h3><video controls preload="metadata" poster="visuals/side_view/comparison_3.png" src="visuals/side_view/current_vs_future.mp4"></video><a href="visuals/side_view/current_vs_future.gif">查看动图</a></div></div>
{table(['固定片段','RAW 时间','发布次数','四椭圆次数','物理有效次数'],interval_rows)}
<p>侧视时四桨总推力往往没有足够观测，融合会转用 CV。画面中仍可能把机身活动误拟合为椭圆，或在丢失重获时发生 ID 变化；这些不能算正确姿态恢复。</p></section>
<section class="card"><h2>真实 RAW 定量结果</h2><p>四段完整录像，共 {total_events:,} 个事件、{total_frames:,} 次发布（约 6 分钟）。这四段均曾用于工程观察，属于回归实验，不能当成新的独立测试集。只在实际人工标注时刻评分，不插值真值；单位均为原生 1280×720 图像中的像素。</p>
<div class="scroll">{table(['预测秒数','发布时观测子集','可预测/目标数','CV','名义推力','自适应推力','融合'],subset_rows)}</div>
<p>“全部”包含物理无效时回退 CV；“物理有效”和“四椭圆”分别筛选，不把回退成绩冒称物理效果。0.4 秒全部目标共 {all04['targets']} 个，可预测 {all04['available']} 个，覆盖率 {all04['coverage']:.1%}。四椭圆子集只占全部目标的 {four04['targets']/all04['targets']:.1%}，并非人工确认的四桨可见真值。</p>
<img src="figures/endpoint_errors.png" alt="各预测时长的端点误差"><div class="scroll">{table(['录像','事件数','发布数','四椭圆','物理有效','残弧帧','累计参数更新','0.4秒覆盖','融合误差px'],sequence_table)}</div>
<p>累计参数更新跨连续跟踪段统计；原始 summary.json 的 physical_updates 只记录最后一段，不能当作全录像总数。2、3、4 中四桨高质量观测不足，学习多数时间冻结。远处小目标的 10ms 检测窗口仍易漏检，0.4 秒平均误差也受重获后的速度估计和急转弯影响。</p>
<img src="figures/paired_errors.png" alt="相同目标时刻CV与融合误差散点"><h3>与旧版在共同可预测目标上比较（0.4 秒）</h3>
{table(['旧版','子集','共同目标数','旧版 adaptive 误差px','本版融合误差px'],common_table)}
<p>每行只比较该旧版和本版的交集；不同旧版的交集未必相同。完整结果不支持“普遍精度提升”，清晰四椭圆条件下的小样本改善值得进一步独立验证。</p></section>
<section class="card"><h2>中间物理量与论文差距</h2><img src="figures/rotor_signals.png" alt="各桨周期、相对推力、有效增益">
<p>曲线显示的是测量与模型输出，频率连续不等于测得准确。22秒附近有启动频率跳变；首次四频率建立的参考可能被早期谐波/短暂误检污染，参考也没有悬停真值保证。后续相对推力明显下移，而自适应增益上升，这更可能反映频率参考、尺度与残差的互相补偿，不能称参数已经收敛到物理真值。当前仍可能发生谐波误选、视觉 ID 更换、弱透视偏差；真实重力投影和相机自运动也未验证，因而自适应分支不保证更好。</p>
<p>本轮实现：稳定逐桨 ID、象限 FFT/频率 KF、相位一致性、频率平方和、推力轴 KF、受推力驱动的位置 KF、多步积分、残弧/轴线和因果融合。与原文的主要差距：缺真实相机 K/观察者位姿/物理尺度；默认姿态是弱透视线索；图像左右频率差没有经过机体系 roll/yaw 标定；没有原文同步 RTK、IMU、RPM 真值。米制物理内核有测试，但真实 RAW 默认走二维像素模型。</p>
<p>假设质量/推力系数/风速全未知时，首先能估计的是有效推力增益和残余加速度。质量与推力系数不能靠同一条单目运动轨迹任意拆开，风和投影变化也可能混淆。画面中桨盘像“木板”时，其法向约束推力方向；它的倾斜方向并不直接等于速度方向。</p></section>
<section class="card"><h2>可控实验与公开方法</h2><p>已知姿态、频率、像素尺度的观测级动力学实验中，0.4 秒名义推力模型误差为 1.35px，CV 为 2.93px。这证明本轮实现确实把姿态/推力送入未来位置预测。它没有经过 RAW 成像或检测，且名义尺度匹配真值，不能拿来代替实飞准确率。</p>
<p>实际调用 <a href="https://github.com/JackPieCZ/EEPPR">EEPPR 原作者核心代码</a>，公共正视/侧视数据使用作者给定 ROI、1 秒、100us 分箱、45px 窗、1800 事件模板和中位数。SDK 桥接已与真实 SDK 比较两个序列各 10,000 个批次，事件及 ROI 一致；算法源文件未改动。</p>
{table(['公共旋转目标','标注Hz','原代码输出Hz(1位小数)','未舍入中位Hz','未舍入相对误差','有效/总窗口','本机整次耗时s'],[[r['sequence'],r['gt_freq_hz'],r['estimated_hz'],f"{r['unrounded_median_hz']:.5f}",f"{r['unrounded_relative_error_percent']:.4f}%",f"{r['valid_windows']}/{r['total_windows']}",fmt(r['wall_s'])] for r in public])}
<p>公共数据是固定旋转目标，不是飞行无人机；两次耗时含不同程度的依赖/JIT/并行争用，不能用来作正侧视速度比较或正式实时基准。结果支持侧视仍能取到周期，不支持侧视已能准确恢复全部姿态/推力。<a href="eeppr_benchmark/combined_results.json">原始数值与源码指纹</a> · <a href="eeppr_benchmark/sdk_bridge_validation.json">IO 一致性验证</a></p>
<p><a href="https://arxiv.org/html/2603.09235v1">HelixTrack §4.8</a> 明确列出极端侧视和重遮挡的限制，原 C++ 系统本机尚未构建；这里只使用相位一致性思路。<a href="https://arxiv.org/html/2603.01997v1">RPM-Modulated KF</a> 已做二维未来轨迹预测，但评测过去位置/速度来自真实标注。EventPro、EVPropNet、EVPeriscope、frequency_cam、EV-Tach、EventRadar 逐项状态见核查文档，未把“读过/下载”写成“已复现”。</p></section>
<section class="card"><h2>速度、因果性和交付</h2><p>单进程固定高事件率区间 109.3–112.3 秒：{timing['event_count']:,} 个事件；3 秒传感器时间用时 {timing['hot_path_wall_s']:.2f} 秒，单批中位 {timing['processing_median_ms']:.2f}ms，P95 {timing['processing_p95_ms']:.2f}ms。包括解码、检测、周期和八模型，不含可视化和日志写入；RAW 打开/跳转另计 {timing['raw_open_seek_s']:.2f} 秒。<b>目前不实时。</b></p>
<p>25 项单元测试通过；真实 RAW 4 秒前缀的 400 次观测和预测与完整运行前缀逐行一致。源码 SHA：<code>{protocol['source_sha256']}</code>。未来真实事件仅用于离线右侧画面和独立评分，不进入发布时估计。</p>
<p><a href="unit_tests.txt">测试记录</a> · <a href="causality_validation.json">前缀一致性</a> · <a href="single_process_timing.json">独立计时</a> · <a href="experiment_protocol.json">冻结协议</a> · <a href="../event_drone_forecast_v0.3.0.zip">源码与实验汇总包</a></p>
<p>下一步优先把独立测速/姿态真值和四桨独立录像实验补齐，定位到底是谐波、投影、动力学还是观测误差；再做事件时间与退化几何联合估计、可见桨数切换、跟踪恢复和观测者运动补偿。已有论文覆盖了其中不少单项方法，是否形成创新需要强基线和消融证据。</p></section></main></html>'''
(OUT/'查看论文复现实验.html').write_text(content,encoding='utf-8')

md=f'''# 事件相机无人机预测 v0.3 实验报告

已实现独立的逐桨周期—推力轴—位置双滤波—未来积分链路，输入只需 RAW。不能因论文未单列未来时长成绩就认为它不能预测；本项目将其动力学结构明确推进到 0.1/0.2/0.4 秒。

## 真实录像结果

完整四段 RAW 约 6 分钟，共 {total_events:,} 个事件、{total_frames:,} 次 10ms 发布。四段已被以前版本观察，属于工程回归而不是新的未见测试集。

0.4 秒全部可预测目标：{all04['available']}/{all04['targets']}，覆盖率 {all04['coverage']:.1%}。CV 误差 {cv04['mean_error_px']:.2f}px；名义推力 {metric(data,'all','all',.4,'paper_nominal')['mean_error_px']:.2f}px；自适应推力 {metric(data,'all','all',.4,'paper_adaptive')['mean_error_px']:.2f}px；融合 {all04['mean_error_px']:.2f}px。总体没有改善。

四椭圆且物理有效子集：只有 {four04['available']} 个稀疏目标时刻，全部来自开发录像 train6；CV {fourcv['mean_error_px']:.2f}px，名义推力 {metric(data,'all','four_ellipses_physics',.4,'paper_nominal')['mean_error_px']:.2f}px，自适应推力 {metric(data,'all','four_ellipses_physics',.4,'paper_adaptive')['mean_error_px']:.2f}px，融合 {four04['mean_error_px']:.2f}px（比 CV 低 {four_improvement:.1f}%）。样本少且时间相关，不支持泛化或超越论文结论。

物理有效但不限四椭圆的子集有 {metric(data,'all','physics_active',.4,'cv')['available']} 个目标，CV {metric(data,'all','physics_active',.4,'cv')['mean_error_px']:.2f}px，融合 {metric(data,'all','physics_active',.4,'paper_fused')['mean_error_px']:.2f}px。回退 CV 的成绩没有冒充物理成绩。全部指标与每个时刻见 metrics.csv / all_endpoint_errors.csv。

## 公开代码实验与可控验证

EEPPR 原作者核心未修改，公共正视真值21.1Hz→输出21.1Hz，侧视26.3Hz→输出26.3Hz；输出本身舍入到一位小数，非零误差的未舍入值见 combined_results.json。两个序列各10,000个100us批次与原 SDK 完全一致，矩形 ROI 也已验证。这是固定目标周期实验，不是无人机姿态或飞行预测真值。

受控观测实验：0.4秒 CV 2.93px，名义物理1.35px，融合1.48px；它预先给定正确几何/频率观测与匹配的名义尺度，不测试 RAW 识别。25项测试和真实RAW前缀400帧逐行一致均通过。

## 现实限制

四桨频率和姿态没有独立实测真值，默认使用两叶桨、画面上方、名义像素重力和弱透视。K/观察者位姿/米制尺度未提供；图像左右频差不是标定机体系 roll 输入，米制物理内核的单独测试不能算真实RAW米制复现。完整论文差异见 PAPER_REPRODUCTION.md。

总体误差未改善的主要可见原因：远处小目标10ms窗口漏检；机身与桨盘可能误分；丢失重获造成速度不稳；频率谐波/身份变化；真实姿态与投影模型偏差；未知尺度、风、相机运动和动力学混淆；未来控制不可观。当前证据不能给这些误差来源分配可靠的百分比。

侧视依然能留下周期或轴线，但不足以唯一恢复椭圆、推力总量和速度。不能将“木板倾斜方向”直接当飞行方向。侧视对照视频中会明确显示物理无效时的CV回退和失败。

单进程3秒高事件率RAW用时 {timing['hot_path_wall_s']:.2f}s，单批中位{timing['processing_median_ms']:.2f}ms、P95 {timing['processing_p95_ms']:.2f}ms，不含预览和日志；尚不实时。

## 文件与下一步

查看论文复现实验.html 包含两段真实未来对照视频、评分表、曲线、公开实验及证据链接。源码、配置、测试、运行日志、每时刻预测CSV和冻结指纹都已保存；旧版本保留。

优先做独立四桨录像和台架/同步测速姿态验证，随后消融纯时间、纯几何、物理模型与在线适应；以 HelixTrack 等强公开方法验证退化几何和恢复，而不是现在就声称首次创新。真正源码可用性与侧视边界见 RELATED_WORK_AUDIT.md。
'''
(OUT/'实验报告.md').write_text(md,encoding='utf-8')
for name in ('PAPER_REPRODUCTION.md','RELATED_WORK_AUDIT.md'):
    (OUT/name).write_bytes((ROOT/name).read_bytes())
summary=dict(total_events=total_events,total_frames=total_frames,all_04s=all04,all_cv_04s=cv04,four_04s=four04,
    four_cv_04s=fourcv,four_improvement_percent=four_improvement,four_subset_development_only=True,
    unit_tests_passed=25,causality=causality,real_metric_pose_rpm_validated=False,claim_exceeds_paper=False)
(OUT/'result_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
