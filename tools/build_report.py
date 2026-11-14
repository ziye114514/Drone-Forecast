"""Build an auditable Chinese result page and scientific figures from frozen runs."""
from pathlib import Path
import json,csv,sys,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.motion import METHODS,extrapolate

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/raw_only_project'
LABELS={'cv':'CV','ca':'CA','adaptive_history':'History online','adaptive_rotor':'Rotor online','adaptive':'Adaptive mixture'}
COLORS={'cv':'#2596be','ca':'#9ca3af','adaptive_history':'#a855b5','adaptive_rotor':'#23985f','adaptive':'#ee8a22'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def load_csv(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    runs={}
    for sid in (6,2,3,4):
        folder=OUT/'runs'/f'train{sid}'
        if not (folder/'evaluation.json').exists():continue
        runs[f'train{sid}']=dict(folder=folder,summary=json.loads((folder/'summary.json').read_text()),
            evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8')),
            details=load_csv(folder/'evaluation_details.csv'))
    held=[k for k in runs if k!='train6'];rows=[r for k in held for r in runs[k]['details']]
    aggregate=[]
    for h in (.1,.2,.4):
        for m in METHODS:
            allrows=[r for r in rows if r['method']==m and float(r['requested_horizon_s'])==h]
            valid=[r for r in allrows if r['available']=='True']
            a=np.array([float(r['error_px']) for r in valid])
            aggregate.append(dict(method=m,horizon_s=h,targets=len(allrows),available=len(valid),coverage=len(valid)/max(len(allrows),1),
                mean_error_px=float(a.mean()) if len(a) else None,median_error_px=float(np.median(a)) if len(a) else None,
                p90_error_px=float(np.percentile(a,90)) if len(a) else None))
    (OUT/'aggregate_metrics.json').write_text(json.dumps(aggregate,indent=2))
    with (OUT/'aggregate_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    # Cluster bootstrap by sequence and 5-second time blocks, conditional on common availability.
    bootstrap=[];rng=np.random.default_rng(20261015)
    for h in (.1,.2,.4):
        groups={}
        for m in ('cv','adaptive','adaptive_rotor','adaptive_history'):
            for r in rows:
                if r['method']==m and r['available']=='True' and float(r['requested_horizon_s'])==h:
                    groups.setdefault((r['sequence'],r['target_time_s']),{})[m]=float(r['error_px'])
        blocks={}
        for (seq,t),values in groups.items():
            if len(values)==4:blocks.setdefault((seq,int(float(t)//5)),[]).append(values)
        keys=list(blocks)
        improvements=[]
        if keys:
            for _ in range(2000):
                sample=[v for i in rng.integers(0,len(keys),len(keys)) for v in blocks[keys[i]]]
                cv=np.mean([v['cv'] for v in sample]);full=np.mean([v['adaptive'] for v in sample])
                improvements.append(100*(cv-full)/max(cv,1e-9))
        cv=next(r for r in aggregate if r['method']=='cv' and r['horizon_s']==h)['mean_error_px']
        full=next(r for r in aggregate if r['method']=='adaptive' and r['horizon_s']==h)['mean_error_px']
        bootstrap.append(dict(horizon_s=h,improvement_percent=100*(cv-full)/cv if cv else None,
            block_count=len(keys),interval95_percent=np.percentile(improvements,[2.5,97.5]).tolist() if improvements else None,
            interpretation='Exploratory 5s block bootstrap, conditional coverage; only three sequences, not proof of population generalization.'))
    (OUT/'paired_bootstrap.json').write_text(json.dumps(bootstrap,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ax,h in zip(axes,(.1,.2,.4)):
        vals=[next(r for r in aggregate if r['method']==m and r['horizon_s']==h) for m in METHODS]
        ax.bar(range(len(METHODS)),[r['mean_error_px'] for r in vals],color=[COLORS[m] for m in METHODS])
        ax.set_ylim(0,max(r['mean_error_px'] for r in vals)*1.2)
        ax.set_xticks(range(len(METHODS)),['CV','CA','History','Rotor','Mixture'],rotation=25)
        ax.set_title(f'Future endpoint +{h:.1f}s');ax.set_ylabel('Mean error (native pixels)')
        ax.text(.02,.98,f"available {vals[0]['available']}/{vals[0]['targets']} ({vals[0]['coverage']:.1%})",transform=ax.transAxes,va='top')
    fig.suptitle('Held-out recordings: same raw-event detector, same forecast anchors')
    fig.savefig(OUT/'prediction_metrics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    seqs=list(runs)
    axes[0].bar(seqs,[runs[s]['evaluation']['detection']['detector_presence_recall']*100 for s in seqs],color=['#96a1a9']+['#2596be']*(len(seqs)-1))
    axes[0].set_ylabel('Human positive labels detected (%)');axes[0].set_ylim(0,100)
    axes[0].set_title('Detection availability; train6 is development')
    for m in ('cv','adaptive'):
        axes[1].plot([.1,.2,.4],[next(r for r in aggregate if r['method']==m and r['horizon_s']==h)['mean_error_px'] for h in (.1,.2,.4)],'o-',color=COLORS[m],label=LABELS[m])
    axes[1].set_xlabel('Forecast horizon (s)');axes[1].set_ylabel('Mean error (pixels)');axes[1].legend()
    fig.savefig(OUT/'coverage_and_error.png',dpi=160);plt.close(fig)
    # Representative cases chosen by error quantiles, not manually selected success examples.
    fig,axes=plt.subplots(len(held),3,figsize=(12,max(4,3.4*len(held))),squeeze=False,layout='constrained')
    cases=[]
    for row,seq in enumerate(held):
        run=runs[seq];details=[r for r in run['details'] if r['method']=='adaptive' and r['available']=='True' and float(r['requested_horizon_s'])==.4]
        details.sort(key=lambda r:float(r['error_px']))
        traces=[json.loads(s) for s in (run['folder']/'forecasts.jsonl').read_text().splitlines()]
        obs=[json.loads(s) for s in (run['folder']/'observations.jsonl').read_text().splitlines()]
        tt=np.array([r['t'] for r in traces])
        for col,q in enumerate((.1,.5,.9)):
            ax=axes[row,col]
            if not details:ax.set_axis_off();continue
            selected=details[int(round(q*(len(details)-1)))];t=float(selected['anchor_time_s']);index=int(np.argmin(abs(tt-t)));tr=traces[index]
            history=np.array([r['center'] for r in obs if t-.7<=r['t']<=t and r['detected']])
            if len(history):ax.plot(history[:,0],history[:,1],'.-',color='#59626c',label='past detections')
            horizon=float(selected['actual_horizon_s'])
            for m in ('cv','adaptive'):
                pts=np.array([extrapolate(tr['states'][m],x) for x in np.linspace(0,horizon,21)])
                ax.plot(pts[:,0],pts[:,1],color=COLORS[m],label=LABELS[m])
                ax.plot(*pts[-1],'x',color=COLORS[m],ms=8)
            ax.scatter(float(selected['gt_x']),float(selected['gt_y']),c='#d93b56',marker='*',s=110,label='human future box center')
            ax.set_title(f'{seq} t={t:.2f}s | error quantile {q:.0%}\nMixture error {float(selected["error_px"]):.1f}px')
            ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('x / px');ax.set_ylabel('y / px')
            cases.append(dict(selected,selection_quantile=q))
    axes[0,0].legend(fontsize=7,loc='best')
    fig.savefig(OUT/'representative_cases.png',dpi=160);plt.close(fig)
    (OUT/'representative_cases.json').write_text(json.dumps(cases,indent=2))
    # Diagnostics: these are unvalidated visual proxies and learned image-space parameters.
    o=[json.loads(s) for s in (runs['train6']['folder']/'observations.jsonl').read_text().splitlines()]
    tr=[json.loads(s) for s in (runs['train6']['folder']/'forecasts.jsonl').read_text().splitlines()]
    visible=[r for r in o if r['detected']]
    fig,ax=plt.subplots(3,1,figsize=(12,7),sharex=True,layout='constrained')
    ax[0].plot([r['t'] for r in visible],[r.get('blade_pass_hz',np.nan) or np.nan for r in visible],'.',ms=1.5)
    ax[0].set_ylabel('Blade-pass proxy / Hz');ax[0].set_title('Development diagnostics: no RPM / attitude / wind ground truth')
    ax[1].plot([r['t'] for r in visible],[r['tilt_proxy_deg'] for r in visible],'.',ms=1.5,color='#23985f');ax[1].set_ylabel('Weak-perspective tilt / deg')
    valid=[r for r in tr if 'expert_weights' in r]
    for m in ('cv','ca','adaptive_history','adaptive_rotor'):
        ax[2].plot([r['t'] for r in valid],[r['expert_weights'][m] for r in valid],color=COLORS[m],label=LABELS[m],lw=1)
    ax[2].set_ylabel('Online mixture weight');ax[2].set_xlabel('RAW time / s');ax[2].legend(ncol=4,fontsize=8)
    fig.savefig(OUT/'online_diagnostics.png',dpi=160);plt.close(fig)
    total_duration=sum(r['summary']['processed_sensor_seconds'] for r in runs.values())
    total_events=sum(r['summary']['event_count'] for r in runs.values())
    table='\n'.join('| '+ ' | '.join([f'{r["horizon_s"]:.1f}',str(r['available'])+'/'+str(r['targets']),f'{r["coverage"]:.1%}',f'{next(x for x in aggregate if x["method"]=="cv" and x["horizon_s"]==r["horizon_s"])["mean_error_px"]:.2f}',f'{r["mean_error_px"]:.2f}'])+' |' for r in aggregate if r['method']=='adaptive')
    seqtable='\n'.join(f'| {s} | {"开发" if s=="train6" else "本原型留出"} | {r["evaluation"]["detection"]["positive_labels"]} | {r["evaluation"]["detection"]["detector_presence_recall"]:.1%} | {r["evaluation"]["detection"]["recall_iou_0_3"]:.1%} |' for s,r in runs.items())
    report=f'''# RAW-only 事件无人机预测项目：实测结果

日期：2026-10-15。项目源代码：`D:/drone forecast/event_drone_forecast`。

已完成从 RAW 自动检测、桨盘几何/周期提取、因果跟踪、在线更新、未来预测到独立评测的运行闭环。**这是可复跑的二维研究原型，尚未证明桨叶在线模型具有稳定优于普通运动模型的效果，也未达到通用连续实时三维预测。**

## 实际数据与输入

共 {len(runs)} 段完整自采录像，{total_duration:.0f} 秒，实际处理 {total_events:,} 个原始事件。预测器唯一外部传感器输入为每段 RAW。未读取人工框、RGB 视频、离线重建或未来 RPM。

train/6 为开发录像；train/2、3、4 在本次算法固定后评分。这些录像在原数据集中名称仍为 train，且早前做过一般数据查看，因此不宣称官方独立测试集、完全未见数据或跨机型泛化。

## 留出录像：未来位置端点误差

1280×720 原生像素。只对真实人工框的未来时刻评分，未插值真值。实际预测时长为请求时长到请求时长加 20 ms。覆盖率以全部阳性未来标签为分母；误差仅对有预测的样本统计，因此必须和覆盖率一起看。

| 请求预测时长/s | 有预测/阳性标签 | 覆盖率 | 恒速 KF 均值/px | 完整自适应均值/px |
|---|---|---|---|---|
{table}

![各方法误差](prediction_metrics.png)

完整五种方法、逐录像/逐时刻误差、大误差尾部见 `aggregate_metrics.csv` 和各运行目录 `metrics.csv`、`evaluation_details.csv`。`paired_bootstrap.json` 提供按序列和 5 秒区块的探索性重采样区间；只有三段留出录像，不能据此宣称普遍显著提升。

## 检测覆盖与漏检

| 录像 | 角色 | 人工阳性时刻 | 发现目标比例 | 框 IoU≥0.3 的比例 |
|---|---|---|---|---|
{seqtable}

“发现目标比例”不等于检测正确率；严格些的定位召回由 IoU 列反映。阴性时刻的误报率另见各录像 `evaluation.json`。不能只报成功时刻的误差而省略漏检。

![覆盖和预测误差](coverage_and_error.png)

## 观看结果

- `visuals/train6/current_vs_future.mp4`：开发录像清晰片段。
- `visuals/train3/current_vs_future.mp4`：另一录像的同规则片段。
- `visuals/train6_close/current_vs_future.mp4`：此前已确认清晰的 109–114 秒近景；按时间指定，包含预测偏差与离开画面的情况。
- 两段均按固定时长内桨盘/目标可见性选取，不根据未来误差挑最好样例。左边为已发布预测，右边为 0.4 秒后真实 RAW；十字为当时已经作出的预测。未来画面只用于这个离线展示。
- 播放帧率是视频时间轴，不是算法推理速度。可视化 manifest 记录具体选片规则与时间。

## 成功、中等与失败的量化样例

每段留出录像按完整方法 0.4 秒误差的 10%、50%、90% 分位选取，避免只展示成功样例。星号是已有人工未来框中心；图中连线是预测，不是插值的真实未来轨迹。

![代表样例](representative_cases.png)

## 在线更新实际在做什么

两条 RLS 支路分别使用历史运动，以及历史运动加桨盘方向/相对频率平方。新事件到来后，先对已经结束的过去预测计算残差，再更新用于后续预测的模型。自适应组合器也只比较已结束的预测，在 CV、CA 和两条学习支路间调整权重。

权重是图像投影关系，不是实测质量、推力系数或风速。周期为未用测速真值验证的叶片通过频率代理；椭圆是未用标定/姿态真值验证的弱透视倾斜线索。它们的跳变是下一阶段应解决的测量问题，不能把曲线当作精确物理估计。

![在线诊断](online_diagnostics.png)

## 为什么还会差

1. 小目标、近侧视、遮挡及背景导致桨盘分割不连续；留出录像的低覆盖首先限制可用性。
2. 桨盘不完整时，用布局恢复目标框仍有偏差；人工框中心也不完全等同于物理质心。
3. 局部 FFT 可能选到谐波或运动引起的峰，转速线索没有逐桨真值验证。
4. 图像加速度包含深度和相机运动；未标定的椭圆法向不能直接转成世界推力。
5. 预测误差无法自动区分参数错误、扰动、检测错误和未来动作；线上拟合有时反而放大长时预测误差。
6. 当前看到的转速不能告诉模型 0.4 秒内飞控下一步会如何改变动作。

## 当下值得做的创新工作

**优先级 1：质量感知的桨叶测量与更新。** 用相位一致性、跨桨谐波约束、多桨几何一致性判断何时值得相信 RPM/椭圆，何时只跟踪机体并冻结物理参数。重点验证能否提高覆盖并避免错误在线适应；普通置信度门控本身不是新意。

**优先级 2：尺度与有效推力、扰动的可辨识建模。** 明确哪些参数组合能从外部观测确定，在条件不足时输出区间/多假设；研究可检验的约束，而非让质量、系数和风扰自由互相抵消。没有额外信息时不能保证米制尺度恢复。

**优先级 3：检验桨叶信息的提前量。** 同一检测器、同一锚点，比较“只看运动历史”和“加椭圆/RPM”的多步误差，包含谐波扰动、错误旋翼关联及特征打乱等负对照。当前结果还不足以证明稳定增益，这正是需要回答的问题。

**优先级 4：建立真正的未知机型/载荷/风扰测试。** 采集独立 RPM、姿态和位置真值只作评测，算法仍只用 RAW；按机型/采集日分组，统计适应耗时、失效尾部和延迟。若只有一架机体，不能以不同录像替代跨机型实验。

在线 RLS、双时间尺度、KF 加残差和自适应模型组合均已有先例；本项目的整合本身不作为论文首创。

## 复核入口

`experiment_protocol.json` 为冻结配置及代码指纹；每段 `run_manifest.json` 列出 RAW-only 输入声明；`causality_validation.json` 为真实 RAW 前缀一致性检查；源代码测试在项目 tests 目录。完整运行日志、观测和预测均保留。
'''
    (OUT/'实验结果与创新方向.md').write_text(report,encoding='utf-8')
    def metric_html():
        return ''.join(f'<tr><td>{r["horizon_s"]:.1f}s</td><td>{r["available"]}/{r["targets"]}</td><td>{r["coverage"]:.1%}</td><td>{next(x for x in aggregate if x["method"]=="cv" and x["horizon_s"]==r["horizon_s"])["mean_error_px"]:.2f}px</td><td>{r["mean_error_px"]:.2f}px</td></tr>' for r in aggregate if r['method']=='adaptive')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>事件无人机预测：RAW-only 实测</title>
<style>body{{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f7;color:#192635;margin:0}}main{{max-width:1120px;margin:auto;padding:35px 24px}}h1{{font-size:32px}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}img,video{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff3dd;border-left:5px solid #e18a20;padding:16px}}a{{color:#126cb3}}code{{overflow-wrap:anywhere}}</style>
<main><h1>事件相机无人机运动预测</h1><p>只读取 RAW · 自动桨盘检测 · 在线适应 · 0.1 / 0.2 / 0.4 秒未来像素位置</p>
<p class="note">已打通可复跑研究原型。当前没有证明桨叶学习稳定胜过运动基线；尚不能保证连续实时三维预测。下面同时保留覆盖率和失败结果。</p>
<section><h2>完整录像实测</h2><p>{len(runs)} 段 / {total_duration:.0f} 秒 / {total_events:,} 个原始事件。人工框仅用于独立评分。</p><table><tr><th>未来时长</th><th>有预测/标签</th><th>覆盖率</th><th>恒速 KF</th><th>完整自适应</th></tr>{metric_html()}</table><img src="prediction_metrics.png" alt="各模型预测误差"></section>
<section><h2>看预测落在实际未来画面的什么位置</h2><p>左：当前已发布预测。右：0.4 秒后的真实 RAW。蓝色为恒速预测，橙色为完整自适应预测；十字是过去已经发布的位置。未来画面只用于离线展示。</p><h3>清晰开发片段 train6</h3><video controls preload="metadata" src="visuals/train6/current_vs_future.mp4"></video><p><a href="visuals/train6/current_vs_future.gif">GIF 预览</a></p><h3>另一录像 train3</h3><video controls preload="metadata" src="visuals/train3/current_vs_future.mp4"></video></section>
<section><h2>覆盖与失败样例</h2><img src="coverage_and_error.png"><p>误差均值只计算有预测的时刻；低覆盖不能被低均值掩盖。train6 为开发录像。其余为本原型留出录像，不是跨机型验证。</p><img src="representative_cases.png"></section>
<section><h2>在线机制与下一步研究</h2><img src="online_diagnostics.png"><p>当前参数是有效图像运动关系，不是真实质量、风速或推力系数。优先研究桨叶测量可靠性、可辨识的尺度/推力/扰动模型，以及桨叶信息相对纯运动历史的真实提前量。</p><p><a href="实验结果与创新方向.md">完整方法、限制、结果和创新方向</a> · <a href="aggregate_metrics.csv">指标 CSV</a> · <a href="paired_bootstrap.json">探索性区块重采样</a> · <a href="experiment_protocol.json">冻结配置与指纹</a></p></section></main></html>'''
    (OUT/'查看项目结果.html').write_text(page,encoding='utf-8')
    page=page.replace('<h3>清晰开发片段 train6</h3>', '<h3>桨盘清晰近景：train6 109–114 秒</h3><video controls preload="metadata" src="visuals/train6_close/current_vs_future.mp4"></video><p><a href="visuals/train6_close/current_vs_future.gif">近景 GIF 预览</a></p><h3>按可见性选取的开发片段 train6</h3>')
    timing_path=OUT/'single_process_timing.json'
    if timing_path.exists():
        timing=json.loads(timing_path.read_text())
        timing_text=f'单进程高事件率片段，RAW 解码、感知及全部预测模型：中位 {timing["median_ms"]:.1f} ms，P95 {timing["p95_ms"]:.1f} ms；20 ms 内完成比例 {timing["fraction_within_20ms"]:.1%}，处理时间倍率 {timing["realtime_factor"]:.2f}×。不含视频写入和初始化，不能据此保证所有场景实时。'
        page=page.replace('<h2>在线机制与下一步研究</h2>',f'<h2>在线机制与下一步研究</h2><p>{timing_text}</p>')
        report+='\n## 单进程耗时复核\n\n'+timing_text+'\n\n完整记录见 `single_process_timing.json`。全片运行期间存在并发任务，其 wall 时间不能当作规范单路速度基准。\n'
        (OUT/'实验结果与创新方向.md').write_text(report,encoding='utf-8')
    (OUT/'查看项目结果.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(recordings=list(runs),sensor_seconds=total_duration,event_count=total_events,metrics=aggregate,bootstrap=bootstrap),ensure_ascii=False))

if __name__=='__main__':main()
