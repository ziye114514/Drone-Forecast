"""Version comparison on all sparse labels and on common available timestamps."""
from pathlib import Path
import argparse,csv,html,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
METHODS=('cv','ca','adaptive_history','adaptive_rotor','adaptive')


def csv_rows(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def statistics(rows):
    ok=[r for r in rows if r['available']=='True']
    errors=[float(r['error_px']) for r in ok]
    return dict(targets=len(rows),available=len(ok),coverage=len(ok)/max(len(rows),1),
                mean_error_px=float(np.mean(errors)) if errors else None,
                median_error_px=float(np.median(errors)) if errors else None,
                p90_error_px=float(np.percentile(errors,90)) if errors else None)


def mean_error(rows):
    return float(np.mean([float(r['error_px']) for r in rows])) if rows else None


def f(value,decimals=2):return f'{value:.{decimals}f}' if value is not None else '—'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'output/raw_only_project_integrated')
    args=parser.parse_args();out=args.output.resolve();baseline=ROOT/'output/raw_only_project'
    runs={};old_rows=[];new_rows=[];observations={};traces={}
    for sid in (6,2,3,4):
        name=f'train{sid}';folder=out/'runs'/name
        runs[name]=dict(summary=json.loads((folder/'summary.json').read_text()),
                        evaluation=json.loads((folder/'evaluation.json').read_text(encoding='utf-8')))
        observations[name]=[json.loads(s) for s in (folder/'observations.jsonl').read_text().splitlines()]
        traces[name]=[json.loads(s) for s in (folder/'forecasts.jsonl').read_text().splitlines()]
        if sid!=6:
            new_rows.extend(csv_rows(folder/'evaluation_details.csv'))
            old_rows.extend(csv_rows(baseline/'runs'/name/'evaluation_details.csv'))
    aggregate=[];comparison=[]
    for horizon in (.1,.2,.4):
        for method in METHODS:
            rows=[r for r in new_rows if r['method']==method and float(r['requested_horizon_s'])==horizon]
            aggregate.append(dict(horizon_s=horizon,method=method,**statistics(rows)))
        old=[r for r in old_rows if r['method']=='adaptive' and float(r['requested_horizon_s'])==horizon]
        new=[r for r in new_rows if r['method']=='adaptive' and float(r['requested_horizon_s'])==horizon]
        old_map={(r['sequence'],r['target_time_s']):r for r in old}
        new_map={(r['sequence'],r['target_time_s']):r for r in new}
        keys=sorted(set(old_map)&set(new_map))
        common=[k for k in keys if old_map[k]['available']=='True' and new_map[k]['available']=='True']
        gained=[k for k in keys if old_map[k]['available']!='True' and new_map[k]['available']=='True']
        lost=[k for k in keys if old_map[k]['available']=='True' and new_map[k]['available']!='True']
        comparison.append(dict(horizon_s=horizon,old=statistics(old),new=statistics(new),common_count=len(common),
                               common_old_mean_px=mean_error([old_map[k] for k in common]),
                               common_new_mean_px=mean_error([new_map[k] for k in common]),
                               gained_count=len(gained),gained_mean_px=mean_error([new_map[k] for k in gained]),
                               lost_count=len(lost),lost_old_mean_px=mean_error([old_map[k] for k in lost])))
    (out/'aggregate_metrics.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
    (out/'version_comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    with (out/'aggregate_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    horizons=[r['horizon_s'] for r in comparison]
    for version,label,color in [('old','v0.1','#2696be'),('new','v0.2','#ed8e32')]:
        axes[0].plot(horizons,[r[version]['coverage']*100 for r in comparison],'o-',label=label,color=color)
        axes[1].plot(horizons,[r[f'common_{version}_mean_px'] for r in comparison],'o-',label=label,color=color)
    axes[0].set(title='Forecast availability',xlabel='Horizon (s)',ylabel='Positive-label coverage (%)',ylim=(0,100))
    axes[1].set(title='Same timestamps, both available',xlabel='Horizon (s)',ylabel='Mean endpoint error (pixels)')
    for ax in axes[:2]:ax.legend()
    for i,method in enumerate(METHODS):
        vals=[r for r in aggregate if r['method']==method]
        axes[2].plot([r['horizon_s'] for r in vals],[r['mean_error_px'] for r in vals],'o-',label=method)
    axes[2].set(title='v0.2 shared detector / anchors',xlabel='Horizon (s)',ylabel='Mean endpoint error (pixels)');axes[2].legend(fontsize=7)
    fig.savefig(out/'version_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained',sharex=True)
    seq='train6';obs=observations[seq];ts=np.array([o['t'] for o in obs])
    for mode,color in [('ellipse','#23985f'),('arc','#b45bbf'),('line','#ba9b19')]:
        axes[0].plot(ts[::5],[o.get('observation_modes',{}).get(mode,0) for o in obs[::5]],label=mode,color=color,alpha=.8)
    axes[0].set(ylabel='Observed rotor count',title='Development recording: measured modes (inferred rotors excluded)');axes[0].legend(ncol=3)
    axes[1].plot(ts[::5],[o.get('geometry_quality',0) for o in obs[::5]],label='geometry quality',color='#9c49bb')
    axes[1].plot(ts[::5],[o.get('frequency_quality',0) for o in obs[::5]],label='frequency quality',color='#2596be',alpha=.7)
    gates=[bool(t.get('rotor_learning_allowed')) for t in traces[seq]]
    axes[1].fill_between(ts[::5],0,1,where=gates[::5],color='#40a46c',alpha=.15,label='rotor learner permitted')
    axes[1].set(xlabel='Sensor time (s)',ylabel='Diagnostic quality',ylim=(0,1.05));axes[1].legend(ncol=3)
    fig.savefig(out/'geometry_and_updates.png',dpi=170);plt.close(fig)
    # Stratified failures remain visible: select error quantiles for each recording.
    fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained');cases=[]
    for i,seq in enumerate(('train2','train3','train4')):
        eligible=sorted([r for r in new_rows if r['sequence']==seq and r['method']=='adaptive' and r['available']=='True'
                         and float(r['requested_horizon_s'])==.4],key=lambda r:float(r['error_px']))
        for j,quantile in enumerate((.1,.5,.9)):
            ax=axes[i,j]
            if not eligible:ax.set_axis_off();continue
            r=eligible[int(round(quantile*(len(eligible)-1)))];anchor=float(r['anchor_time_s'])
            k=int(np.argmin(abs(np.array([o['t'] for o in observations[seq]])-anchor)))
            trace=traces[seq][k];history=[o['center'] for o in observations[seq] if o['detected'] and anchor-.6<=o['t']<=anchor]
            if history:
                p=np.array(history);ax.plot(p[:,0],p[:,1],'.-',color='#8b939a',label='past measurements')
            for method,color in [('cv','#2596be'),('adaptive','#ed8e32')]:
                s=trace['states'][method];h=np.linspace(0,float(r['actual_horizon_s']),30)
                p=np.array(s['position'])+h[:,None]*s['velocity']+.5*h[:,None]**2*s['acceleration']
                ax.plot(p[:,0],p[:,1],color=color,label=method)
            ax.scatter(float(r['gt_x']),float(r['gt_y']),marker='*',s=100,c='#2eaf6c',label='future human box centre')
            ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim')
            ax.set_title(f"{seq} q{int(quantile*100)} | error {float(r['error_px']):.1f}px")
            if i==0 and j==0:ax.legend(fontsize=7)
            cases.append(r)
    fig.savefig(out/'representative_cases.png',dpi=150);plt.close(fig)
    (out/'representative_cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    timing=json.loads((out/'single_process_timing.json').read_text()) if (out/'single_process_timing.json').exists() else None
    total_seconds=sum(v['summary']['processed_sensor_seconds'] for v in runs.values())
    total_events=sum(v['summary']['event_count'] for v in runs.values())
    sections=[
        '# 二维 RAW-only 无人机预测 v0.2：整合与回归结果',
        '\n本轮优先完善工程闭环，没有将整合本身作为论文创新。预测器仅输入 RAW；现有人工框只进入独立评分。',
        '\n新增完整椭圆/残弧/近线段观测、局部几何不确定性、法向正负不变特征、频谱谐波歧义与半窗相干诊断、低质量冻结学习与历史回退、短时单桨关联。',
        '\n近线段只提供轴方向，残弧补全不等于独立观测到完整形状；全部为未标定图像线索。当前不会把倾斜直接当成飞行速度。',
        f'\n本轮运行 {total_seconds:.0f} 秒，处理 {total_events:,} 个事件。train6 用于开发；另外三段此前也已检查/评分，本轮称固定配置工程回归，不声称全新留出测试。',
        '\n## 新旧版本：各自覆盖与相同样本',
        '\n误差仅对有预测样本统计。为了防止漏检差异影响比较，另外报告两个版本都有预测的相同标签时刻。原生图像 1280×720；稀疏标签只能评端点误差，不是密集 ADE。',
        '\n| 时长 | 旧版覆盖 | 新版覆盖 | 相同样本数 | 旧版相同样本误差/px | 新版相同样本误差/px | 新增/丢失预测 |',
        '|---|---|---|---|---|---|---|']
    for r in comparison:
        sections.append(f"| {r['horizon_s']:.1f}s | {r['old']['coverage']:.1%} | {r['new']['coverage']:.1%} | {r['common_count']} | {f(r['common_old_mean_px'])} | {f(r['common_new_mean_px'])} | {r['gained_count']}/{r['lost_count']} |")
    r=next(row for row in comparison if row['horizon_s']==.2)
    change=100*(r['common_new_mean_px']/r['common_old_mean_px']-1)
    sections += [f"\n**本轮结论：功能覆盖更完整，预测精度没有同步提高。** 0.2 秒覆盖率由 {r['old']['coverage']:.1%} 增至 {r['new']['coverage']:.1%}；双方都有预测的 {r['common_count']} 个相同样本，平均误差由 {r['common_old_mean_px']:.2f} 增至 {r['common_new_mean_px']:.2f} 像素，增加 {change:.1f}%。不能把整合版称为精度升级，旧版予以保留。",
                 '\n新几何前端、中心不确定性加权、频率筛选和学习特征均发生了变化。本轮只能判断整合后的总体效果；尚不能把精度变化唯一归因于某个模块。定位偏差、滤波滞后和在线学习需进一步逐项检查，不能用补全曲线看起来平滑代替验证。']
    sections+=['\n![新旧对照](version_comparison.png)','\n## 新版各分支，使用相同观测和预测时刻',
               '\n| 时长 | CV | CA | 纯历史在线 | 桨叶在线/质量回退 | 完整组合 | 覆盖率 |','|---|---|---|---|---|---|---|']
    for horizon in horizons:
        rows=[next(r for r in aggregate if r['method']==m and r['horizon_s']==horizon) for m in METHODS]
        sections.append(f"| {horizon:.1f}s | "+' | '.join(f(r['mean_error_px']) for r in rows)+f" | {rows[0]['coverage']:.1%} |")
    sections+=['\n全部逐时刻误差、P90、分录像成绩、新增和丢失样本统计见 JSON/CSV；不依据成绩修改已冻结配置。',
               '\n## 检测、定位与模式使用','\n| 录像 | 发现目标比例 | IoU≥0.3召回 | 阴性误报率 | 椭圆/残弧/线段次数 |','|---|---|---|---|---|']
    for seq,v in runs.items():
        d=v['evaluation']['detection'];counts=v['summary']['rotor_geometry_counts']
        sections.append(f"| {seq} | {d['detector_presence_recall']:.1%} | {d['recall_iou_0_3']:.1%} | {d['false_positive_rate']:.1%} | {counts['ellipse']}/{counts['arc']}/{counts['line']} |")
    sections+=['\n模式次数是自动分类结果，没有逐桨几何真值，不能当作分类准确率。',
               '\n![几何和更新](geometry_and_updates.png)',
               '\n## 可控几何检查','\n![合成几何检查](visuals/geometry_synthetic.png)',
               '\n上图为明确标记的合成几何实验，不是飞行实测成绩。过短弧段可能只能提供局部线方向，不能确认它就是整个桨盘平面的方向。',
               '\n## 成功、中等与失败样例','\n![分位数样例](representative_cases.png)',
               '\n## 观看实际 RAW 对照',
               '\n蓝色为旧版完整预测，橙色为新版完整预测；绿色为椭圆，粉色为残弧补全，黄色为轴方向。右侧为 0.4 秒后实际 RAW，仅用于离线展示。',
               '\n- [近景固定片段](visuals/train6_close/current_vs_future.mp4)',
               '- [另一录像固定片段](visuals/train3/current_vs_future.mp4)',
               '- [开发录像可见性选片](visuals/train6/current_vs_future.mp4)',
               '\n近景与另一录像使用上一版相同时间段；自动选片只看可见性，不按未来误差挑选。',
               '\n## 使用范围与工程限制',
               '\n- 单主要目标、SDK 支持的 RAW、二维像素预测；目标出画面/长时全遮挡后会停止发布。',
               '- 频率仍是叶片通过代理，不是已用测速真值验证的机械 RPM。相干诊断不能保证找到正确基频。',
               '- 全部补全、轴向与中心协方差均为近似；预测半径未校准，不能宣称具有 95% 覆盖。',
               '- 不知道未来飞控动作；长时预测和突然转向仍可能失败。',
               '- 只有稀疏人框，没有独立姿态/RPM/连续位置真值；历史单目标关联也不能保证不误跟。']
    if timing:
        sections+=['\n## 单进程速度检查',
                   f"\n与上一版相同高事件率区间，单进程每 20ms 窗口处理耗时中位 {timing['median_ms']:.1f}ms，P95 {timing['p95_ms']:.1f}ms；{timing['measured_sensor_s']:.1f}s 数据处理用时 {timing['wall_processing_s']:.2f}s。不含初始化、预热或视频写入。不能保证实时。"]
    sections+=['\n## 复核文件',
               '\n16 项测试已通过；真实 RAW 的前 4 秒单独处理后，与完整录像的前 4 秒逐项比较。具体结果保存在测试记录、因果性记录和最终验证文件中。',
               '\n`experiment_protocol.json`、`frozen_config.json`、各 run_manifest、`validation.json`、`causality_validation.json` 和测试日志用于复核；旧版结果位于相邻 raw_only_project，旧源码备份在 baseline_v010。']
    markdown='\n'.join(sections)+'\n';(out/'整合结果报告.md').write_text(markdown,encoding='utf-8')
    # Minimal dependency-free document conversion for this controlled report.
    chunks=[];table=[];in_list=False
    def flush_table():
        if table:
            chunks.append('<table>'+''.join('<tr>'+''.join('<'+('th' if i==0 else 'td')+'>'+html.escape(c.strip())+'</'+('th' if i==0 else 'td')+'>' for c in row.strip('|').split('|'))+'</tr>' for i,row in enumerate(table) if not set(row.replace('|','').strip())<={'-',' '})+'</table>');table.clear()
    for line in markdown.splitlines():
        if line.startswith('|'):table.append(line);continue
        flush_table()
        if line.startswith('!['):
            path=line.split('](')[1][:-1];chunks.append(f'<img src="{html.escape(path)}" loading="lazy">');continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));chunks.append(f'<h{level}>{html.escape(line[level:].strip())}</h{level}>');continue
        if line.startswith('- ['):
            label=line.split('](')[0][3:];path=line.split('](')[1][:-1]
            chunks.append(f'<h3>{html.escape(label)}</h3><video src="{html.escape(path)}" controls preload="metadata"></video>');continue
        if line.strip():chunks.append('<p>'+html.escape(line)+'</p>')
    flush_table()
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>事件无人机二维预测 v0.2</title>
<style>body{font:16px/1.75 "Microsoft YaHei",sans-serif;color:#dbe4ee;background:#111923;margin:0}main{max-width:1180px;margin:auto;padding:32px}h1,h2{color:#fff}h2{margin-top:42px;border-bottom:1px solid #344050;padding-bottom:8px}table{width:100%;border-collapse:collapse;background:#1a2634;font-size:14px}th,td{padding:11px;border:1px solid #364456;text-align:left}th{color:#ffbd75}img,video{display:block;width:100%;border-radius:10px;margin:18px 0}p{color:#c1cedd}a{color:#75c8ff}.banner{padding:16px;background:#2a3143;border-left:4px solid #ed8e32}</style><main>
<p class="banner">RAW-only · 完整椭圆 / 残弧 / 轴方向 · 质量门控在线学习 · 二维研究原型 · 真实回归与失败案例</p>'''+''.join(chunks)+'''<p><a href="../../event_drone_forecast/README.md">运行说明</a> · <a href="整合结果报告.md">文字报告</a> · <a href="version_comparison.json">新旧版逐项比较</a> · <a href="validation.json">最终验证记录</a></p></main></html>'''
    (out/'查看整合结果.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(comparison=comparison,frames=sum(len(v) for v in traces.values()),events=total_events),ensure_ascii=False))


if __name__=='__main__':main()
