"""One report for accumulated structural/physical improvements and their limits."""
from pathlib import Path
import sys,json,html
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'output/three_continuation_20261026'


def read(path):return json.loads((OUT/path).read_text('utf-8'))
def observations(name):return list(map(json.loads,(OUT/name/'predictions.jsonl').read_text('utf-8').splitlines()))
def fmt(x):return f'{x:.3f}'


def main():
    runs={name:read(f'raw_release_{name}/summary.json') for name in ['clear','side']}
    rows={name:observations(f'raw_release_{name}') for name in runs}
    rates={}
    for name,data in rows.items():
        protocol=read(f'raw_release_{name}/protocol.json')
        elapsed=data[-1]['t']-protocol['start']+data[-1]['cost']['completion_lag_ms']/1000
        rates[name]=dict(source_updates_hz=len(data)/elapsed,
            successful_body_observations_hz=runs[name]['paced_fresh_observation_rate'],
            successful_body_fraction=runs[name]['fresh_image_observations']/len(data),
            count_scope='Success means current NN/KLT body observation, not labelled detection recall.')
    pair=read('paired_normal/summary.json');force=read('gain_quality/summary.json')
    accuracy=read('raw_release_side/accuracy.json');coupled=read('coupled_observer/summary.json')
    speed={name:read(f'{name}/summary.json') for name in ['joint_counts','shared_roi_counts','batched_uncertainty','shared_frequency_pixels']}
    values=accuracy['scores']['0.4'];names=['physical_8row_control','physical','cv','kinematic']
    fig,ax=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    ax[0].bar(['old physical','accumulated','constant velocity','offline kinematic'],[values[n]['mean'] for n in names],color=['#8c9bad','#e99829','#729ac1','#406f9e'])
    ax[0].set(ylabel='Mean 0.4 s error (px)',title='Same 13 scored origins, current side RAW')
    ax[0].tick_params(axis='x',rotation=16)
    for name,label,color in [('physical','accumulated','#c57413'),('physical_8row_control','old physical','#8493a3'),('kinematic','offline kinematic','#315f8d')]:
        scored=[s for s in accuracy['samples'] if s['horizon']==.4]
        ax[1].plot([s['t'] for s in scored],[s['errors'][name] for s in scored],'.-',label=label,color=color)
    ax[1].set(xlabel='Causal forecast origin (s)',ylabel='0.4 s error (px)',title='Sparse references, no turn-onset labels')
    ax[1].legend(fontsize=8);fig.savefig(OUT/'forecast_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3),layout='constrained')
    for name,label,color in [('contrast_area','without normal constraints','#8294a6'),('normal_constraints','with normal constraints','#bd7621')]:
        modes=pair['arms'][name]['modes'];x=np.arange(3)+(0 if name=='contrast_area' else .36)
        ax.bar(x,[modes.get(n,0) for n in ['four','three','partial_two']],.36,label=label,color=color)
    ax.set_xticks(np.arange(3)+.18,['four','three','partial two']);ax.set(ylabel='Current source ticks',title='Same events + same current body boxes, 60 ticks');ax.legend()
    fig.savefig(OUT/'structure_comparison.png',dpi=150);plt.close(fig)
    # A saved native frame with the actual three-centre output, not a montage
    # drawn from predicted or otherwise missing rotor pixels.
    i=next(i for i,r in enumerate(rows['clear']) if r['structure']['mode']=='three')
    cap=cv2.VideoCapture(str(OUT/'raw_release_clear/results.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read();cap.release();assert ok
    r=rows['clear'][i];b=np.array(r['body']['bbox']);lo=np.maximum(0,np.floor(b[:2]-[35,35]).astype(int));hi=np.minimum([1280,720],np.ceil(b[2:]+[35,35]).astype(int))
    cv2.imwrite(str(OUT/'three_current_frame.jpg'),frame[lo[1]:hi[1],lo[0]:hi[0]])
    table=''.join(f'<tr><td>{name}</td><td>{s["frames"]}</td><td>{s["fresh_image_observations"]}</td><td>{s["valid_forecasts"]}</td><td>{s["steady_capacity_fps"]:.2f}</td><td>{rates[name]["source_updates_hz"]:.2f}</td><td>{rates[name]["successful_body_observations_hz"]:.2f}</td><td>{s["timing_ms"]["total_service_ms"]["p95"]:.2f}</td></tr>' for name,s in runs.items())
    summary=dict(runs=runs,rate_definitions=rates,same_input_structure=pair,
        same_input_force=force,current_side_accuracy=accuracy,
        component_speed=speed,coupled_observer_not_promoted=coupled,
        stable30_proven=False,physics_advantage_proven=False,traditional_forecast_in_runtime=False,
        original_v7_unchanged=True,labels_only_after_predictions=True)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>三桨续接、二力反馈与计算加速</title>
<style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1100px;margin:36px auto;padding:0 22px;color:#1d2937;background:#f3f6f8;line-height:1.75}}section{{background:white;border:1px solid #dce3e9;border-radius:10px;padding:20px 26px;margin:18px 0}}h1{{font-size:29px}}h2{{font-size:21px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #dce3e9;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%;height:auto}}video{{width:100%;background:#222}}.note{{color:#735012;background:#fff3d6;padding:12px;border-radius:6px}}code{{background:#eef2f5;padding:1px 4px}}a{{color:#1b628e}}small{{color:#596c7b}}</style>
<h1>五圆模型：三桨续接、二力反馈与计算加速</h1><p>2026-10-26晚 · 累计保留小改进 · 输入为自采 train6 RAW · 此页为本轮完成结果。</p>
<section><h2>这次改了什么</h2><ol><li>三桨中有足够独立当前中心时，续接原五圆模型；两桨、轴线和模型自身推测不续期。</li><li>短弧只提供轮廓法线方向约束，不强行拟合出一个完整桨盘。当前弧与面积支持合格后，才允许进一步恢复真实桨心。</li><li>未知合力d不再等增益k的8行门槛：从首个成熟0.2秒位移开始慢修正；学习端点与位置状态使用一致坐标。缺测时最多保持120ms总加速度。</li><li>联合ON/OFF计数、批量椭圆不确定性、共享扩大ROI和每像素周期记录，减少重复计算。</li></ol>
<p class="note">模型是二维图像模型；相对闪烁频率平方仍不是已验证的真实RPM/牛顿推力。当前也没有恢复传统实时加速度预测分支。</p></section>
<section><h2>结构：有进步，但还没持续锁住三桨</h2><p>相同事件、相同当前整机框的60时刻对照，短弧法线约束更新7次，three从0次到2次，恢复当前弧2→4条。两次three相隔100ms，中间有两桨信息帧；不是连续30Hz三桨检测。完整RAW中也有2次three。</p>
<img src="structure_comparison.png" alt="同输入结构计数"><p>下面是当前保存输出的一帧：红点为当前提取中心，灰色部分是模型推断；隐藏点不算新的观测。</p><img src="three_current_frame.jpg" alt="实际三桨帧局部图"></section>
<section><h2>预测：同起点小收益保留，物理分支仍未胜出</h2><p>当前侧视完整RAW的13个相同0.4秒目标：旧8行物理规则均值 <b>{values['physical_8row_control']['mean']:.3f}→{values['physical']['mean']:.3f}px</b>，中位 {values['physical_8row_control']['median']:.3f}→{values['physical']['median']:.3f}px；P95 {values['physical_8row_control']['p95']:.3f}→{values['physical']['p95']:.3f}px变差。离线传统同起点对照 {values['kinematic']['mean']:.3f}px仍然更好。</p>
<img src="forecast_comparison.png" alt="当前同起点预测误差"><p>另一个历史14目标固定输入试验：57.492→提前d55.948→加短保持56.011px。组合中位61.834→55.675，但尾部84.944→86.756px变差。两组起点不同，不能把56.011→52.718写成算法增益。</p>
<p>参考为已用于开发的稀疏人工框中心，插值仅跨≤150ms；不是质心/姿态/转速真值，也没有独立变向起点标签。没有清晰26–28秒的匹配独立中心真值。</p></section>
<section><h2>速度：平均能力到了30附近，低延迟稳定性还没到</h2><table><tr><th>片段</th><th>源帧</th><th>成功整机观测</th><th>有效预测</th><th>计算能力FPS</th><th>源更新Hz</th><th>成功整机观测Hz</th><th>处理P95 ms</th></tr>{table}</table>
<p>计算能力=1000/平均服务毫秒；源更新率计全部处理的当前源时刻；成功整机观测率只计当前NN/KLT支持的整机框，并非人工真值检出率。三者含义不同。视频30FPS是播放速度。</p>
<p>清晰段完成延迟P95 {runs['clear']['timing_ms']['completion_lag_ms']['p95']:.1f}ms，侧视 {runs['side']['timing_ms']['completion_lag_ms']['p95']:.1f}ms。初次seek解码、启动和EOF等待另有记录；短片平均超过30不等于持续低延迟30FPS。</p>
<table><tr><th>组件配对</th><th>旧中位ms</th><th>新中位ms</th><th>等价证据</th></tr><tr><td>联合ON/OFF计数</td><td>3.033</td><td>2.104</td><td>57 ROI×3，逐像素精确</td></tr><tr><td>椭圆不确定性批算</td><td>0.425</td><td>0.270</td><td>298 conic×3，最大差1.78e-15</td></tr><tr><td>复用已存在的扩大ROI</td><td>2.314</td><td>2.207</td><td>几何计数精确相同</td></tr><tr><td>周期摘要共享最新像素</td><td>1.696</td><td>1.099</td><td>60包，逐桨/整机摘要精确</td></tr></table><small>各项来自独立组件配对，不能直接相加当作整条流水线测速。</small></section>
<section><h2>实际结果视频</h2><p>只把保存的因果输出画回原生ON/OFF画面，没有重跑检测或读取未来框作推理。网页使用VP8/WebM预览，原MP4保留。</p><p>清晰到侧向退化（2秒）</p><video controls preload="metadata" poster="raw_release_clear/poster.jpg" src="raw_release_clear/preview.webm"></video><p>侧视（3秒）</p><video controls preload="metadata" poster="raw_release_side/poster.jpg" src="raw_release_side/preview.webm"></video></section>
<section><h2>没有硬塞进主线的东西</h2><p>二倍sigma代理的增益资格在本小段没有改变预测，保留<code>--strict-gain-quality</code>可选。协方差来自重叠行，是启发式代理，不称95%置信区间。</p><p>复用已有联合位置/未知力观察器后，共同12目标0.4秒均值54.470→56.268，P9590.833→146.525；保持实验状态。无事件输入的观察器对照也保留，避免把普通位置反馈收益说成桨叶物理创新。</p></section>
<section><h2>当前卡点与下一步</h2><ol><li>三桨续接只维持两次更新：变薄后当前有效弧和形状支持不足。下一步需要更可靠的共享几何/周期联合跟踪，不用虚拟桨心凑数。</li><li>周期平方与像素加速度关联仍弱：未知有效增益、谐波歧义、投影及定位噪声混在一起。改善d启动不等于测出了推力。</li><li>平均计算可过30，耗时尖峰和初期积压仍影响实际延迟。下一步优先约束昂贵轮廓分解的关键路径，并保留当前观测时间。</li><li>真值不足：目前只能证明这个已暴露片段的小收益，不能证明泛化、变向优势或论文首创。</li></ol></section>
<section><h2>机制来源、复现与记录</h2><p>共享几何与0.2s慢形状更新参照 <a href="https://arxiv.org/abs/2610.31384">CoHelix</a>。短弧法线约束参考 <a href="https://pubmed.ncbi.nlm.nih.gov/16805268/">ViSP/VVS论文</a>与<a href="https://github.com/lagadic/visp">官方项目</a>；本地为自写NumPy实现，未复制或打包ViSP代码，不宣称该通用跟踪原理首创。</p>
<p><a href="summary.json">完整结果JSON</a> · <a href="paired_normal/summary.json">同输入结构对照</a> · <a href="gain_quality/summary.json">力学累计对照</a> · <a href="coupled_observer/summary.json">联合观察器负结果</a> · <a href="raw_release_clear/protocol.json">清晰RAW协议/源哈希</a> · <a href="raw_release_side/protocol.json">侧视RAW协议/源哈希</a> · <a href="../../event_drone_forecast/docs/THREE_CONTINUATION.md">代码与控制开关说明</a></p>
<p>原位置v7与旧源快照保留；未重新打包历史wheel/ZIP，未发布远端仓库。当前报告不改变原始预测日志或历史评分。</p></section></html>'''
    (OUT/'index.html').write_text(report,'utf-8')
    print(json.dumps(dict(report=str(OUT/'index.html'),rates=rates,stable30=False),ensure_ascii=False))


if __name__=='__main__':main()
