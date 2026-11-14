"""Summarize root causes, retained fixes, failures and actual output timing."""
from pathlib import Path
import sys,json,html
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'tools')]
from audit_target25 import continuity
OUT=ROOT.parent/'output/stability_audit_20261031'


def load(p):return json.loads(p.read_text('utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text('utf-8').splitlines()]


def main():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':150})
    scene=load(OUT/'scene_audit.json');names=list(scene)
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.1),layout='constrained')
    for axis,key,title in zip(axes,['error_px','error_box_diagonals','true_displacement_04_px'],
        ['0.4秒平均绝对误差 / px','误差 / 目标框对角线','真实0.4秒位移中位 / px']):
        vals=[scene[n]['statistics'][key]['mean' if key!='true_displacement_04_px' else 'median'] for n in names]
        axis.bar(names,vals,color=['#d76944' if n=='train2' else '#258394' for n in names]);axis.set_title(title,fontsize=11);axis.tick_params(axis='x',rotation=40);axis.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'scene_comparison.png');plt.close(fig)
    pairs={name:load(OUT/(name+'_scores.json')) for name in ['FRED7_final','FRED9_budget','FRED0_recovery']}
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for axis,name,title in zip(axes,['FRED7_final','FRED9_budget'],['FRED7：开发片段','FRED9：本轮配置预登记留出']):
        result=pairs[name];target=result['control']['targets'];other={r['slot']:r for r in result['candidate']['targets']}
        for key,label,color in [('control','原30Hz异步配置','#888f99'),('candidate','预算恢复工作配置','#117e97')]:
            rr=result[key]['targets'];xx=[r['t'] for r in rr];yy=[r['error_px'] if r['available'] else np.nan for r in rr]
            axis.plot(xx,yy,'.-',label=label,color=color)
            missing=[r['t'] for r in rr if not r['available']];axis.scatter(missing,[-10]*len(missing),marker='x',color=color,s=35)
        axis.axhline(25,color='#b4513c',ls='--',lw=1,label='25px目标');axis.set(title=title,xlabel='源时间 / s',ylabel='0.4s误差 / px');axis.legend(fontsize=8);axis.grid(alpha=.2)
    fig.savefig(OUT/'forecast_comparison.png');plt.close(fig)
    runtime={};table=[]
    for p in sorted(OUT.glob('RAW_*')):
        if not (p/'summary.json').exists():continue
        s=load(p/'summary.json');rr=rows(p/'predictions.jsonl');pub=[]
        if (p/'publications.jsonl').exists():pub=rows(p/'publications.jsonl')
        valid=[r['t'] for r in pub if r['valid']]
        intervals=np.diff(valid)*1000
        runtime[p.name]=dict(summary=s,continuity=continuity(rr),valid_publication_interval_p95_ms=float(np.percentile(intervals,95)) if len(intervals) else None,
            longest_valid_publication_gap_ms=float(np.max(intervals)) if len(intervals) else None)
        table.append(f'<tr><td>{html.escape(p.name)}</td><td>{s["fresh_image_observations"]}/{s["frames"]}</td><td>{s["timing_ms"]["completion_lag_ms"]["p95"]:.1f}</td><td>{s.get("publication",{}).get("valid_outputs",0)}/{len(pub)}</td><td>{s.get("publication",{}).get("actual_mean_hz",0):.2f}</td></tr>')
    (OUT/'runtime_inventory.json').write_text(json.dumps(runtime,indent=2,allow_nan=False),'utf-8')
    fig,axes=plt.subplots(2,1,figsize=(12,6.5),layout='constrained')
    for ax,paths,title in [(axes[0],['RAW_FRED9_control4s','RAW_FRED9_budget4s'],'FRED9：增加当前检测后仍在源周期内完成'),
                          (axes[1],['RAW_side_recovery8s','RAW_side_budget8s'],'侧视：预算减少阻塞，冷启动和重计算长尾仍存在')]:
        for path in paths:
            rr=rows(OUT/path/'predictions.jsonl');tt=np.array([r['t']-rr[0]['t'] for r in rr]);lag=[r['cost']['completion_lag_ms'] for r in rr]
            ax.plot(tt,lag,label=path.replace('RAW_',''),lw=1)
        ax.axhline(1000/30,color='#aa4a3b',ls='--',label='33.33ms');ax.set(title=title,xlabel='从首个源时刻经过 / s',ylabel='源截止滞后 / ms');ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.savefig(OUT/'latency_stability.png');plt.close(fig)
    cells=[]
    for name,label in [('FRED7_final','FRED7'),('FRED9_budget','FRED9'),('FRED0_recovery','FRED0')]:
        s=pairs[name];m=s['matched'];a=m['old_matched_error'];b=m['new_matched_error'];aa=s['control'];bb=s['candidate']
        cells.append(f'<tr><td>{label}</td><td>{a["mean"]:.2f} → {b["mean"]:.2f}</td><td>{a["p95"]:.2f} → {b["p95"]:.2f}</td><td>{m["new_available"]}/{m["original_origins"]}</td><td>{aa["predicted_target_slots"]}/{aa["target_slots"]} → {bb["predicted_target_slots"]}/{bb["target_slots"]}</td></tr>')
    walls=[]
    for name in ('FRED7','FRED9'):
        s=load(OUT/(name+'_common_publication.json'));m=s['matched'];a=s['statistics']['control'];b=s['statistics']['candidate']
        walls.append(f'<tr><td>{name}</td><td>{m["control_error_px"]["mean"]:.2f} → {m["candidate_error_px"]["mean"]:.2f}</td><td>{m["n"]}</td><td>{a["valid"]}/{a["query_slots"]} → {b["valid"]}/{b["query_slots"]}</td></tr>')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>跨片段误差调查与30Hz连续性修复</title>
<style>body{{max-width:1140px;margin:36px auto;padding:0 22px;background:#f2f5f8;color:#203248;font:16px/1.8 system-ui}}section{{background:white;padding:22px;margin:20px 0;border-radius:9px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #ccd6df;padding:7px;text-align:left}}img,video{{max-width:100%}}.note{{padding:16px;background:#e2f1f4;border-left:5px solid #218596}}pre{{white-space:pre-wrap;background:#e9eef3;padding:16px}}a{{color:#12748d}}</style>
<h1>为什么28px与60px并存，以及这次修了什么</h1><p>2026-10-31 · 二维0.4秒预测 · 非侧视25px／持续有效30Hz仍未全面达到</p>
<p class="note">已修复一处真实配置退化：失锁时只有异步旧框、又无法做当前光流转移，造成小目标反复断跟。新30Hz配置在实际截止预算允许时复用已有当前NN恢复，同时融合原稳健拟合与椭圆残差计算。FRED9同14个源起点59.33→56.72px，覆盖14/19→17/19。没有把不同起点的整体均值当同输入提升。</p>
<section><h2>不能把28px当作各段都接近达标</h2><p>FRED0目标对角线中位44.82px，train2约183.43px，绝对误差28.29与60.16对应的平均框尺度误差为0.632与0.322。小目标28px相对目标本身仍很大。按尺度诊断不替代绝对25px目标。</p><p>train2把当前估计位置替换成人工当前位置后，未来误差仍55.61px；评分起点没有合格周期力输入。速度／加速度估计和未来延续仍是主要卡点。原生时间戳保留微秒分辨率，没有发现这次差异来自压成低帧率视频。不能把当前误差全部解释为“未来突然变向不可避免”。</p><img src="scene_comparison.png"></section>
<section><h2>过拟合风险：有，之前的结论需要收紧</h2><p>现有片段反复参与开发调参，不能称严格独立测试。上游<a href="https://github.com/takara-ai/DroneStalker#drone-detection">DroneStalker说明</a>区分只用FRED0的初版和整个FRED的后续模型；本地权重的精确上游成员尚待逐权重核实，不能把新增FRED片段说成全系统从未见过。FRED8／9只代表本轮参数先冻结、后评分的过程留出。检测模型卡已补当前ed801等权合并权重与旧e253入口区别。未进行新训练。</p><p>28与60的差异不单独证明或排除过拟合。这里同时存在目标尺度、运动难度、周期可观测性、工作配置及延迟差异。</p></section>
<section><h2>已接入的优化与同起点结果</h2><p>优先复用项目已有功能：当前KLT失败时，如当下源时隙余量足以覆盖近期NN服务和下游工作，就检测当前图像；忙碌或已经迟到时保留异步路径。候选框来自当前图像，过去状态只用于搜索。该预算是估计，不能当硬实时证明。</p><table><tr><th>序列</th><th>同原起点均值px</th><th>P95 px</th><th>原起点保留</th><th>各自预测覆盖</th></tr>{''.join(cells)}</table><p>另段train2同15起点60.159→60.032，基本仍在60px；侧视预算版同26起点32.519→32.847，P95 59.214→58.501，是小幅取舍。FRED0新增一个较难起点后，各自13起点均值31.48；不能只报告保留12起点的28.15掩盖新增难例。FRED7增加覆盖后各自20起点81.82px，仍然很差。</p><img src="forecast_comparison.png"></section>
<section><h2>按真正发布之后0.4秒再算一次</h2><p>固定相同5Hz墙钟查询与相同未来目标，只允许使用查询前已经发布完成的状态。源时刻正确但尚未计算完成的结果不能拿来评分；将同一力学模型积分到查询+0.4，不挪时间戳。</p><table><tr><th>序列</th><th>共同可用查询均值px</th><th>共同查询数</th><th>各自可用查询</th></tr>{''.join(walls)}</table><p>FRED7各自查询均值104.95→105.75，新增覆盖包含难例；共同10查询才是104.95→97.24。实际时间口径仍明显差于源时刻口径，不能混用来宣传精度。</p></section>
<section><h2>帧率：输出约30Hz，当前图像与误差仍未稳定</h2><p>编译稳健拟合保持白化Huber目标，不能认证梯度就回原SciPy；七段冻结输入资格相同，最大预测数值差8.43e-7px。融合椭圆残差不删点、不删假设：60个相同真实RAW时刻完整几何一致。清晰4秒端到端仍有>33ms长尾；局部函数变快不等于全链路30Hz。</p><img src="latency_stability.png"><p>train2一次源44.166225的解码耗时3010.79ms、解码进程CPU2890.63ms，当帧预测计算18.77ms。后续带GC记录的运行没有复现3秒停顿；记录到的108ms二代GC出现在初始解码范围内，不能据此宣称3秒根因已找到。滚动读取＋有限流GC延后没有证明普遍更快，保持可选。侧视冷启动和几何／周期重计算也有长尾。</p><details><summary>全部本轮短RAW：含失败设置与慢运行</summary><table><tr><th>运行</th><th>当前图像观测</th><th>源滞后P95 ms</th><th>有效发布/全部发布</th><th>发布Hz</th></tr>{''.join(table)}</table></details><p>旧状态推进和无效心跳均不算新检测，所有漏失保留；详见runtime_inventory.json中的有效发布最大间隔与连续及时观测段。</p></section>
<section><h2>看看实际预测</h2><video controls preload="metadata" poster="FRED9_comparison.png" src="FRED9_comparison.mp4"></video><p>同一源事件、左右旧/新配置。绿色为当前检测框，黄色为源时刻+0.4秒预测，白十字是事后未来标注，红线是误差。显示裁剪用标注帮助看清小目标，标注没有进入推理。影片30FPS仅为播放速率，不代表全部时刻都有有效检测。</p></section>
<section><h2>继续累计：约0.33ms的小收益已经采用</h2><p>复用LSD对象，并精确查表计算其原本使用的log(原生计数)图。没有引入G通道、灰度重建、采样或阈值修改。三个真实计数场景线检测输出完全相同，60个同输入完整几何完全相同；组件中位10.051→9.720ms、P95 16.404→14.897ms。最终工作区完整清晰RAW与之前源预测逐项相同。单独运行的清晰4秒处理34.628FPS，但滞后P95仍143.712ms、120发布仅97有效，不能当稳定30Hz。早一次final测速与报告绘图可能重叠，不作纯速度归因，慢记录保留。</p><p><a href="body_workspace/summary.json">组件对照</a> · <a href="body_geometry_checks/summary.json">60时刻完整几何</a></p></section>
<section><h2>净力补充的小精度收益保留为候选</h2><p>新增可选position-net-force：周期、保持力及合格光流都缺失时，由过去机体位置估计当前图像净加速度，继续同一个力学模型，原p/v不变。固定观测FRED7 81.818→75.895、FRED9 54.180→53.756、train2 60.159→60.661。预登记FRED10固定同18输入66.850→65.475、P95 147.127→134.929。</p><p>但完整RAW中，原18起点仅17共同可用，65.450→66.527；相同实际查询共同12为76.197→80.148，未保住收益。新计算会影响按墙钟预算决定的检测刷新，使输入序列改变；还需隔离调度影响。候选代码和小收益没有丢弃，也没有按FRED10误差重调权重后默认上线。</p><p><a href="position_net_force/FRED10_fixed_scores.json">固定输入收益</a> · <a href="FRED10_netforce_scores.json">真实RAW</a> · <a href="FRED10_common_publication.json">相同实际查询</a></p></section>
<section><h2>试过而未默认的其他路径</h2><p>机体中心采点60.159→60.678；成熟自身误差校力→64.545；逐轴恢复加速度→64.093；拟合速度端点→62.359。低分框续接在FRED0恢复7次但当前观测数不增、同12起点28.288→29.197。扩大光流搜索通过合成30px移动／空图拒绝，但真实FRED7没有得到大位移恢复；均保留试验记录，未按某段标签调权重后强行合并。这里不是因为收益小才不合并，而是尚未证明方向正确。</p></section>
<section><h2>运行与下一步</h2><pre>python run_fast_structure.py --raw "events.raw" --output "新的输出目录" --start 32 --duration 4 --sdk-avx2 --structure-guided-nn --realtime30
旧30Hz控制：将 --realtime30 替换为 --legacy-realtime30</pre><p>原五圆／可见模式、独立侧轴、周期相对平方输入、合并未知力与当前净加速度逻辑保留，没有回到另一条传统预测器。下一步优先：让机动期的真实周期变化进入模型、按控制变化拆解外推误差；对解码长停顿加调用级追踪并压冷启动长尾。保持跨片段冻结配置，继续记录P95、最差情况与可用率。真实RPM、牛顿推力和完整3D姿态仍未测得。</p><p><a href="scene_audit.json">跨片段审计</a> · <a href="FRED9_budget_scores.json">FRED9源起点</a> · <a href="FRED9_common_publication.json">FRED9相同实际查询</a> · <a href="runtime_inventory.json">全部时延与漏失</a> · <a href="holdout9/preregister.json">FRED9事前协议</a> · <a href="fast_huber/summary.json">稳健拟合</a> · <a href="conic_residual_checks_v2/summary.json">几何一致性</a> · <a href="source_snapshot/manifest.json">最终源码</a></p><p>复用<a href="https://docs.opencv.org/4.x/dc/d6b/group__video__track.html">OpenCV LK</a>和现有SDK；低分续接参考<a href="https://github.com/FoundationVision/ByteTrack">ByteTrack</a>的已知思路，未称首创或作者复现。新功能未发布到GitHub，原权重及发布ZIP保持冻结。</p></section></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    print('report and 3 plots saved')


if __name__=='__main__':main()
