"""Summarize completed pilots and render measured trajectories; no inference."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'tools')]
from compare_final_pipeline import reference_at
OUT=ROOT.parent/'output/reacquisition_motion_20261029'
TRANS=ROOT.parent/'output/transport_stability_20261029'
STYLE="""body{font:17px/1.8 'Microsoft YaHei',sans-serif;background:#f1f5fa;color:#183047;max-width:1160px;margin:32px auto;padding:0 22px}section{background:white;border-radius:12px;padding:20px 28px;margin:20px 0}h1{font-size:30px}h2{font-size:23px}table{width:100%;border-collapse:collapse;font-size:15px}td,th{text-align:left;padding:9px;border-bottom:1px solid #dbe4ee}img{max-width:100%}a{color:#185d98}code{background:#eaf0f7;padding:2px 4px}.note{background:#fff1cc;padding:12px}"""


def load(path):return json.loads(path.read_text('utf-8'))
def rows(path):return [json.loads(x) for x in path.read_text('utf-8').splitlines()]


def runtime(run,start,end):
    r=rows(run/'predictions.jsonl');cost=rows(run/'timings.jsonl')[1:]
    selected=[v for v in cost if start<v['t']<end]
    observed=sum(v['body'].get('detected',False) for v in r if start<v['t']<end)
    valid=sum(v['valid'] for v in r if start<v['t']<end)
    last=max((v for v in r if start<v['t']<end),key=lambda x:x['t'])
    elapsed=last['t']-start+last['cost']['completion_lag_ms']/1000
    s=load(run/'summary.json')
    return dict(window=[start,end],source_ticks=sum(start<v['t']<end for v in r),
        processing_FPS=1000/np.mean([v['total_service_ms'] for v in selected]),
        fresh_observation_hz=observed/elapsed,valid_forecast_hz=valid/elapsed,
        completion_lag_p95_ms=float(np.percentile([v['completion_lag_ms'] for v in selected],95)),
        first_result_after_s=s['first_result_after_s'],NN_calls_whole_run=s['nn_frames'])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':150})
    replay=load(OUT/'replay_velocity_only/summary.json');c=replay['streams']['side_search']
    original=rows(TRANS/'RAW_side_search8s/predictions.jsonl')
    patched=rows(OUT/'replay_velocity_only/side_search_patched.jsonl')
    t=116.465502;before=min(original,key=lambda r:abs(r['t']-t));after=min(patched,key=lambda r:abs(r['t']-t))
    labels=sorted(load(ROOT.parent/'output/dense_labels_side_110_1195/labels.json')['records'].values(),key=lambda x:x['t'])
    target=reference_at(labels,t+.4)
    critical=dict(t=t,old_error_px=float(np.linalg.norm(np.asarray(before['physical']['0.4'])-target)),
        new_error_px=float(np.linalg.norm(np.asarray(after['physical']['0.4'])-target)),
        old_velocity=before['coefficient']['velocity'],new_velocity=after['coefficient']['velocity'],
        source_position_unchanged=before['coefficient']['position']==after['coefficient']['position'])
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.5),layout='constrained')
    ts=np.arange(t-.14,t+.401,.01);centers=[reference_at(labels,x) for x in ts]
    valid=[i for i,p in enumerate(centers) if p is not None]
    axes[0].plot(ts[valid],[centers[i][0] for i in valid],'k-',label='人工框中心（离线参考）')
    for r,color,title in [(before,'#bd7f22','原门槛'),(after,'#17816c','保留实测位移')]:
        axes[0].plot([t,t+.1,t+.2,t+.4],[r['coefficient']['position'][0]]+[r['physical'][str(h)][0] for h in (.1,.2,.4)],'o--',color=color,label=title)
    axes[0].axvline(t,color='gray',lw=1);axes[0].set_xlabel('源时刻 / s');axes[0].set_ylabel('水平位置x / px')
    axes[0].set_title('同一当前框，避免把框扩张变成速度');axes[0].legend(fontsize=9)
    pair=c['fixed_original_origins']
    # fixed_original_origins.old is the historical baseline, not this new
    # unpatched RAW observation stream. Use the latter at the same24 times.
    a=load(TRANS/'search_side_scores.json')['matched_original']['new_matched_error']['mean']
    b=pair['new_matched_error']['mean']
    axes[1].bar(['原门槛','修复后'],[a,b],color=['#bd7f22','#17816c'])
    for i,x in enumerate([a,b]):axes[1].text(i,x+1,f'{x:.2f}px',ha='center')
    axes[1].axhline(30,color='gray',ls='--',label='30px目标');axes[1].set_ylim(0,55)
    axes[1].set_ylabel('同24个原起点，0.4秒均值 / px');axes[1].set_title('观测与评分起点固定');axes[1].legend(fontsize=9)
    fig.savefig(OUT/'reanchor_effect.png');plt.close(fig)
    details={};table=[];perf=[]
    for name,folder,scorefile,start,end,title in [
        ('side','RAW_side_default8s','RAW_side_default_scores.json',110.,114.,'侧视：默认候选'),
        ('other','RAW_other_patch_control6s','RAW_other_patch_control_scores.json',40.,46.,'另一段：默认候选'),
        ('side_search','RAW_side_velocity_only8s','RAW_side_velocity_only_scores.json',110.,114.,'侧视：最终速度分工＋可选搜索降频'),
        ('other_search','RAW_other_burst_patch6s','RAW_other_burst_patch_scores.json',40.,46.,'另一段：加可选搜索降频')]:
        s=load(OUT/scorefile);p=s['matched_original'];rt=runtime(OUT/folder,start,end)
        details[name]=dict(score=s,runtime_window=rt)
        missing=', '.join(f"{r['t']:.6f}" for r in p['records'] if not r['available'])
        table.append(f"<tr><td>{title}</td><td>{p['new_available']}/{p['original_origins']}</td><td>{p['new_matched_error']['mean']:.2f}px</td><td>{p['new_matched_error']['p95']:.2f}px</td><td>{missing or '无'}</td></tr>")
        perf.append(f"<tr><td>{title}</td><td>{start:g}–{end:g}s</td><td>{rt['processing_FPS']:.1f}</td><td>{rt['fresh_observation_hz']:.2f}</td><td>{rt['valid_forecast_hz']:.2f}</td><td>{rt['completion_lag_p95_ms']:.1f}</td></tr>")
    summary=dict(critical_case=critical,streams=details,
        same_observations_fixed24=dict(old=a,new=b),
        promotion=dict(contained_patch_in_target_candidate=True,empty_search_default=False,raw_slot_default_events=2000000),
        target_30px_and_stable_30Hz_achieved=False,
        limits='Exposed development labels, short paced RAW replay, startup excluded from steady cadence; no official benchmark claim')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>重入画面速度修复与实际预测</title><style>{STYLE}</style>
<h1>重入画面速度修复与实际预测</h1><p>2026-10-29 · 修复已接候选主线；30px与持续有效30Hz尚未同时达到。原发布ZIP与默认权重保持冻结。</p>
<section><h2>这次实际改了什么</h2><p>重入画面时，弱框只盖住机体一部分；强NN给出完整框后，中心会因框扩张跳动。新修复把“整机位置校正”和“原区域的真实位移”分开：当前强框确认包含旧区域，且当前双向跟踪／刚体拟合足够可靠时，沿用已有位移来约束速度。没有新增传统并行预测器，五圆、独立侧轴、周期有效推力与总扰动反馈继续运行。</p><p>仅新增严格资格许可：包含≥80%、面积比≤4、≥12点／内点、内点率≥75%、位移方差≤9px²；当前强锚与旧弱锚置信度沿用现有规则。没有按未来误差调参数，没有把弱框升级成整机或桨盘观测。</p></section>
<section><h2>一处能明确归因的精度收益</h2><p>116.465502秒原速度约+69px/s，图像位移支持约−885px/s；原门槛因框IoU不足而不用它。最终修复使该起点0.4秒误差<strong>{critical['old_error_px']:.2f}→{critical['new_error_px']:.2f}px</strong>。同批24个原可用起点，均值<strong>{a:.2f}→{b:.2f}px</strong>，未删预测起点，当前框与周期不变；最终只有该重锁起点的评分改变。</p><p>第一版把跨部分／整机参考的局部速度也加入求导历史，导致后续116.632167秒35.66→90.01px。最终版把这个过渡样本标成“仅当前速度”，不加入加速度和未知力求导；后续完整机体参考的观测照常使用。该后续误差恢复原35.66px。初版32.61px及退化保留在replay目录。此前约79px口头数字算错，精确单点为20.85px。</p><img src="reanchor_effect.png"><p>六条冻结观测均先复原原计算最大差0；旧侧视26起点31.62px、另一段15起点59.79px均不变，FRED 7两条120帧观测亦逐项不变。它修复特定重锁失效，不是每段都会触发。</p></section>
<section><h2>最新完整RAW结果</h2><table><tr><th>运行</th><th>原起点保留</th><th>0.4秒均值</th><th>P95</th><th>漏掉的原起点 / s</th></tr>{''.join(table)}</table><p>每行只在保留的原起点上计算，漏失完整列出。不同覆盖率的均值不能直接当算法胜负。当前默认侧视与上一批实际RAW共有24起点为35.00→32.42px；另一段共有14起点为62.56→58.93px，但这包含异步观测时序改变，不能全部归因于新门槛。</p><p class="note">新默认侧视和另一段运行中，本次新增门槛均未触发，按同观测旧／新计算逐项一致。因此31.74／58.93px是实际全流程成绩，不能归因于这一门槛。最终可选搜索版确实触发了一次修复，其自身26个5Hz评分起点同观测旧／新47.39→32.90px；对应原26起点保留25，最终32.13px。两种起点定义分开记录。</p></section>
<section><h2>速度与有效输出分开看</h2><table><tr><th>运行</th><th>固定窗口</th><th>处理能力FPS</th><th>当前观测Hz</th><th>有效预测Hz</th><th>完成滞后P95 ms</th></tr>{''.join(perf)}</table><p>源流固定30Hz，处理能力高于30不代表每秒都有30个有效预测。侧视固定前4秒，另段完整6秒含入画前空场；当前观测率／预测率因获取与缺席而较低。所有本批4条最终运行的有效0.4秒预测，在源时钟下计算完成时均未过期；这不是相机在线部署证明。侧视默认首个结果约19.14秒，包含文件定位／模型启动，未计入steady吞吐；行写出成本另记timings。</p><p>相邻另一段原调度P95约15.36ms，降频约16.16ms，两者都很快。旧运行531ms的滞后不能全部归功于搜索降频；缓存、运行条件和异步应用时刻均不同，没有测出各自的独立贡献。</p></section>
<section><h2>保留与下一步</h2><p><code>--target-candidate</code>采用部分区域速度修复，<code>--no-contained-reanchor-motion</code>保留控制。<code>--empty-search-stride</code>仍可选：初始0.3秒密集搜索，随后空场10Hz、近期失锁立即搜索；侧视覆盖由24恢复25，但另一段仍比原调度多漏一个原起点，暂不默认。3M共享槽也保持显式可选，见<a href="../transport_stability_20261029/index.html">大事件包与搜索调度记录</a>。</p><p>下一步应处理重入时尚未形成可靠速度的短过渡、NN锚应用相位造成的结果波动，以及另一段变向期没有稳定周期输入的问题。五圆／侧轴资格与运动资格继续分工；不把缺桨身份设为侧视门槛，不靠放行不可信推力或省略难起点达标。</p></section>
<section><h2>可复查记录</h2><p><a href="summary.json">本批汇总</a> · <a href="replay_velocity_only/summary.json">最终六条同观测对照</a> · <a href="replay/summary.json">初版及其退化</a> · <a href="mechanism_checks_final.json">最终机制检查</a> · <a href="fred7_frozen_regression.json">FRED 7固定观测</a> · <a href="RAW_side_default_scores.json">默认侧视评分</a> · <a href="RAW_other_patch_control_scores.json">默认另一段评分</a> · <a href="RAW_side_velocity_only_same_observations.json">最终RAW同观测控制</a> · <a href="source_snapshot_final/manifest.json">当前源码清单</a></p><p>已有<a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.pdf">OC-SORT</a>讨论失锁后运动状态误差；本批复用已有<a href="https://docs.opencv.org/4.6.0/dc/d6b/group__video__track.html">OpenCV LK接口</a>和相似变换代码，未复制OC-SORT或实现其虚拟轨迹重更新，未宣称首创。</p></section></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    cap=load(TRANS/'slot_capacity_env_fixed/summary.json');v=cap['variants'];warm=load(TRANS/'capacity_initial_scores.json')
    (TRANS/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>事件包与搜索调度</title><style>{STYLE}</style><h1>事件包传输与空场搜索</h1>
<section><h2>大包不再走字节队列回退</h2><p>可选<code>--raw-slot-events 3000000</code>将共享槽2M→3M，60个原生截止窗的raw／filtered x/y/p/t及边界完全相同；超过槽容量的包6→0。原回退也不丢事件。共享内存128,000,000→192,000,000字节，多64MB，默认仍2M。</p><p>2秒探针总耗时19.91→17.35秒，其中逐字段哈希也会反压生产者，缓存条件不同，不作为干净加速率。完整108–118秒3M运行滞后P95约1667ms，旧2M约2899ms，仍远不能实时；3M运行72.4%的有效0.4秒预测在源时钟下计算好时已过期。大包复制只是其中一个问题，未宣称SDK解码停顿全解决。</p></section>
<section><h2>搜索节奏：速度与召回必须一起记录</h2><p>旧空场每帧请求NN；可选新策略在最近失锁时仍立即搜索，长期空场使用普通stride。最初按NN成功时间判“最近”，会推迟两次近期图像失锁，已改成当前真实图像观测时间。再为初始0.3秒保留密集获取。</p><p>早期另一段新策略55.02处理FPS、14/15原起点58.79px，而旧实际同14起点62.56px；侧视第一版24/26且出现398px重锁异常，不能直接推广。重锁与求导资格已修复，最新组合侧视25/26约32.13px、另一段13/15约59.69px。相邻原调度也低延迟，降频的独立速度收益小于跨次数字表象。新策略继续显式可选。</p><p><a href="../reacquisition_motion_20261029/index.html">查看已完成的精度修复与最新默认结果</a></p></section>
<section><h2>证据</h2><p><a href="slot_capacity_env_fixed/summary.json">事件字段一致性探针</a> · <a href="capacity_initial_scores.json">3M完整RAW与过期审计</a> · <a href="search_side_scores.json">未修复搜索版侧视</a> · <a href="search_observed_scores.json">初版修正版另一段</a> · <a href="search_burst_mechanism.json">初始搜索机制检查</a></p><p>实现复用<a href="https://docs.python.org/3/library/multiprocessing.shared_memory.html">Python共享内存接口</a>，不改SDK或RAW内容。未进行新公开发布。</p></section></html>''','utf-8')
    print(json.dumps(dict(report=str(OUT/'index.html'),critical=critical,
        runtime={k:v['runtime_window'] for k,v in details.items()}),indent=2))


if __name__=='__main__':main()
