"""Assemble existing stage165 evidence; never launches inference or reads RAW."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,html
import numpy as np

ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
OUT=PROJECT/'output/native_plane165_20261104'


def read(path):return json.loads(Path(path).read_text('utf-8'))
def rows(path):return list(map(json.loads,Path(path).read_text('utf-8').splitlines()))
def f(value):return '—' if value is None else f'{value:.2f}'


def run_summary(name):
    directory=OUT/name;s=read(directory/'summary.json');trace=rows(directory/'publication_trace.jsonl')
    predictions=rows(directory/'predictions.jsonl');valid=[r for r in trace if r['valid']]
    ages=[r['source_age_s']*1000 for r in trace]
    s.update(actual_valid_outputs=len(valid),output_count=len(trace),
        actual_age_median_ms=float(np.median(ages)) if ages else None,
        actual_age_p95_ms=float(np.percentile(ages,95)) if ages else None,
        actual_valid_outputs_with_current_detection=sum(r['valid'] and predictions[r['index']]['body']['detected'] for r in trace))
    s['source_hash_guard_passed']=not read(directory/'protocol.json')['source_changed_during_run']
    return s


def build():
    relative=read(OUT/'relative_observer_scores.json')
    names=['raw_sparse_rates_paced','raw_native_join_nodrop','raw_native_join_paced','raw_relative_bounded']
    runs={name:run_summary(name) for name in names}
    if (OUT/'raw_train2_inside_FOV/summary.json').exists():runs['raw_train2_inside_FOV']=run_summary('raw_train2_inside_FOV')
    for name in ['raw_train2_final_candidate','raw_train2_low_queue','raw_clear_low_queue','raw_clear_sparse_history']:
        if (OUT/name/'summary.json').exists():runs[name]=run_summary(name)
    train2={name:read(OUT/(name+'_scores.json')) for name in ['train2_schedule_control','train2_schedule_guard','train2_schedule_reference','train2_schedule_border','train2_cache_control','train2_cache_border']}
    if (OUT/'raw_train2_final_candidate_scores.json').exists():train2['raw_train2_final_candidate']=read(OUT/'raw_train2_final_candidate_scores.json')
    if (OUT/'raw_train2_low_queue_scores.json').exists():train2['raw_train2_low_queue']=read(OUT/'raw_train2_low_queue_scores.json')
    result=dict(updated_utc=datetime.now(timezone.utc).isoformat(),research_goal_met=False,
        goals=dict(horizon_s=.4,mean_error_px=30,sustained_new_valid_observations_Hz=30),
        source_time_scores=relative,runs=runs,
        paced_actual_scores={name:read(OUT/file) for name,file in [
            ('earlier_nodrop','actual_publication_old.json'),('bounded','actual_publication_bounded.json'),
            ('relative_bounded','actual_publication_relative.json')]},
        original_entry_error_px=44.10575381606213,entry_fix_error_px=40.35609951108299,
        weak_regional=read(OUT/'weak_regional.json'),hybrid_histogram=read(OUT/'hybrid_histogram.json'),
        native_join=read(OUT/'native_join.json'),
        native_history=read(OUT/'native_history_checks.json'),
        registered_transport=read(OUT/'registered_transport.json'),
        spatial_statistics_prototype=read(OUT/'spatial_statistics_pilot.json'),
        independent_recording40_41=read(OUT/'train2_regional/summary.json'),
        train2_fixed_comparisons=train2,
        input_route='native ON/OFF rotor evidence; no body NN or G reconstruction',
        labels='Existing manual body-box centers for development, not physical COM/RPM/force truth',
        actual_score_with_skips_is_not_paired_with_nodrop_source_score=True)
    (OUT/'summary165.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    x=np.arange(2);width=.34
    control=[relative[f'{n}_False']['fixed']['error']['mean'] for n in ['cold','warm']]
    candidate=[relative[f'{n}_True']['fixed']['error']['mean'] for n in ['cold','warm']]
    axes[0].bar(x-width/2,control,width,label='Integrated-center observation')
    axes[0].bar(x+width/2,candidate,width,label='Relative displacement candidate')
    axes[0].axhline(30,color='crimson',ls='--',label='30px goal')
    axes[0].set(xticks=x,xticklabels=['Cold start, 9 queries','Entry history, 9 queries'],ylabel='Mean 0.4s source-time error (px)',ylim=(0,50))
    axes[0].legend(fontsize=8);axes[0].grid(axis='y',alpha=.2)
    for name,label in [('raw_sparse_rates_paced','Earlier no-drop'),('raw_relative_bounded','Bounded + relative')]:
        trace=rows(OUT/name/'publication_trace.jsonl')
        axes[1].plot([r['source_t'] for r in trace],[r['source_age_s']*1000 for r in trace],label=label)
    axes[1].axhline(160,color='crimson',ls='--',label='Expiration budget')
    axes[1].set(xlabel='Observation source time (s)',ylabel='Actual output source age (ms)',title='Real RAW paced replay; dropped packets reported')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(OUT/'progress165.png',dpi=160);plt.close(fig)
    # Show every predefined warm-start query, rather than selecting the best.
    candidate_records=relative['warm_True']['fixed']['records']
    fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
    target=np.asarray([r['target'] for r in candidate_records if r['valid']]);prediction=np.asarray([r['prediction'] for r in candidate_records if r['valid']])
    ax.plot(target[:,0],target[:,1],'go-',label='Manual future box centers')
    ax.plot(prediction[:,0],prediction[:,1],'rx--',label='Predicted future centers')
    for a,b,r in zip(prediction,target,[r for r in candidate_records if r['valid']]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color='.6',lw=.8)
        ax.annotate(f"{r['t']:.1f}s",a,fontsize=8)
    ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim')
    ax.set(xlabel='Event image x (px)',ylabel='Event image y (px)',title='All nine fixed queries; source time +0.4s')
    ax.legend();ax.grid(alpha=.2);fig.savefig(OUT/'prediction165.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    for key,label in [('train2_schedule_control','Same dropped-frame schedule: control'),('train2_schedule_border','Same schedule: reference + border fix'),('train2_cache_border','All67 packets: same candidate')]:
        entries=train2[key]['source']['records'];ax.plot([r['t'] for r in entries],[r.get('error_px',np.nan) for r in entries],'o-',label=label)
    ax.axhline(30,color='crimson',ls='--',label='30px goal');ax.set(xlabel='Predeclared source query (s)',ylabel='0.4s error (px)',title='Different recording: all seven predefined queries')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.savefig(OUT/'train2_recovery165.png',dpi=160);plt.close(fig)
    performance=''.join(f'<tr><td>{html.escape(n)}</td><td>{s["frames"]}</td><td>{s["actual_valid_outputs"]}/{s["output_count"]}</td><td>{f(s["actual_age_median_ms"])} / {f(s["actual_age_p95_ms"])}</td><td>{f(s["steady_pipeline_fps"])}</td><td>{s.get("skipped_observations",0)}</td></tr>' for n,s in runs.items())
    score_rows=''.join(f'<tr><td>{name}</td><td>{f(relative[key+"_False"]["fixed"]["error"]["mean"])}</td><td>{f(relative[key+"_True"]["fixed"]["error"]["mean"])}</td><td>{f(relative[key+"_False"]["fixed"]["error"]["p95"])}</td><td>{f(relative[key+"_True"]["fixed"]["error"]["p95"])}</td></tr>' for key,name in [('cold','冷启动，9/9'),('warm','带入画历史，9/9')])
    train2_table=''.join(f'<tr><td>{html.escape(name)}</td><td>{s["source"]["available"]}/{s["source"]["eligible"]}</td><td>{f(s["source"]["error"]["mean"])}</td><td>{f(s["source"]["error"]["p95"])}</td></tr>' for name,s in train2.items())
    actual=train2.get('raw_train2_low_queue',{}).get('actual')
    final_actual=(f'最终真实RAW候选，预定7查询可用{actual["scored"]}/7，真正发布时间+0.4秒均值{f(actual["error"]["mean"])}px，P95 {f(actual["error"]["p95"])}px。' if actual else '最终真实RAW候选尚待评分；不以缓存试验替代。')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>阶段165：原生桨叶入口、相关观测与真实时延</title><style>body{max-width:1080px;margin:36px auto;padding:0 20px;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#223044;background:#f8fafc}h1,h2{line-height:1.35}section{background:white;padding:22px 28px;margin:20px 0;border:1px solid #dbe4ef;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #dbe4ef;padding:9px;text-align:left}img{max-width:100%}code,pre{background:#edf2f7;overflow:auto}pre{padding:16px}a{color:#1763a2}.warn{border-left:5px solid #c76320}small{color:#526177}</style>
<h1>阶段165：先修真实瓶颈，再看预测收益</h1><p>2026-11-04—11-05 · 目标：0.4秒平均误差≤30px，持续30Hz有效新观测。<strong>目前尚未同时达到。</strong></p>
<section><h2>本轮保留了什么</h2><p>仍按“原生RAW ON/OFF桨叶筛选→四/三/部分可见或侧视轴→当前可测贡献＋不可测合力→二维预测”推进，没有改回整机神经网络，也没有灰度G通道。隐藏桨力归不可测项d，不补隐藏转速。</p><p>保留原生匹配的boundingRect与OpenCV等价模糊、侧轴工作区、GPU固定空间算子与缓冲复用、稀疏周期像素分配、原生字节窗口拼接、远端稀疏图解包及有限深度流水线。桨叶证据不因这些计算优化而减少。新相对位移观测与有限延迟策略均有显式选项，旧控制可运行。</p><p>边缘入画时，先前0初速先验会把真实速度误当大加速。仅在已有暂定入画后，用前两次完整位置初始化速度，原9查询44.11→40.36px。普通冷启动保持旧初始化；这是具体失效修复。</p></section>
<section><h2>预测精度：小收益保留，大误差也如实保留</h2><p>原始匹配位移累计而来的中心不是每帧独立绝对定位。新候选把前一时刻位置连同交叉协方差保留下来，直接更新当前相对位移；当前合格五圆中心仍可提供绝对定位。它与二力分账共用同一个状态，没有新增并行轨迹。</p><table><tr><th>相同当前观测</th><th>原均值px</th><th>新均值px</th><th>原P95</th><th>新P95</th></tr>SCORES</table><p>后续时间延伸：冷启动可评分6个，42.02→35.53px；入画历史可评分8/10个，44.42→37.69px。均值改善，但前9点的P95变差，因此没有宣称稳定性或30px达成。此处是固定源时刻+0.4秒，同一录像的开发参考，不是独立测试或真实发布误差。</p><img src="progress165.png" alt="固定观测误差与实际输出时延"><img src="prediction165.png" alt="全部9个查询预测与未来参考对照"></section>
<section class="warn"><h2>不能用平均FPS掩盖预测已经过期</h2><p>新增按源时间30Hz释放的RAW回放，记录真正写出结果的时刻。发布时间超过观测160ms时，预测明确失效，不把旧状态投影当新观测。早期平均处理能力约31FPS，实际却只有20/97次输出有效。</p><table><tr><th>实际RAW轮次</th><th>处理包</th><th>实际有效/输出</th><th>年龄中位/P95 ms</th><th>平均处理FPS</th><th>跳过包</th></tr>PERFORMANCE</table><p>最后的侧视相对观测候选，61/66次发布有效，年龄中位93.71ms、P95 134.79ms；跳过31/97包。实际有效输出约19.70Hz，<strong>不是持续30Hz新检测</strong>。79个预定30Hz查询有49个可用，真正发布时刻+0.4秒均值46.35px、P95 79.74px。不能拿它与无跳帧的源时刻35.63px直接比较。</p><p>保留全部帧的同版本控制，97包源预测有效93，但这次发布时间全部过期。说明启动积压、解码长尾和模型长尾仍是完整链路的问题。不同负载轮次间的FPS变化不能全部归因某一项优化。RAW寻址/初始化约15秒、GPU和模型构建约6–7秒另外列出，不算入稳态FPS。</p></section>
<section><h2>另一录像发现了大失败，并据此修了机制</h2><p>train2的40–41.2秒有4次当前桨叶候选、0次有效预测。两个人工正样本都在画面右上边缘，不能算完整中心。追加42.5–44.7秒完整视野后，53/64时刻有当前桨叶，但最初7源查询均值185.61px，P95 495.61px。检测到桨叶不等于定位到了无人机中心。</p><p>44.2秒重新捕获的一小片桨叶，曾被当成机体中心，位置参考跳变制造了847px/s速度与约3000px/s²加速度。修复分三步：小碎片不提供新中心；初始小片扩展成更多桨叶支撑时，显式更换跟踪参考，不向速度/力注入位置偏移；边缘匹配只用可共同观测的内部像素，零均值归一化避免亮度与裁剪边界偏差。</p><table><tr><th>同输入候选</th><th>预定源查询</th><th>均值px</th><th>P95 px</th></tr>TRAIN2TABLE</table><p>前四行固定重放第一次真实RAW的同64包时序；控制逐帧中心与原实际运行最大差0px，避免把不同丢帧数当算法改进。全部67包的控制121.64→99.59px，P95 198.68→188.41px；同实际时序185.61→101.22px。全部查询均保留，仍远未30px。最终两条缓存成绩约29–31FPS，<strong>不是完整RAW实时证据</strong>。FINALACTUAL</p><img src="train2_recovery165.png" alt="另一录像全部7查询修复前后误差"><p>原18px初始区域并不是已确认全机中心，支撑缩放阈值0.5/2是保守工程先验，不是已测尺寸或物理不可变定律。部分旋转/深度变化仍会混淆，尚无独立质心真值。归一化到未来人工框对角线，同实际时序平均仍约0.543，也不算高精度。</p><p>弱信号另试每半窗3事件参与区域联合证据，维持逐像素6事件资格：7窗严格像素/选频不变，但原生支撑未增，不接CLI。空场一窗无新种子不代表一般假警率为零。CPU/GPU混合直方图证据不变但明显更慢，也未接CLI。</p></section>
<section><h2>现在真正卡在哪里，接下来怎么攻</h2><ol><li><strong>相邻桨叶图像的对应仍会受自转、形变影响。</strong>相对观测修正了重复计数，尚未完整建模共享前帧带来的观测噪声相关性；局部转子纹理不等于刚体运动。下一步应在同一条状态内利用多个原生弧段的共同平移，保留分歧与不确定性，而不是硬提高某个谱峰权重。</li><li><strong>真实时延长尾。</strong>原生窗口拼接的单算子中位16.10→7.57ms是已保留的小收益；但反复出现约3秒的解码CPU停顿，单独SDK试验未稳定复现，不能宣称GC已修好。需分别记录HAL解码/窗口重组/模型阶段，优先换成有界的原生连续流接口；最新帧策略只能保鲜，无法创造30Hz算力。</li><li><strong>周期量仍是推力代理。</strong>侧视无需四桨身份，但频谱混合来源、倍频与有效系数k/d不可辨识仍存在；目前默认k=200是像素动力学尺度，不是牛顿推力。开放学习本轮0次合格更新。应先保留竞争周期/区间以及可测集合变化的不确定性，不能把未知力全学成任意补偿后称物理测量成功。</li><li><strong>完整视野、足够像素与参考规模。</strong>几何补点不补周期；部分入画不等于知道质心。需在另一录像当前可见区间继续报告漏检、有效新观测率、源误差和发布时间误差，保持失败样本。FRED小目标性能不能借旧NN结果充作纯桨叶成绩。</li></ol></section>
<section><h2>哪些尝试没有进入推荐</h2><p>桨叶LK共同仿射：80次资格，暖启动44.11→46.88px且更慢；粗到细匹配无整线收益；统一曝光延迟冷/暖退化；Matérn净力变化率冷40.48、暖47.07、后续50.58px；局部混合增益学习0次更新；弱区域候选与混合CPU直方图未带来最终检测收益。锁页64MB原生输入传输的收益混合，未接默认。GPU新增两半窗原生复数Fourier统计不改变原种子/选频，但4个清晰窗均未达到同频率9扇区等空间相位资格，未接IPC/推力主线，也未以降低门槛制造成功。保留结果和研究代码，避免下个窗口把这些再当首次想法。</p></section>
<section><h2>文献、代码与交接</h2><p>相对观测参考<a href="https://authors.library.caltech.edu/records/enrkd-zam24">Roumeliotis &amp; Burdick，Stochastic Cloning，2002</a>及<a href="https://intra.ece.ucr.edu/~mourikis/papers/MourikisRoumeliotisBurdick_TRO_07.pdf">SC-KF，2007</a>。这里只移植前后位置克隆及交叉协方差思想，尚未实现完整的跨图像观测噪声模型，没有复制作者代码。</p><p><a href="https://arxiv.org/abs/2610.31384">CoHelix</a>也使用共享投影几何与多桨相位/转速状态，不能把“五圆/共享平面”本身宣称首次创新。该文需要各桨初始化信息；本文原型没有宣称复现其成绩。周期筛选沿用<a href="https://arxiv.org/abs/2603.08386">DDHF</a>等思想，非作者实现或可靠总推力标定。</p><p><a href="summary165.json">完整机器可读汇总</a> · <a href="../../event_drone_forecast/docs/HANDOFF_20261105_NATIVE165.md">最新交接</a> · <a href="raw_relative_bounded/predictions.jsonl">最后侧视实际预测</a> · <a href="raw_relative_bounded/publication_trace.jsonl">真实发布时间</a> · <a href="raw_relative_bounded/skipped_observations.json">被跳过的包及解码时间</a></p><p>没有修改RAW、权重、两处yolov10目录，没有上传GitHub或重打发行ZIP。所有完整推理以启动前源码快照及结束哈希守卫记录；标签仅在推理完成后评分。输出中单算子小试、离线固定观测、非限速RAW与真正限速发布均分别标明。</p></section></html>'''
    (OUT/'index.html').write_text(template.replace('SCORES',score_rows).replace('PERFORMANCE',performance).replace('TRAIN2TABLE',train2_table).replace('FINALACTUAL',final_actual),'utf-8')
    return result


if __name__=='__main__':
    build();print(OUT/'index.html')
