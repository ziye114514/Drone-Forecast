"""Summarize current FRED transfer results without changing frozen outputs."""
from pathlib import Path
import hashlib,html,json,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/fred_transfer_20261029'


def load(p):return json.loads(p.read_text('utf-8-sig'))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':150})
    baseline=load(OUT/'scores.json');motion=load(OUT/'motion_observability_v3/summary.json')
    actual=load(OUT/'scores_motion_supported.json')['1']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    names=['FRED 0','FRED 1','FRED 2','自采侧视','自采另一段']
    keys=['fred0','fred1','fred2','side','other'];a=[];b=[]
    for k in keys:
        scores=motion[k]['scores']
        metric=lambda v:v['error_04s_px']['mean'] if k.startswith('fred') else v['physical_future_error_px']['0.4']['mean']
        a.append(metric(scores['control']));b.append(metric(scores['motion']))
    x=np.arange(5)
    axes[0].bar(x-.18,a,.36,label='原候选');axes[0].bar(x+.18,b,.36,label='运动资格分工')
    axes[0].axhline(30,color='gray',ls='--',lw=1);axes[0].set_xticks(x,names,rotation=15)
    axes[0].set_ylabel('0.4秒平均欧氏误差 / px');axes[0].legend();axes[0].set_title('固定当前输入、相同预测时刻')
    for xx,yy in zip(x,b):axes[0].text(xx+.18,yy+1,f'{yy:.2f}',ha='center',fontsize=8)
    for tag,color,label in [('control','#a76317','原候选'),('motion','#187c52','运动资格分工')]:
        target=motion['fred1']['scores'][tag]['targets']
        good=[r for r in target if r['available']]
        axes[1].plot([r['t'] for r in good],[r['error_px'] for r in good],'.-',color=color,label=label)
    axes[1].axhline(30,color='gray',ls='--',lw=1);axes[1].set_xlabel('FRED 1 当前时刻 / s')
    axes[1].set_ylabel('预测误差 / px');axes[1].set_title('FRED 1：全部20个目标保留');axes[1].legend()
    fig.savefig(OUT/'forecast_comparison.png');plt.close(fig)
    rows=[]
    for seq,r in baseline.items():
        s=r['score'];m=motion['fred'+seq]['scores']['motion'];q=s['current_center_error_px']
        rows.append(f"<tr><td>FRED {seq}</td><td>{s['current_outputs']}/{s['annotation_source_ticks']}</td><td>{q['median']:.2f}</td><td>{s['predicted_target_slots']}/{s['target_slots']}</td><td>{s['error_04s_px']['mean']:.2f} → {m['error_04s_px']['mean']:.2f}</td><td>{m['error_04s_box_diagonal']['mean']:.2f}</td></tr>")
    speed=[]
    for folder,label in [('RAW_1_motion_supported','FRED 1 运动分工'),('RAW_clear_integrated2s','清晰2秒：集成候选'),('RAW_clear_inertial2s','清晰2秒：惯性搜索初值试验')]:
        p=OUT/folder/'summary.json'
        if p.exists():
            v=load(p);speed.append(f"<tr><td>{label}</td><td>{v['steady_capacity_fps']:.2f}</td><td>{v['paced_fresh_observation_rate']:.2f}</td><td>{v['timing_ms']['completion_lag_ms']['p95']:.2f}</td></tr>")
    diagnostic=load(OUT/'past_box_controls/summary.json')
    past=[]
    for seq,modes in diagnostic.items():
        past.append(f"<tr><td>FRED {seq}</td><td>{modes['same_observed_pv_CV']['error_04s_px']['mean']:.2f}</td><td>{modes['past_provided_box_CV']['error_04s_px']['mean']:.2f}</td><td>{modes['past_provided_box_force_control']['error_04s_px']['mean']:.2f}</td></tr>")
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>FRED迁移、运动资格优化与录像差异</title>
<style>body{{font:17px/1.7 'Microsoft YaHei',sans-serif;max-width:1140px;margin:35px auto;padding:0 22px;background:#f3f6fa;color:#18263b}}section{{background:white;padding:22px 28px;margin:20px 0;border-radius:12px}}h1{{font-size:29px}}h2{{font-size:23px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{border-bottom:1px solid #dbe2ed;padding:10px;text-align:left}}img{{max-width:100%}}code{{background:#edf1f7;padding:3px 6px}}.note{{background:#fff1ce;padding:12px}}a{{color:#1559a3}}</style>
<h1>FRED迁移与当前运动观测优化</h1><p>2026-10-29 · 目标：0.4秒平均误差&lt;30px，处理能力30FPS。目标尚未全面完成。</p>
<section><h2>本批得到的实际改进</h2><p>把整机语义置信度、运动可观测性与桨盘几何资格分开。弱小目标必须有3个不同的近期NN观测，以及当前至少12个双向匹配点、合格刚体拟合和足够低的位移散布，才能复用已算出的运动。检测框、置信度、五圆/逐桨资格和有效输出时刻不因此升级。</p>
<p>FRED 1同输入：48.4871→43.1819px，P95 82.6102→67.2605px；真实RAW接线43.1708px。FRED 2保留0.067px的小幅收益；自采两段本条改动保持34.6049/59.7896px。原基线保留；<code>--target-candidate</code>已包含严格的运动资格分工，<code>--no-weak-motion-support</code>可关闭。</p><img src="forecast_comparison.png"></section>
<section><h2>FRED当前能力</h2><table><tr><th>固定窗口</th><th>当前检出/有标注时刻</th><th>框中心误差中位px</th><th>预测覆盖</th><th>平均误差px：原→优化</th><th>优化误差/框对角线</th></tr>{''.join(rows)}</table>
<p>三个本地旧版FRED原始事件短窗口，在生成预测前，按标注覆盖和尺寸规则固定为0号25–29s、1号38–42s、2号20–24s。标注不进入RAW预测器。使用4字段coordinates.txt，按十进制秒解析，检查帧名微秒时间与33.333ms步长。未来参考只在≤134ms标注间隔内插值，5Hz预测起点；漏检/缺预测同时保留。不是完整官方测试、mAP或未见数据泛化结论。</p>
<p class="note">三段都没有合格的桨叶周期推力输入，几乎没有完整五圆锁定。因此它们可用于整机捕获/运动迁移诊断，不能用来证明逐桨推力反演成功。像素误差还需结合目标尺寸看，23px对小目标也可能很大。</p></section>
<section><h2>另一录像为什么差</h2><p>已确认的因素包括当前周期极少、40秒冷启动没有学出有效增益、部分运动观测被整机资格挡住，以及变向时当前净加速度不能代表未来0.4秒的持续输入。43.632897秒附近，框只越出上边界约0.57px，仍有49个KLT匹配点，但原运动层拒绝了它。</p><p>放开小幅边界截断恢复了运动，但后续强加速度延续过久，固定输入另一段59.7896→68.0584px；双侧力持续性校准未解决。该路径保持显式试验关闭，不能把恢复更多观测当成预测改善。不能据此确定录像本身有质量问题。</p>
<table><tr><th>位置诊断，非RAW-only</th><th>原观测p/v直接CV</th><th>仅过去标注框CV</th><th>过去标注框+未知净力拟合</th></tr>{''.join(past)}</table><p>过去标注框对照只使用时间戳≤当前时刻的框，且按原RAW有效时刻评分。标注不提供真实速度/推力，位置的离散量化也影响差分，因此这是诊断对照，不能称作完美状态或部署功能。单纯换成标注位置仍未消除变向误差。</p></section>
<section><h2>速度的实际界限</h2><table><tr><th>本机短测</th><th>处理能力FPS</th><th>当前新观测Hz</th><th>完成滞后P95 ms</th></tr>{''.join(speed)}</table><p>均保留原始源时刻、不删慢帧。处理能力、有效新观测率与准时完成不同；小FRED目标负载低，不能把其约98处理FPS当成清晰四桨的速度。清晰集成候选最新29.15FPS，惯性初值短测30.03FPS但跟踪分项未加速，尚无稳定提速结论。惯性初值使用上一物理状态，只初始化OpenCV搜索；当前双向匹配仍须通过，默认关闭。</p></section>
<section><h2>保留的试验与下一步</h2><p>局部放大复核、8–11点的空间包围/分组协方差、边界运动恢复、双侧持续性、切向阻力辨识、共享速度写回均有代码和失败结果。切向阻力模型实际没有通过可辨识性条件；已有可见性周期选择器复用后侧视均值变差，未替代默认。默认收益集中在运动资格分工，未宣称识别真实RPM、牛顿推力、风速或质量。</p><p>继续优先查连续录像中的模型/标定历史、未来输入持续性和可靠平面变化；明确区分当前位置、当前净力和未来命令。原两版GitHub ZIP冻结，本批未重新发布，保护目录未改。</p></section>
<section><h2>证据与来源</h2><p><a href="protocol.json">固定协议与输入哈希</a> · <a href="scores.json">三段RAW基线</a> · <a href="motion_observability_v3/summary.json">同输入运动分工</a> · <a href="scores_motion_supported.json">实际优化RAW</a> · <a href="mechanism_checks.json">机制检查</a> · <a href="annotation_alignment.png">标注对齐预览</a> · <a href="past_box_controls/summary.json">过去框诊断</a> · <a href="frequency_gauge/summary.json">现有频率选择器复用</a></p>
<p><a href="https://github.com/miccunifi/FRED">FRED官方项目</a>提供原始事件/标注格式与任务背景；<a href="https://github.com/obss/sahi">SAHI</a>是局部放大检测的相关既有方法，本项目只复用自有单ROI适配器；<a href="https://docs.opencv.org/4.x/dc/d6b/group__video__track.html">OpenCV LK文档</a>提供初值接口；<a href="https://arxiv.org/abs/1712.02402">Faessler等的转子阻力模型</a>提供速度相关阻力参考，原论文属于控制，不证明本项目事件代理的物理有效性。</p></section></html>'''
    (OUT/'index.html').write_text(report,'utf-8')
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True);manifest={}
    names=['run_fast_structure.py','motion_observability.py','shared_image_motion.py','live_local_flow.py','live_flow_spatial.py',
        'live_flow_acceleration.py','adaptive_force_memory.py','tangential_force_response.py','shared_velocity_state.py','live_worker.py',
        'tools/prepare_fred_transfer.py','tools/score_fred_transfer.py','tools/probe_motion_observability.py','tools/probe_same_window_motion.py',
        'tools/probe_fred_position_controls.py','tools/check_motion_observability.py','tools/probe_frequency_gauge.py','tools/report_fred_transfer.py']
    if (snapshot/'manifest.json').exists():
        # A refreshed report must never replace the code that produced an old run.
        for name,digest in load(snapshot/'manifest.json').items():
            if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Frozen source snapshot changed: '+name)
    else:
        for name in names:
            data=(ROOT/name).read_bytes();p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            manifest[name]=hashlib.sha256(data).hexdigest()
        (snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2),'utf-8')
    print(str(OUT/'index.html'))


if __name__=='__main__':main()
