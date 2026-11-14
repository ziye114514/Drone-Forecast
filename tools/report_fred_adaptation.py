"""Publish local frozen results; no inference, model selection, or relabeling."""
from pathlib import Path
import csv,hashlib,html,json,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'output/fred_domain_adaptation_20261029'
FRED=ROOT.parent/'output/fred_transfer_20261029'


def load(p):return json.loads(p.read_text('utf-8-sig'))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':155})
    adapted=load(OUT/'scores.json');verified=load(OUT/'scores_verified.json');unseen=load(OUT/'scores_unseen7.json')
    fixed=load(FRED/'motion_observability_v3/summary.json')
    integrated=load(FRED/'reader_output_integration.json')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    names=['FRED 1\n本轮开发段','FRED 7\n未参与本轮训练']
    before=[verified['1']['matched']['old_matched_error']['mean'],unseen['matched']['old_matched_error']['mean']]
    after=[verified['1']['matched']['new_matched_error']['mean'],unseen['matched']['new_matched_error']['mean']]
    x=np.arange(2);axes[0].bar(x-.18,before,.36,label='运动资格优化后')
    axes[0].bar(x+.18,after,.36,label='原定位器+小目标复核')
    axes[0].set_xticks(x,names);axes[0].set_ylabel('0.4秒平均欧氏误差 / px');axes[0].legend()
    axes[0].axhline(30,color='gray',ls='--',lw=1)
    for xx,a,b in zip(x,before,after):
        axes[0].text(xx-.18,a+1,f'{a:.2f}',ha='center');axes[0].text(xx+.18,b+1,f'{b:.2f}',ha='center')
    targets=verified['1']['matched']['records']
    axes[1].plot([r['t'] for r in targets],[r['old_error_px'] for r in targets],'.-',label='运动资格优化后')
    axes[1].plot([r['t'] for r in targets],[r['new_error_px'] for r in targets],'.-',label='小目标复核')
    axes[1].axhline(30,color='gray',ls='--',lw=1);axes[1].set_xlabel('FRED 1 预测源时刻 / s')
    axes[1].set_ylabel('0.4秒误差 / px');axes[1].set_title('保留原来的全部20个预测起点');axes[1].legend()
    fig.savefig(OUT/'results.png');plt.close(fig)
    rows=[]
    for key,label in [('0','FRED 0'),('1','FRED 1')]:
        m=verified[key]['matched'];rows.append(f"<tr><td>{label}</td><td>{m['new_available']}/{m['original_origins']}</td><td>{m['old_matched_error']['mean']:.2f}</td><td>{m['new_matched_error']['mean']:.2f}</td><td>{m['old_matched_error']['p95']:.2f} → {m['new_matched_error']['p95']:.2f}</td></tr>")
    m=unseen['matched'];rows.append(f"<tr><td>FRED 7，本轮训练外</td><td>{m['new_available']}/{m['original_origins']}</td><td>{m['old_matched_error']['mean']:.2f}</td><td>{m['new_matched_error']['mean']:.2f}</td><td>{m['old_matched_error']['p95']:.2f} → {m['new_matched_error']['p95']:.2f}</td></tr>")
    runtime=[]
    for path,label in [(FRED/'RAW_clear_integrated2s','清晰26–28秒，完整记录'),(FRED/'RAW_clear_compact2s','清晰26–28秒，精简记录'),(FRED/'RAW_clear_rolling_compact4s','清晰26–30秒，滚动读取+精简记录'),(OUT/'RAW_7_baseline','FRED 7原定位'),(OUT/'RAW_7_verified','FRED 7双模型小目标复核')]:
        s=load(path/'summary.json');runtime.append(f"<tr><td>{label}</td><td>{s['steady_capacity_fps']:.2f}</td><td>{s['paced_fresh_observation_rate']:.2f}</td><td>{s['timing_ms']['completion_lag_ms']['p95']:.2f}</td><td>{100*s['paced_deadline_fraction']:.1f}%</td></tr>")
    val=load(OUT/'parent_validation.json');trained=list(csv.DictReader((OUT/'training/results.csv').read_text().splitlines()))
    # Ultralytics selected best.pt may be an earlier epoch; report the best
    # validation fitness row, not an assumed final-epoch checkpoint.
    best=max(trained,key=lambda r:float(r['metrics/mAP50-95(B)']))
    serial=load(FRED/'compact_output_check.json');domain=load(OUT/'export/verifier_domain.json')
    lean=load(FRED/'lean_history_check.json')
    force=load(FRED/'geometry_force_observer_irregular/summary.json')['sequences']
    m=integrated['verified56_rolling_compact']['matched']
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>FRED迁移、条件检测优化与真实速度</title>
<style>body{{font:17px/1.75 'Microsoft YaHei',sans-serif;background:#f1f5fa;color:#172a40;max-width:1150px;margin:35px auto;padding:0 22px}}section{{background:white;border-radius:12px;margin:20px 0;padding:22px 28px}}h1{{font-size:30px}}h2{{font-size:23px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #dbe3ee}}img{{max-width:100%}}code{{background:#edf1f5;padding:2px 5px}}a{{color:#145a9e}}.note{{background:#fff0ce;padding:12px}}</style>
<h1>FRED迁移与完整RAW流水线优化</h1><p>2026-10-29 · 目标：0.4秒平均误差&lt;30px、30FPS。两项目标仍未在各场景同时达到。</p>
<section><h2>回答：是不是另一录像本身有问题？</h2><p>目前没有证据把59.79px主要归因于录像坏了。FRED中同样有23–80px的场景差异；小目标、有效当前运动缺测、缺少合格桨叶周期和未来输入未知均会影响结果。自采另一段从0秒连续运行到46秒，原15个起点59.7896→59.7890px，几乎不变；冷启动也不是主要解释。</p><p>FRED三个原窗口均没有合格周期推力输入，因而当前FRED成绩主要反映整机定位与运动入口，不能作为五圆二力推力预测成功的证据。FRED 7未参与本轮训练/调参，但项目历史可能看过该序列，不能宣称严格盲测或官方完整基准。</p></section>
<section><h2>保留下来的精度优化</h2><p>第一步已纳入<code>--target-candidate</code>：语义置信度低的整机必须有3次独立近期NN锚点、当前至少12个双向匹配点，以及合格刚体运动和低散布，才允许使用已观测到的位移。位置、语义置信度和逐桨几何资格没有被强行升级。固定输入FRED 1为48.49→43.18px，FRED 2为23.0033→22.9364px，自采两段不变；原控制可用<code>--no-weak-motion-support</code>。</p><p>第二步为可选的小目标复核：保留原模型定位，仅在原观测弱或缺失且有当前搜索约束时调用适配模型。有原框时只复核语义，不替换其位置；原框缺失时仍需真实当前第二模型框通过空间一致性。它不是两次独立证据的概率相乘，也不是靠运动模板伪造检测。</p><table><tr><th>相同起点</th><th>新覆盖/原起点</th><th>复核前均值px</th><th>复核后均值px</th><th>P95 px</th></tr>{''.join(rows)}</table><img src="results.png"><p>FRED 7当前检出从87/120到95/120，IoU≥0.5占全部有标注时刻从51.7%到56.7%；中心误差中位7.90→8.67px，没有改善。预测均值下降3.59px但P95略增，仍远未达到30px。FRED 1首次复核试验用192px上限，成功复核框最大55.9px；随后从训练数据95分位得到56px门限，FRED 7使用该门限，二者协议均保留。</p></section>
<section><h2>为何没有直接换新模型</h2><p>本轮使用FRED 3/4/5原生20ms ON/OFF和原自采1/4号弱标注训练，FRED 6验证，共729训练图、216验证图；训练中事件支持不足的116张正样本只从训练列表剔除，没有改成背景。原评分用FRED 0/1/2与自采2/6未进入本轮训练；只做一次固定3轮训练，没有按轨迹评分搜索超参。</p><p>验证集训练父权重PT模型mAP50={val['metrics/mAP50(B)']:.3f}、mAP50–95={val['metrics/mAP50-95(B)']:.3f}；适配权重最佳验证轮次为{int(float(best['epoch']))}，mAP50={float(best['metrics/mAP50(B)']):.3f}、mAP50–95={float(best['metrics/mAP50-95(B)']):.3f}。这不是对运行中原ONNX重新计算的mAP。召回上升、精度和整体mAP下降。全局换模型后FRED 1同起点43.18→41.30px，但自采另一段59.79→69.31px，因此没有替代原权重。</p><p class="note">默认模型与已发布ZIP均保持原样。小目标复核默认关闭；56px上限来自491个本轮训练正框最大边95分位，元数据与新ONNX哈希绑定。两个神经网络增加计算量，不是免费精度提升。</p><a href="training_alignment.png">原生训练图与标注对齐</a> · <a href="protocol.json">训练协议</a> · <a href="export/verifier_domain.json">训练域门限</a></section>
<section><h2>速度优化落实到了哪里</h2><p>新增<code>--compact-output</code>只精简写出的记录，冻结60时刻的预测/位置/有效性数值完全一致；体积从{serial['total_encoded_bytes']['full']:,}到{serial['total_encoded_bytes']['compact']:,}字节，平均序列化CPU从{serial['mean_serialization_CPU_ms']['full']:.3f}到{serial['mean_serialization_CPU_ms']['compact']:.3f}ms。</p><p>随后新增<code>--lean-history</code>，仅保留上一帧完整计算状态，更旧的诊断保存在磁盘，内存留数值计时与计数。{lean['scaling_records']}条记录的pickle占用{lean['full_history_pickle_bytes']:,}→{lean['lean_history_pickle_bytes']:,}字节，不能等同进程RSS。当前上一帧对象、最终计时与统计输入等值。实际清晰2秒接线31.31处理FPS，计时数值列8760字节；完整内部计算不减少，精确分位数存储仍随长度增长，未声称无限流恒定内存。上述RAW精简输出测速为v1；最新工作v2补回幅值保持年龄、是否真实力及未来命令状态，同60条208,200字节，预测及来源标志等值，v2未单独测速。</p><p>新增可选公开SDK滚动读取<code>--raw-reader-backend rolling</code>，8百万事件有界缓冲；两段240时刻的x/y/p/t及窗口边界逐项一致，包括环形缓冲回绕。NumPy结构体填充字节未定义，不能声称内存每一字节都相同。没有改SDK。一次6秒实测不再出现3秒解码峰值，但不能据此宣布根因已修复，默认仍为iterator。</p><table><tr><th>本机完整RAW短测</th><th>处理能力FPS</th><th>实际新观测Hz</th><th>滞后P95 ms</th><th>准时比例</th></tr>{''.join(runtime)}</table><p>不同窗口/负载和异步NN到达时间影响结果。清晰4秒42.37FPS不能与清晰2秒29.15FPS直接当成同窗加速；同26–28秒精简记录短测32.20FPS，但当前观测仍约28Hz。未把慢帧删掉，也未把预览播放帧率当作检测速度。</p><p>滚动读取虽保留事件场，但改变异步完成调度：自采另一段同15起点59.79→62.13px；56px复核+滚动+精简组合为{m['new_matched_error']['mean']:.2f}px（复核实际未调用），15个预测都保留。因此速度选项尚不作为全局精度候选。低内存历史清晰实际重跑也有最多3.26px预测差，冻结状态等值不等于异步RAW逐帧等值，不能只报更高FPS。</p></section>
<section><h2>五圆与二力模型继续卡在哪里</h2><p>已实现条件反演：aᵢ=Tᵢnᵢ+d，取切线qᵢ⊥nᵢ可消去Tᵢ，得到qᵢᵀaᵢ=qᵢᵀd；多时刻有足够不同的准确法线，才能求局部恒定的二维未知合力，再反求当前推力幅值。法线近乎不变时矩阵秩不足；强行补值只会把先验当观测。</p><p>原型包含法线误差、端点加速度误差和重叠历史的相关性，不靠增加重叠样本数虚降方差。侧视实际有{force['side']['reasons'].get('plane_changes_do_not_identify_two_unknown_components',0)}次方向激励不足、{force['side']['reasons'].get('inferred_amplitude_not_resolved_above_uncertainty',0)}次幅值不显著；另一段有{force['other']['reasons'].get('inferred_amplitude_not_resolved_above_uncertainty',0)}次幅值不显著，最终全部没有通过资格，预测不变。受控模型可恢复已知未知力，不等于真实RAW推力已准确。</p><p>另有按预测时距学习净力持续性的原型，实际两段34.60→34.69、59.79→62.42px；未加入默认。当前主要缺口是可持续的平面/周期观测及未来输入变化，而不是继续增加同类标定系数。下一步应优先提升侧视可测周期和法线的可信度，并用公开独立RPM数据验证，再进入推力变化率；不要把事件密度直接当牛顿推力。</p></section>
<section><h2>补充：不用桨心的侧视周期小试验</h2><p>新增整机轴线方向事件分布方差Var[u(t)]，用中心化消除共同平移，用事件数归一化抑制共同计数变化，再复用SciPy Lomb–Scargle处理缺失时间桶。受控旋转直径产生预期400Hz分量，固定形状的共同事件数波动没有伪造空间周期。没有改既有角度相位、计数FFT或默认推力。</p><p>预先列6个开发时刻，5个有当前整机/平面，3个通过两半窗口条件：侧视约304Hz、清晰约232/254Hz；背景都未通过。侧视另外两处不稳定。清晰26.499735秒当前四桨测得303.25/230.21/256.48/284.43Hz，整机空间方差的231.88Hz仅对应其中一个分量，不能将其平方作为四桨总推力。无独立RPM真值，也没有新增轨迹收益，保持独立原型。<a href="../fred_transfer_20261029/side_spatial_moment/summary.json">全部样本与背景/受控结果</a>。</p><img src="../fred_transfer_20261029/side_spatial_moment/moment_signal.png"></section>
<section><h2>复现证据</h2><p><a href="scores_verified.json">复核同起点评分</a> · <a href="scores_unseen7.json">FRED 7完整评分</a> · <a href="scores.json">直接换模型的回退证据</a> · <a href="../fred_transfer_20261029/reader_output_integration.json">读取/输出完整集成评分</a> · <a href="../fred_transfer_20261029/compact_output_check.json">精简输出等值证据</a> · <a href="../fred_transfer_20261029/lean_history_check.json">低内存历史检查</a> · <a href="../fred_transfer_20261029/lean_history_actual.json">低内存实际RAW差异</a> · <a href="../fred_transfer_20261029/rolling_reader_diagnostic/summary.json">原生事件等值</a> · <a href="../fred_transfer_20261029/geometry_force_observer_irregular/summary.json">二力可辨识性原型</a> · <a href="../fred_transfer_20261029/index.html">第一批FRED报告</a> · <a href="source_snapshot/manifest.json">首版报告快照</a> · <a href="source_snapshot_continuation/manifest.json">后续内存/周期原型快照</a></p><p>来源：<a href="https://github.com/miccunifi/FRED">FRED官方项目</a>；<a href="https://docs.ultralytics.com/modes/train/">Ultralytics训练接口</a>；<a href="https://github.com/obss/sahi">SAHI局部放大相关工作</a>；<a href="https://arxiv.org/abs/2604.18289">事件桨叶相对状态估计</a>；<a href="https://github.com/radimspetlik/effqpdet">公开桨叶频率/姿态项目effqpdet</a>；<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html">SciPy周期图接口</a>。本轮未复制这些项目的作者代码；局部检测/扰动观测本身不是新发明。</p></section></html>'''
    (OUT/'index.html').write_text(report,'utf-8')
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    manifest=snapshot/'manifest.json'
    if manifest.exists():
        for name,digest in load(manifest).items():
            if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=digest:raise ValueError(name)
    else:
        records={}
        for name in ['run_fast_structure.py','live_worker.py','live_decode.py','fast_raw_reader.py','forecast_output.py',
            'motion_observability.py','shared_image_motion.py','geometry_force_observer.py','horizon_force_calibration.py',
            'tools/prepare_fred_adaptation.py','tools/filter_fred_training_support.py','tools/train_fred_adaptation.py',
            'tools/export_fred_adaptation.py','tools/score_fred_adaptation.py','tools/prepare_fred_unseen7.py','tools/report_fred_adaptation.py']:
            data=(ROOT/name).read_bytes();p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            records[name]=hashlib.sha256(data).hexdigest()
        manifest.write_text(json.dumps(records,indent=2),'utf-8')
    print(str(OUT/'index.html'))
    continuation=OUT/'source_snapshot_continuation';continuation.mkdir(exist_ok=True)
    manifest=continuation/'manifest.json'
    if manifest.exists():
        for name,digest in load(manifest).items():
            if hashlib.sha256((continuation/name).read_bytes()).hexdigest()!=digest:raise ValueError(name)
    else:
        records={}
        for name in ['run_fast_structure.py','forecast_output.py','side_moment_period.py','tools/check_lean_history.py',
            'tools/probe_side_moment_period.py','tools/report_fred_adaptation.py']:
            data=(ROOT/name).read_bytes();p=continuation/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            records[name]=hashlib.sha256(data).hexdigest()
        manifest.write_text(json.dumps(records,indent=2),'utf-8')


if __name__=='__main__':main()
