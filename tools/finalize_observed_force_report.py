"""Render the bounded stage162 batch and record source/data qualifications."""
from pathlib import Path
import ast, hashlib, html, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'output/observed_force_partition_20261101'


def read(p):return json.loads(p.read_text('utf-8-sig'))
def write(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
def lines(p):return [json.loads(s) for s in p.read_text('utf-8').splitlines()]


def main():
    result=read(OUT/'final_replay/summary.json')
    raw=read(OUT/'raw_clear_frozen/summary.json');protocol=read(OUT/'raw_clear_frozen/protocol.json')
    assert protocol['source_snapshot_before_worker_start'] and not protocol['source_changed_during_run']
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in result['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    actual=lines(OUT/'raw_clear_frozen/predictions.jsonl');replay=lines(OUT/'final_replay/clear_observed.jsonl')
    assert len(actual)==len(replay)==37
    for a,b in zip(actual,replay):
        assert a['t']==b['t'] and a['body']==b['body'] and a['valid']==b['valid']
        np.testing.assert_allclose(a['force']['acceleration_px_s2'],b['force']['acceleration_px_s2'],atol=1e-8,rtol=0)
    # Preserve old rows/protocol; make their qualification explicit.
    warning=dict(source_snapshot_trustworthy=False,timing_qualified=False,
        reason='CUDA source edited while old worker was running; old runner captured at end. Do not use this as final-source speed evidence.',
        successor_final_source_run=str(OUT/'raw_clear_frozen'))
    write(OUT/'raw_side_final/SOURCE_SNAPSHOT_WARNING.json',warning)
    # Changed arithmetic was already compared to old control and to actual RAW.
    parsed=[]
    for name in list(result['source_sha256'])+['tools/probe_sparse_copy.py','tools/finalize_observed_force_report.py']:
        ast.parse((ROOT/name).read_text('utf-8-sig'));parsed.append(name)
    check=dict(raw_source_hashes_match=True, frozen_replay_source_hashes_match=True,
        raw_clear_matches_frozen_observation_replay=True,ast_parsed=parsed,
        labels_only_for_scores=True, no_new_experiments_after_handoff=True,
        raw_side_prior_run_qualified=False)
    write(OUT/'artifact_checks.json',check)
    gpu=read(OUT/'sparse_copy_serial_warm.json')
    result.update(final_raw_clear=raw,artifact_checks=check,sparse_copy_microtest=gpu,
        final_replay=str(OUT/'final_replay/summary.json'),latest_numbered_stage=162,
        accuracy_gain_proven=False,full_pipeline_30Hz_proven=False)
    write(OUT/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    q=[r['query_t'] for r in result['scored_rows']]
    for arm,label in [('stage161','阶段161'),('observed','当前分账'),('off','当前状态关周期')]:
        axs[0].plot(q,[r[arm] for r in result['scored_rows']],'o-',ms=3,label=label)
    axs[0].axhline(25,color='gray',ls=':',label='25px目标')
    axs[0].set(xlabel='固定源查询时刻 / s',ylabel='0.4s终点误差 / px',title='九个开发查询：均值没有提升');axs[0].legend(fontsize=8)
    checks=read(OUT/'final_replay/mechanism_checks.json')
    for speed,label in [('360.0','20%同槽变速（保留响应）'),('600.0','2倍频跳变（尚不能判别）')]:
        rows=checks['step_diagnostics'][speed]['rows']
        axs[1].plot([r['frame'] for r in rows],[r['acceleration'][1] for r in rows],'o-',label=label)
    axs[1].set(xlabel='合成帧序号（第8帧改变）',ylabel='当下y加速度代理 / px/s²',title='机制响应；不代表独立RPM真值');axs[1].legend(fontsize=8)
    fig.savefig(OUT/'results.png',dpi=150);plt.close(fig)
    rows=''.join(f'<tr><td>{name}</td><td>{s["n"]}</td><td>{s["mean"]:.3f}</td><td>{s["median"]:.3f}</td><td>{s["p95"]:.3f}</td></tr>'
        for name,s in result['scores'].items())
    doc=ROOT/'docs/HANDOFF_20261101_FORCE_PARTITION_IMPLEMENTED.md'
    # HTML explanation is local, with no network assets or external scripts.
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>当前可测贡献＋不可测合力</title>
    <style>body{{font:17px/1.8 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#263b50}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #b9c6d3;padding:8px}}.note{{padding:16px;background:#fff2ce}}pre{{white-space:pre-wrap;background:#edf3f8;padding:16px}}a{{color:#1559a0}}</style>
    <h1>阶段162：可测／不可测分账已接入，周期预测收益仍待解决</h1>
    <p>2026-11-01。只看原生RAW ON/OFF桨叶，不使用G重建或整机NN为前置。隐藏桨力归不可测状态d；四／三／二／侧视共同维护机体位置状态。</p>
    <pre>RAW ON/OFF → 密度与周期过滤 → 原生弧段／侧视线 → 共享几何
    → 当前可见贡献 u（固定单位）＋不可测项 d → 位置／速度／加速度 → 0.4秒预测
    隐藏桨、重力、风、空气阻力等 → d；几何补点 ≠ 补测RPM</pre>
    <p>已解除四路周期首次齐全的bank条件，隐藏历史先验不进可测平方和；当前证据丢失便转入d，重新出现时避免重复加力。集合改变转移均值与协方差，保留同槽真实转速／法线变化。侧视混合周期是变化代理，不能乘4当总推力。</p>
    <h2>实际预测结果</h2><img src="results.png"><table><tr><th>控制</th><th>查询数</th><th>均值px</th><th>中位px</th><th>P95px</th></tr>{rows}</table>
    <p class="note">均值39.54→39.73px，没有改善；当前关周期39.69px。清晰37帧的周期输入16→21，其中12次可见槽、9次混合代理；有效源预测仍33。侧视57/61次有效预测，40次输入全部是混合变化代理，尚无已映射逐桨输入。不能把输入覆盖增加写成预测精度提高。</p>
    <p>同一train6的两个暴露过的短窗、九个相关开发查询；人工整机框中心参考，非重心／RPM／推力真值。源时间评分未计处理延迟，清晰段没有合格未来位置评分。阶段161控制逐帧复原，预测结束后才打开标签。</p>
    <h2>机制修复与剩余负例</h2><p>4→3→2→4不会凭空加速；已映射同槽20%频率变化可即时响应，快速变化提高不确定度且不更新增益。2倍频跳变仍有真实变速／谐波歧义，资格断档后重锚可能吞掉真实变化；合成负例保留，下一步靠相位创新与多候选区分。</p>
    <h2>速度与RAW入口</h2><p>GPU三份稀疏结果已合并一次回传，三类窗口输出相同。交替预热短试总体设备均值45.45→44.32ms，侧视和空场中位反而增加；不宣称稳定整线加速。</p>
    <p>最终RAW清晰1.2秒：37包、36当前区域、33有效源预测、21周期输入。核心{raw['compute_capacity_fps']:.2f}FPS，中位{raw['compute_ms_median']:.1f}ms、P95 {raw['compute_ms_p95']:.1f}ms；启动{raw['worker_and_model_startup_s']:.2f}秒、运行{raw['end_to_end_elapsed_s']:.2f}秒。源码在启动前快照且结束未变。未达持续30Hz；这个短试不能与不同负载的旧速度直接归因比较。</p>
    <h2>下一步已排好</h2><ol><li>复用shared_phase_fft.py，解除其四槽首次初始化要求；逐可见槽或局部区域提供相位资格。</li><li>用相位创新竞争频率／谐波候选，解决真实变速被资格断档吞掉；侧视不强恢复完整五圆。</li><li>保持当前可测＋不可测分账，资格稳定、有激励才慢学k。</li><li>性能另做有界双缓冲、候选频带跟踪与周期重发现，统计新观测年龄及P95。</li></ol>
    <p>文献与已有代码：<a href="https://arxiv.org/abs/2603.09235">HelixTrack</a>、<a href="https://github.com/ros-event-camera/frequency_cam">FrequencyCam</a>、<a href="https://github.com/ethz-asl/mav_control_rw">ETH外扰观测器</a>、<a href="https://arxiv.org/html/2603.08386v1">DDHF</a>和<a href="https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html">NVIDIA最佳实践</a>。本地HelixTrack已有适配，不能重复算首次接入；这些部件均不是本项目全部条件的现成替代。</p>
    <p><a href="{doc.as_uri()}">完整交接与实施顺序</a> · <a href="summary.json">数值与检查</a> · <a href="final_replay/mechanism_checks.json">机制用例与负例</a></p>
    <p>旧对照保留：--rotor-legacy-decomposition；更早残差：--rotor-legacy-force。模型、保护目录、原始RAW和GitHub发行包未改，无本批后台实验。早期沙箱失败和运行中改源码的目录已明确降级，不作为最终速度证据。</p></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    print(json.dumps(dict(checks=check,report=str(OUT/'index.html')),ensure_ascii=False))


if __name__=='__main__':main()
