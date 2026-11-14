"""Deliver the continuous shared-template milestone without overstating accuracy."""
from pathlib import Path
import json,hashlib,html,py_compile,sys
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/continuous_five_circle_20261026'


def load(p):return json.loads(p.read_text('utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text('utf-8').splitlines()]
def main():
    replay=load(OUT/'replay_final/summary.json');forecast=load(OUT/'forecast_final/summary.json')
    runs={n:load(OUT/n/'summary.json') for n in ['raw_clear_v2','raw_side','raw_clear_axis','raw_shared_arcs','raw_guarded_arcs']}
    current=rows(OUT/'raw_guarded_arcs/predictions.jsonl')
    ablation=load(OUT/'shared_arc_ablation/summary.json')
    axis=[r['structure']['continuous_axis'] for r in current if r['structure'].get('continuous_axis',{}).get('accepted')]
    partial=[r for r in current if r['center_bridge'].get('reason')=='prior_conditioned_partial_center_CI']
    audits={}
    for name in runs:
        p=load(OUT/name/'protocol.json')
        audits[name]=dict(source_snapshot_matches=all(hashlib.sha256((OUT/name/'source_snapshot'/f).read_bytes()).hexdigest()==sha for f,sha in p['sources'].items()),
            raw_unchanged=runs[name]['raw_unchanged'],traditional_prediction_in_runtime=not p['physical_only'])
        assert audits[name]['source_snapshot_matches'] and audits[name]['raw_unchanged'] and not audits[name]['traditional_prediction_in_runtime']
    old=ROOT.parent/'output/flow_acceleration_20261026/actual_side_headless/source_snapshot/src/event_drone_forecast/position'
    unchanged={p.name:p.read_bytes()==(ROOT/'src/event_drone_forecast/position'/p.name).read_bytes() for p in old.glob('*.py')}
    assert len(unchanged)==7 and all(unchanged.values())
    for p in [ROOT/'continuous_five_circle.py',ROOT/'shared_arc_recovery.py',ROOT/'fast_structure_front.py',ROOT/'live_structure_bridge.py',ROOT/'run_fast_structure.py']:
        py_compile.compile(str(p),doraise=True)
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),constrained_layout=True)
    c=replay['results']['clear_release_v2'];x=np.arange(2)
    axes[0].bar(x,[c['original']['template_outputs'],c['continuous']['template_outputs']],color=['#7896ac','#278880'])
    axes[0].set(xticks=x,xticklabels=['Original','Continuous'],ylim=(0,60),ylabel='Source ticks',title='Template availability; NOT new detections')
    f=forecast['results']['clear_release_v2'];axes[1].bar(x,[f['original_CI'],f['continuous_CI']],color=['#7896ac','#278880'])
    axes[1].set(xticks=x,xticklabels=['Original','Continuous'],ylabel='Source ticks',title='Geometry contributes to actual position state')
    axes[2].plot([r['t'] for r in current[1:]],[r['cost']['service_ms'] for r in current[1:]],lw=1)
    axes[2].axhline(1000/30,color='red',ls='--',label='30Hz budget');axes[2].legend(fontsize=8)
    axes[2].set(xlabel='Native source time [s]',ylabel='Service [ms]',title='Final guarded-arc RAW run; target unmet')
    fig.savefig(OUT/'results.png',dpi=140);plt.close(fig)
    active_rates={}
    for name,s in runs.items():
        trace=rows(OUT/name/'predictions.jsonl');p=load(OUT/name/'protocol.json');last=trace[-1]
        # Previously EOF waiting was mixed into the rate. This saved-clock
        # correction ends at the last state completion, before its final log.
        active_rates[name]=s['fresh_image_observations']/((last['t']-p['start'])+last['cost']['completion_lag_ms']/1000)
    table=''.join(f'<tr><td>{name}</td><td>{s["frames"]}</td><td>{s["steady_capacity_fps"]:.2f}</td><td>{active_rates[name]:.2f}</td><td>{s["timing_ms"]["total_service_ms"]["p95"]:.1f}</td></tr>' for name,s in runs.items())
    scores=load(OUT/'raw_side/accuracy.json')['scores']
    scoretable=''.join(f'<tr><td>{h}s</td><td>{scores[h]["cv"]["mean"]:.2f}</td><td>{scores[h]["kinematic"]["mean"]:.2f}</td><td>{scores[h]["physical"]["mean"]:.2f}</td></tr>' for h in ['0.1','0.2','0.4'])
    axis_before=float(np.mean([abs(z['innovation_deg']) for z in axis]));axis_after=float(np.mean([abs(z['posterior_axis_residual_deg']) for z in axis]))
    summary=dict(replay=replay,forecast=forecast,actual=runs,axis_updates=len(axis),partial_center_CI=len(partial),
        axis_fit_residual_before_deg=axis_before,axis_fit_residual_after_deg=axis_after,
        axis_residual_is_not_independent_pose_accuracy=True,new_RAW_source_ticks=sum(z['frames'] for z in runs.values()),
        shared_arc_same_input_ablation=ablation,active_observation_rates_pre_last_log=active_rates,
        stable30=False,trajectory_improvement_proven=False,three_current_hub_recovery_observed=True,
        current_jobs=[],first_RAW_read_failed_then_recovered=True)
    (OUT/'batch_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    checks=dict(snapshots=audits,original_position_core_unchanged=unchanged,
        information_controls=load(OUT/'controls/information_controls.json'),legacy_controls=load(OUT/'legacy_controls/checks.json'),
        render=load(OUT/'raw_guarded_arcs/render_check.json'),arc_controls=load(OUT/'controls/arc_controls.json'),syntax_passed=True,
        post_RAW_runtime_change='Only EOF/active completion timing separated after final RAW; prediction logic unchanged')
    (OUT/'artifact_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),'utf-8')
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>连续五圆与局部信息更新</title><style>body{{font:16px/1.75 system-ui,Microsoft YaHei;color:#203040;background:#f7f9fc;max-width:1100px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:9px;border:1px solid #ccd6e1;text-align:left}}video,img{{width:100%}}.note{{background:#fff1d6;padding:16px;border-left:5px solid #d89e38}}code{{background:#e8eef5}}</style>
    <h1>连续五圆：从切换分支到更新同一共享状态</h1><p>按用户选定方向，优先实现可见性退化时的信息分工。传统加速度/尺度预测没有回到实时循环，位置和速度只作为力学积分的初始状态。</p>
    <div class="note"><b>这批有实际算法改动，但尚未证明轨迹更准。</b> 同一当前观测回放，模板可输出30→58/60，几何参与位置更新17→25次。随后把共享模型用于当前原弧提取：同事件、同整机观测消融中，三桨分支0→1次。新增桨心有128个当前真实轮廓点、79.3%弧覆盖、0.637px拟合残差、0.837当前轮廓面积IoU。不是直接拿模板预测点当检出。</div>
    <h2>改了什么</h2><p>新增<code>continuous_five_circle.py</code>：用同一个六维图像仿射状态表达中心与桨盘布局，沿用原来的四桨初始化和匹配/形状边界。四桨更新共享状态；三桨、两桨只加入当前数据能够约束的信息；轴线只加入一个角度约束，未观测的尺度/压扁方向保留不确定性。持续有局部证据时可维持模型，短时缺证据只保留标为推测的模板。身份改变或整机长时间丢失会清除模板。</p>
    <table><tr><th>当前证据</th><th>图像状态的信息秩</th><th>不能宣称</th></tr><tr><td>四个合格桨心</td><td>6</td><td>完整三维姿态/质心真值</td></tr><tr><td>三个非共线合格桨心</td><td>最多6，结合近期形状先验</td><td>第四桨被直接看见或其转速已知</td></tr><tr><td>两个非扁平合格桨心</td><td>4，保留两个未观测方向的不确定性</td><td>两点独立恢复完整平面</td></tr><tr><td>当前退化轴线</td><td>1，只约束角度</td><td>当前四个隐藏桨心已经测得</td></tr></table>
    <p>只有已经建立四桨参考、当前局部观测通过匹配/形状检查、中心协方差代理最大特征值不超过36px²时，才允许低权重CI把两桨条件中心用于位置。此中心另有<code>model_conditioned_center</code>字段，直接当前几何中心字段仍为空。协方差是模型近似，不是经过校准的统计保证。</p>
    <h2>同输入结果</h2><img src="results.png"><p>旧模型与新模型使用完全相同的已保存当前桨盘候选。原物理预测的p/v与未来点已逐帧精确重建，随后才读取旧人工框参考。清楚片段没有对应时段的人工中心真值，所以不能用模板更完整来代替精度证明。</p>
    <p>进一步复用147的原弧拟合和148的距离/覆盖/形状/面积门控，模板只给搜索范围。所有用于拟合的点来自当前阈值图闭运算之前的外轮廓，互相歧义的像素排除，当前面积IoU必须>=.72。空白和稀疏杂波控制未产生补桨，完整当前椭圆控制成功恢复2桨。</p><p>严格同RAW事件、同当前整机观测、交替执行开关的60时刻几何消融：关闭时four15、partial_two8、three0；开启后four15、partial_two7、three1。单个新增当前桨心通过面积/弧检查。画面实际四桨仍可能可见，three表示当前只接纳三个合格桨心，不是物理上只剩三副桨。该消融不运行NN/周期/未来预测，不能当完整FPS或轨迹优势证据。</p>
    <p>去掉整机当前位移后的中心步长代理：均值0.991→0.804px，P952.849→2.154px，但中位0.247→0.618px变差；这只是结构连续性代理，不是独立定位误差。冷启动侧视没有完整模板，两种方法的14个共同未来目标误差逐项相同。</p>
    <h2>恢复RAW之后的小试</h2><table><tr><th>实际版本</th><th>源更新数</th><th>平均计算能力FPS</th><th>真实新图像观测Hz</th><th>服务P95ms</th></tr>{table}</table>
    <p>完整入口共330次成功RAW源更新，另有60次RAW重解码用于同输入几何消融；均来自同一录像的短窗，不是独立飞行数量。最初E盘读取失败、用户恢复后成功。前两个版本平均计算能力越过30，但最终带面积门控原弧版这次整链仅19.00FPS；不能用早期较快结果代替终版。各CPU阶段均有波动，没有同负载完整流水线配对证据把差异归因于某个模块或硬件。稳定30FPS仍未完成。</p><p>发现旧速率还混入了RAW文件结束等待：最终原记录9.26Hz包含EOF等待。表中改为由保存的最后状态完成时间计算的新观测率，截止于最后一次日志之前；不重写原日志。当前入口已把运行中的新观测率与EOF等待分开记录。此前/此后的计时差异不代表检测或预测精度变化。</p>
    <p>终版清楚片段实际出现{len(partial)}次条件中心更新、{len(axis)}次共享模型轴线更新。轴线拟合残差平均{axis_before:.3f}→{axis_after:.3f}°，只是对同一当前观测拟合得更近，不是姿态真值准确率。侧视RAW缺乏初始化模板，仍走轴线/周期代理，没有虚构四桨。</p>
    <h2>当前物理预测仍有差距</h2><p>新侧视RAW、每时域14个共同起点，人工框参考是历史开发标注，传统方法只在保存当前输出后离线重建。该实际运行与旧运行的异步NN观测时刻略有差异，不是严格单因素精度提升。</p><table><tr><th>时域</th><th>CV均误差px</th><th>离线传统均误差px</th><th>物理均误差px</th></tr>{scoretable}</table>
    <h2>查看当前模型和预测</h2><video controls preload="metadata" poster="raw_guarded_arcs/three_poster.jpg" src="raw_guarded_arcs/results.mp4"></video><p>紫色为当前支持的模型点，灰色为推测模板点，红色为当前拟合候选，黄色为方向和物理预测。视频按源时间30FPS播放，事后渲染，不是测速成绩。</p>
    <h2>下一步真正要改的地方</h2><ol><li>三桨仅1个时刻成功，尚未形成稳定段。引导目前仍要求最近.12s有完整四桨；下一步让合格三桨的信息和几何不确定性决定引导能否续接，避免一退化就失去引导。继续保留当前面积/弧证据要求，不把单纯延长模板寿命当恢复。</li><li>频率池化仍会受区域变化、目标运动及谐波影响。先解决周期含义跨可见性变化的一致性，再把有效增益更新称为推力学习。</li><li>几何与解码调度继续保留低成本前台/低频完整重建方向；用新观测率、完成滞后及尾耗时评价30FPS。</li></ol>
    <p>本轮方法参考<a href="https://arxiv.org/html/2610.31384v1">CoHelix共享几何</a>与<a href="https://www.scitepress.org/PublishedPapers/2014/49060/49060.pdf">已有可见桨数分工研究</a>。信息更新和协方差处理是成熟估计方法；当前候选贡献是其在事件桨盘退化到二维预测中的具体观测分工，尚不声称首创。</p>
    <p><a href="batch_summary.json">完整汇总</a> · <a href="artifact_checks.json">检查记录</a> · <a href="forecast_final/summary.json">同输入预测比较</a></p></html>'''
    (OUT/'index.html').write_text(page,'utf-8');print(str(OUT/'index.html'))


if __name__=='__main__':main()
