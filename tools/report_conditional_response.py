"""Standalone honest progress artifact, all fixed origins and negative results."""
from pathlib import Path
import sys,json,html,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent

if __name__=='__main__':
    out=WORK/'output/precision_update_20261110';out.mkdir(exist_ok=True)
    result=json.loads((WORK/'output/unknown382_20261110/results.json').read_text('utf8'));cumulative=json.loads((WORK/'output/combined389_20261110/results.json').read_text('utf8'));cases=result['cases']
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True);table=[];aggregate={a:[] for a in ('control','disk','camera','no_future_plane')}
    for ax,(n,c) in zip(axes.flat,cases.items()):
        for arm,color,label in [('control','#7a8598','Preserved357'),('camera','#138a72','Rotor + fixed-image residual')]:
            rr=c[arm]['records'];ax.plot([r['t'] for r in rr],[r['error_px'] if r['valid'] else np.nan for r in rr],'o-',ms=3,color=color,label=label)
        ax.axhline(35,color='#b23e35',ls='--',lw=1);ax.set(title=f'Recording {n}: same {c["control"]["available"]} origins',xlabel='source query(s); gaps = unscorable',ylabel='0.4s FDE(px)',xlim=(10.7,15.7));ax.grid(alpha=.2)
        for a in aggregate:aggregate[a].extend(r['error_px'] for r in c[a]['records'] if r['valid'])
        a=c['control']['error'];b=c['camera']['error'];table.append(f'<tr><td>{n}</td><td>{b["n"]}</td><td>{a["mean"]:.2f}</td><td>{b["mean"]:.2f}</td><td>{b["mean"]-a["mean"]:+.2f}</td><td>{a["p95"]:.2f} → {b["p95"]:.2f}</td></tr>')
    axes.flat[0].legend(fontsize=9);fig.savefig(out/'all_fixed_origins.png',dpi=150);plt.close(fig)
    summary={a:dict(n=len(e),mean=float(np.mean(e)),median=float(np.median(e)),p95=float(np.percentile(e,95)),within35=sum(x<=35 for x in e),within30=sum(x<=30 for x in e)) for a,e in aggregate.items()}
    (out/'summary.json').write_text(json.dumps(dict(source382=summary,cumulative389=cumulative['pooled_original_origins'],target35_achieved=False,weak_native_outline_not_independent_COM_truth=True),indent=2),'utf8')
    old=summary['control'];new=summary['camera']
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>桨叶预测精度更新：35px尚未达成</title><style>body{{font:17px/1.75 system-ui,"Microsoft YaHei";max-width:1120px;margin:35px auto;padding:0 24px;color:#243046;background:#f6f8fc}}section{{background:white;padding:24px;margin:18px 0;border-radius:14px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #dde3eb;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}.status{{border-left:6px solid #bd503e;background:#fff5ef;padding:18px}}code,pre{{background:#eef2f7;padding:4px;white-space:pre-wrap}}a{{color:#136f84}}</style>
    <h1>有实际精度收益，35px目标尚未达成</h1><p>2026-11-10 · 原生桨叶ON/OFF主线 · 所有6段均已暴露开发 · 没有换成机体NN</p>
    <div class="status">同102个原始起点，0.4秒平均误差 <b>{old['mean']:.2f} → {new['mean']:.2f}px</b>（下降{100*(1-new['mean']/old['mean']):.2f}%）。P95 {old['p95']:.2f} → {new['p95']:.2f}px。35px内{old['within35']}→{new['within35']}个，30px内{old['within30']}→{new['within30']}个。不能把均值改善说成所有场景稳定达标。</div>
    <section><h2>实际改动</h2><p>原生ON/OFF → 周期种子与原生支撑 → 四/三/二桨、侧线与五圆约束 → 同一个位置/速度与可见力＋未知力状态 → 原有桨盘方向响应＋固定相机坐标中的未知净力残差 → 未来位置。</p><p>5个小型响应系数由已经发生的原生轮廓位移训练。已知/估计的轴向响应仍沿桨盘法线；额外未知残差不跟着桨盘转，并未改用机体NN。按整段录像留一：每一段的评测模型都没用该段训练，所有推断冻结后才读评分框。74个教师不是人工真值；参数是图像域统计先验，不是隐藏RPM或牛顿推力。</p></section>
    <section><h2>每段都列出，回退保留</h2><table><tr><th>录像</th><th>相同起点</th><th>357基线</th><th>固定坐标残差候选</th><th>变化px</th><th>P95</th></tr>{''.join(table)}</table><p>13号仍退步0.91px，16号38.96px，3/4号约60px。原始起点和全部异常误差见图，没有按录像拼接版本。</p><img src="all_fixed_origins.png" alt="所有六录像相同起点误差，包含退步与大误差"></section>
    <section><h2>收益来自哪里</h2><p>初版额外残差也按桨盘分解：{summary['disk']['mean']:.2f}px；相同分解训练但去掉未来平面演化：{summary['no_future_plane']['mean']:.2f}px，支持原有桨叶方向信息有一定作用。随后让额外残差保留固定相机坐标，桨盘响应仍留原力学链，降到{new['mean']:.2f}px。这不是NN预测，也不是不再使用桨叶；不能把全部统计校准收益说成五圆几何独有创新。</p><p>396保留的慢相位几何累计为39.61px（较初版39.65略好），397与固定坐标残差累计为38.70px，未胜38.62；全部结果保留，没有按录像挑开关。</p></section>
    <section><h2>累计优化与没有采用的方向</h2><p>388统一处理11个已有缓存区间；续接使教师74→91，但额外5段只有14号贡献8个，其他未凭噪声补样本。389累计版保住102个原起点并新增5个；4号同10点63.36→50.61px，但全局41.17px，未胜382，不按录像挑优拼接。</p><p>过程噪声自适应、少量样本直接在线更新、轮廓ECC、增加历史净力变化率、GP残差都未形成更稳定的全局提升，未替换主入口。393短窗口直接力学拟合、394长参数拟合也回退；仅靠把未知力参数拟合得更激进不能解决。</p></section>
    <section><h2>现在能运行的内容</h2><p>新入口：<code>run_rotor_learned_candidate.py</code>，用<code>--response-model</code>显式传入固定模型，默认采用固定图像残差。<a href="../../event_drone_forecast/docs/NATIVE_RESPONSE_CANDIDATE.md">完整运行与训练说明</a>。旧入口和已发布NN包保留。</p><p>实际GPU工厂399复现15号34.298px，与冻结交叉训练结果最大差2.7e−12px；391独立ODE差7.7e−10px，398核验未知残差不随桨盘转动、发布模型保留。默认共享模型用全部6段训练，供新录像测试；不能把交叉训练38.62px说成共享模型在训练录像上的独立成绩。尚无新的完整RAW持续30FPS认证。</p></section>
    <section><h2>结论边界</h2><p>这里评分的是可见原生轮廓框中心，不是独立COM真值。35/30px目标未完成。390的事后“理想状态”诊断使用未来锚点，只作误差分解，不能作为预测结果或达标证据，也不能据此断言误差全是当前状态估计错误。</p><p>相关思路参考<a href="https://rpg.ifi.uzh.ch/docs/RAL21_Torrente.pdf">Data-Driven MPC for Quadrotors</a>和<a href="https://github.com/uzh-rpg/data_driven_mpc">作者仓库</a>；没有复制其GPL控制实现或借用其指标。</p></section>
    <p><a href="../unknown382_20261110/results.json">382全部结果</a> · <a href="../combined389_20261110/results.json">389累计与覆盖</a> · <a href="../factory392_20261110/results.json">392实际工厂</a> · <a href="../../event_drone_forecast/docs/HANDOFF_20261110_CONDITIONAL_RESPONSE.md">当前交接</a></p></html>'''
    (out/'index.html').write_text(text,'utf8');print(dict(report=str(out/'index.html'),summary=summary))
