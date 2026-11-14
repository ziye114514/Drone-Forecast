"""Publish one globally selected candidate and all original origins/regressions."""
from pathlib import Path
import sys,json,argparse,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',default='combined410_20261110');p.add_argument('--arm',default='joint');p.add_argument('--factory');p.add_argument('--model',default='models/native_response_plane_v1.json');a=p.parse_args()
    result=json.loads((WORK/'output'/a.batch/'results.json').read_text('utf8'));cases=result['cases'];new=result['pooled'][a.arm];old=result['pooled']['control']
    out=WORK/'output/precision_update_20261110';out.mkdir(exist_ok=True);table=[]
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for ax,(n,c) in zip(axes.flat,cases.items()):
        for arm,color,label in [('control','#7a8598','Preserved357'),(a.arm,'#138a72','One shared native-rotor model')]:
            rr=c[arm]['records'];ax.plot([r['t'] for r in rr],[r['error_px'] if r['valid'] else np.nan for r in rr],'o-',ms=3,color=color,label=label)
        ax.axhline(35,color='#b23e35',ls='--',lw=1);ax.axhline(30,color='#d0a34d',ls=':',lw=1)
        ax.set(title=f'Recording {n}: same {c[a.arm]["available"]} origins',xlabel='source query(s); gaps = unscorable',ylabel='0.4s FDE(px)',xlim=(10.7,15.7));ax.grid(alpha=.2)
        x=c['control']['error'];y=c[a.arm]['error'];table.append(f'<tr><td>{n}</td><td>{y["n"]}</td><td>{x["mean"]:.2f}</td><td>{y["mean"]:.2f}</td><td>{y["mean"]-x["mean"]:+.2f}</td><td>{x["p95"]:.2f} → {y["p95"]:.2f}</td></tr>')
    axes.flat[0].legend(fontsize=8);fig.savefig(out/'all_fixed_origins.png',dpi=150);plt.close(fig)
    history=[]
    for batch,arm,label in [('unknown382_20261110','camera','382: fixed-image residual'),('training401_20261110','shared_outside_all6','401: other-recording teachers'),('visibility402_20261110','visibility','402: visibility context'),('drive406_20261110','drive','406: collective future drive'),('plane409_20261110','plane','409: plane conditioning'),('combined410_20261110','joint','410: cumulative response'),('expanded411_412_20261110','visibility411','411: expanded data'),('expanded411_412_20261110','joint412','412: expanded cumulative')]:
        path=WORK/'output'/batch/'results.json'
        if not path.exists():continue
        d=json.loads(path.read_text('utf8'))
        values=d.get('pooled',{}).get(arm)
        if values is None:
            es=[q['error_px'] for c in d['cases'].values() for q in c[arm]['records'] if q['valid']];values=dict(n=len(es),mean=float(np.mean(es)),p95=float(np.percentile(es,95)),within35=sum(e<=35 for e in es))
        history.append(dict(batch=batch,arm=arm,label=label,**values))
    trail=''.join(f'<tr><td><a href="../{h["batch"]}/results.json">{h["label"]}</a></td><td>{h["mean"]:.2f}</td><td>{h["p95"]:.2f}</td><td>{h["within35"]}/{h["n"]}</td></tr>' for h in history)
    factory=json.loads((WORK/a.factory).read_text('utf8')) if a.factory else None
    factory_text='完整RAW持续30FPS尚未通过本批认证。'
    if factory:
        factory_text=f'实际GPU工厂重放{factory["frames"]}包，候选预测与冻结研究结果最大差{factory["actual_factory_parity_px"]:.3g}px，当前基线状态差{factory["baseline_current_state_delta_px"]:.3g}px。'+factory_text
    (out/'summary.json').write_text(json.dumps(dict(selected_batch=a.batch,selected_arm=a.arm,baseline=old,current=new,history=history,
        target35_achieved_on_this_exposed_weak_reference_mean=new['mean']<=35,target30_achieved=False,
        detection_recall_improvement_measured=False,weak_native_outline_not_independent_COM_truth=True,factory=factory),indent=2),'utf8')
    status='35px均值目标已在这组开发弱参考上达到' if new['mean']<=35 else '35px目标尚未达成'
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>桨叶预测精度更新</title><style>body{{font:17px/1.75 system-ui,"Microsoft YaHei";max-width:1120px;margin:35px auto;padding:0 24px;color:#243046;background:#f6f8fc}}section{{background:white;padding:24px;margin:18px 0;border-radius:14px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #dde3eb;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}.status{{border-left:6px solid #bd503e;background:#fff5ef;padding:18px}}code{{background:#eef2f7;padding:4px}}a{{color:#136f84}}</style>
    <h1>实际精度收益：{old['mean']:.2f} → {new['mean']:.2f}px</h1><p>2026-11-10 · 原生桨叶ON/OFF主线 · 单一共享模型 · {status}</p>
    <div class="status">同6段、102个原始起点，预测0.4秒后可见桨叶轮廓中心。均值下降{old['mean']-new['mean']:.2f}px（{100*(1-new['mean']/old['mean']):.1f}%），P95 {old['p95']:.2f} → {new['p95']:.2f}px。35px内{old['within35']}→{new['within35']}，30px内{old['within30']}→{new['within30']}。这是预测改进；本批没有更换检测前端，不能宣称新测得检出率提升。</div>
    <section><h2>处理逻辑</h2><p>RAW原生ON/OFF → 逐像素周期与支撑滤波 → 原生弧/线及4/3/2可见性、五圆约束 → 同一个位置/速度与可见项＋未知项状态 → 原有旋转桨盘响应＋可见性条件的未知净力延续 → 未来位置。</p><p>观察到的桨盘数量、椭圆扁率连续调节响应；当前无向法线用于平面条件响应，不用机体头尾或桨盘身份决定飞行方向。隐藏桨叶仍归未知项。额外参数是从已成熟的原生轮廓位移学习的图像域统计先验，没有牛顿推力、真实RPM或质量标定。</p><p>401起，训练彻底排除这6个评测源，单一模型用于所有6段；不再把6个留一折误写成共享模型独立成绩。但这些录像都曾被开发使用，继续按结果选择方案仍有开发集选择偏差，不能称全新独立泛化。</p></section>
    <section><h2>所有录像与异常起点</h2><table><tr><th>录像</th><th>相同起点</th><th>357基线</th><th>当前统一候选</th><th>变化px</th><th>P95</th></tr>{''.join(table)}</table><img src="all_fixed_origins.png" alt="六段全部固定起点，保留大误差与缺口"><p>3/4号仍有明显长尾；不能用中位数、单段低于30px或新增覆盖后的不同样本集合冒充整体达标。参考是弱原生轮廓框，4号13.6s还接树枝，评分未删除。102是共同可评分输出数，不是整段无人机检出率分母。</p></section>
    <section><h2>累计过程和代价</h2><table><tr><th>统一版本</th><th>均值px</th><th>P95px</th><th>35px内</th></tr>{trail}</table><p>402的同模型固定可见性对照为37.50px，使用当前/过去可见性降到36.92px；这是该信号的小幅贡献。409的去平面列对照36.94px，启用36.68px。410去平面列36.71px，完整36.46px；同拟合系数把未来驱动增益固定1反而36.34px，因此不能声称未来推力幅值标定本身带来了收益。</p><p>403将可见量/平面退化拆成两个上下文为37.11px；405现有周期续接增加5个输出，但原102点37.46px，未胜402。407参考偏移速度拟合为0，不强行加入。旧基线、所有负结果均保留，没有按录像拼接最好开关。</p></section>
    <section><h2>运行与机制证据</h2><p>入口<code>run_rotor_learned_candidate.py</code>，显式模型<code>{a.model}</code>。<a href="../../event_drone_forecast/docs/NATIVE_RESPONSE_CANDIDATE.md">运行与训练说明</a>。{factory_text}</p><p>404合成固定机体、均匀桨盘摆动与4→2→4可见变化没有创造运动（0px），真实发布年龄核验差6.9e−13px；它是机制检查，不是实际35px成绩。原RAW、SDK、受保护目录及已发布NN包未改。</p></section>
    <section><h2>仍卡在哪里</h2><p>当前方向和频率仍是有歧义的图像代理；周期混合、侧线退化和未来控制变化令0.4秒延续不足。未知力学习已改善统计预测，但尚未唯一分离物理推力/风/重力/阻力。当前评测真值规模与独立性也限制结论。下一步必须用固定模型在新录像检验；不能靠删除困难起点凑35。</p></section>
    <p><a href="../{a.batch}/results.json">当前全部原始结果</a> · <a href="../../event_drone_forecast/docs/HANDOFF_20261110_CONDITIONAL_RESPONSE.md">交接</a></p></html>'''
    (out/'index.html').write_text(text,'utf8');print(dict(report=str(out/'index.html'),current=new))
