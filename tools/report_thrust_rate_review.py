"""Small diagnosis/report for the user's thrust-rate question; no RAW rerun."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'event_drone_forecast';OUT=ROOT/'output/thrust_rate_review_20261029'
sys.path.insert(0,str(CODE/'tools'))
from compare_final_pipeline import reference_at
def main():
    p=OUT/'experiment/summary.json';result=json.loads(p.read_text('utf-8'));side=result['clips']['side'];other=result['clips']['other']
    rows=[json.loads(s) for s in (OUT/'experiment/other_baseline.jsonl').read_text('utf-8').splitlines()]
    worst=other['worst_forecasts'][0];r=min(rows,key=lambda x:abs(x['t']-worst['t']))
    truth=sorted(json.loads((ROOT/'output/dense_labels_train2_40_46/labels.json').read_text('utf-8'))['records'].values(),key=lambda x:x['t'])
    now=reference_at(truth,r['t']);target=reference_at(truth,r['t']+.4)
    estimate=np.asarray(r['coefficient']['position'])-now;forecast=np.asarray(r['physical']['0.4'])-now;reference=target-now
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    x=np.arange(2);width=.34
    for i,(field,label,color) in enumerate([('qualified_axis','平面法线','#6988ab'),('current_force','当前周期+姿态力输入','#178c9f')]):
        values=[100*c[field]/c['current_body'] for c in [side,other]]
        bars=axes[0].bar(x+(i-.5)*width,values,width,label=label,color=color)
        axes[0].bar_label(bars,fmt='%.1f%%',padding=3)
    axes[0].set(xticks=x,xticklabels=['侧视175次当前整机','另一录像111次当前整机'],ylim=(0,112),ylabel='当前整机观测中的可用比例/%',title='物理输入可用性差异');axes[0].legend(loc='upper right',fontsize=9);axes[0].grid(axis='y',alpha=.2)
    ax=axes[1];ax.scatter([0],[0],color='#263a50',label='当前人工参考中心',s=40)
    ax.scatter(*estimate,color='#2278ad',s=30,label='当前估计中心')
    ax.annotate('',xy=forecast,xytext=estimate,arrowprops=dict(arrowstyle='->',color='#2278ad',lw=2))
    ax.annotate('',xy=reference,xytext=[0,0],arrowprops=dict(arrowstyle='->',color='#26945d',lw=2))
    ax.scatter(*forecast,color='#2278ad');ax.scatter(*reference,color='#26945d')
    ax.plot([forecast[0],reference[0]],[forecast[1],reference[1]],'--',color='#ba6652',lw=1)
    ax.annotate('预测0.4秒后',forecast,xytext=(-12,-18),textcoords='offset points',ha='center',color='#2278ad')
    ax.annotate('人工参考0.4秒后',reference,xytext=(0,13),textcoords='offset points',ha='center',color='#26945d')
    ax.text(-150,70,'未来位置误差151.9px\n当前中心误差6.5px',color='#9b4936')
    ax.set(xlim=(-180,35),ylim=(165,-65),xlabel='相对当前参考中心X/px',ylabel='相对当前参考中心Y/px（向下为正）',title='43.633s：漏掉了接下来的明显下移')
    ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=8)
    fig.savefig(OUT/'input_and_failure.png',dpi=150);plt.close(fig)
    names=['thrust_trend_forecast.py','run_fast_structure.py','tools/probe_thrust_rate_review.py','tools/report_thrust_rate_review.py','docs/THRUST_RATE_REVIEW.md']
    hashes={}
    for name in names:
        data=(CODE/name).read_bytes();q=OUT/'source_snapshot'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    result.update(source_hashes=hashes,default_enabled=False,default_trajectory_changed=False,goal_complete=False,
        interpretation='Force magnitude rate plus plane-normal rate; source consistency and observations required. No Newton thrust or mechanical RPM truth.',
        paper='https://arxiv.org/html/1712.02402')
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    table=''.join(f'<tr><td>{label}</td><td>{side[k]}</td><td>{other[k]}</td></tr>' for k,label in [('frames','源时刻'),('current_body','当前整机'),('qualified_axis','合格平面法线'),('qualified_period','合格周期'),('current_force','当前周期+姿态力输入'),('held_amplitude','旧幅值+当前轴线'),('gain_updates','有效增益更新')])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>两段录像差异与推力变化率</title><style>body{{max-width:1120px;margin:36px auto;padding:0 24px;background:#f4f6fa;color:#20364f;font:17px/1.8 system-ui}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #ccd5df;padding:9px;text-align:left}}pre{{background:#e6edf4;padding:16px;white-space:pre-wrap}}img{{width:100%}}a{{color:#126d9c}}</style>
<h1>为什么另一段59.79px，以及怎样加入推力变化率</h1><p>2026年10月29日。该段来自自身62.68px基线；侧视来自42.49px基线。分别报告两段开发录像，不以视角名称单独解释差异。</p>
<table><tr><th>实际输入</th><th>侧视110–118秒</th><th>另一录像40–46秒</th></tr>{table}</table>
<p>另一录像主要缺少连续可信的周期/姿态力输入，增益没有标定成功。转向时还会失去当前图像运动资格或把已有加速度外推过头；定位中心准，不代表未来位置准。</p>
<img src="input_and_failure.png" alt="物理观测覆盖及典型变向错误">
<h2>应当预测推力向量的变化</h2><pre>F_T = T * n
dF_T/dt = (dT/dt) * n + T * (dn/dt)
jerk = d(acceleration)/dt
p(t+h) ≈ p + h*v + 0.5*h²*a + (h³/6)*jerk</pre>
<p>大小变化和机体平面法线变化分别进入预测；不需要头尾或电机编号。过去推力的作用已有一部分体现在当前速度和积分反馈里，新增的是未来变化趋势。</p>
<p>当前未知质量/尺度下，仍使用相对f²与有效图像增益，不把事件频率直接当真实RPM或牛顿推力。增益重标定的变化不当作真实推力导数。
<a href="https://arxiv.org/html/1712.02402#S4">Faessler等式(16)</a>也将推力导数、机体转动和阻力变化联系到加加速度；该文研究控制，不能替代本项目的观测验证。</p>
<h2>本轮做了什么，结果怎样</h2><p>修正实验分支的历史管理：完整四桨、整机代表周期和稳定坐标周期各自保留实际观测，短缺测不清掉全部历史。
部分桨均值不能借用另一群体的斜率；旧幅值和几何重置不伪装为新测量。原噪声门槛/窗口/幅值边界保留，没有调参挑标框。</p>
<table><tr><th>同输入重放</th><th>原默认</th><th>旧变化率</th><th>改进变化率历史</th></tr>
<tr><td>侧视可应用时刻</td><td>0</td><td>4</td><td>56</td></tr>
<tr><td>侧视0.4秒均值/px</td><td>{side['scores']['baseline']['0.4']['mean']:.7f}</td><td>{side['scores']['old_rate']['0.4']['mean']:.7f}</td><td>{side['scores']['source_aware_rate']['0.4']['mean']:.7f}</td></tr>
<tr><td>另一录像0.4秒均值/px</td><td>{other['scores']['baseline']['0.4']['mean']:.7f}</td><td>{other['scores']['old_rate']['0.4']['mean']:.7f}</td><td>{other['scores']['source_aware_rate']['0.4']['mean']:.7f}</td></tr></table>
<p>历史衔接更完整，但实用轨迹精度未提高，因此变化率保持显式实验。另一录像连续输入仍不足，变化率可用时刻为0；对缺失信息做差分无法自动补出新信息。
没有重跑RAW或测FPS；组件CPU均值约{side['component_CPU_mean_ms']['source_aware_rate']:.3f}ms，不能当完整系统速度。</p>
<p>下一步优先稳定同源周期及其与总推力的关系，同时联合幅值变化、法线变化和未知合力反馈；不以增加可用时刻数代替预测误差收益。</p>
<p><a href="summary.json">结果与机制检查</a> · <a href="experiment/">全部冻结预测</a> · <a href="../motion_consistency_30px_20261028/index.html">上一批实际RAW结果</a></p></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    log=ROOT/'WORKLOG.md';text=log.read_text('utf-8');marker='## 2026-10-29 两段差异与推力变化率'
    block=marker+'\n\n- 用户要求解释34.60/59.79差异并考虑上一时刻推力与变化率；核对175/111当前整机、118/4周期、101/3当前力输入、2/0增益更新。另一段由62.68改善到59.79，不是从侧视退步。\n- 43.632897s当前中心误差6.48px，未来下移漏掉、误差151.86px，当前流资格失败且没有物理输入；不能归咎单纯框中心或视角名称。\n- 新SourceAwareThrustTrendForecast分别保留同源实际周期历史，部分集合被选中不清除独立合格整机周期历史；变化率只能用于匹配当前源，不把隐藏/保持幅值当新观测。包含幅值变化与已有法线方向变化，未知项不随幅值缩放。\n- 同冻结输入侧视变化率4→56可用，但34.6048757→34.6049175px，另一段仍0次、59.7895553px；实用精度无改善，默认不启用。只改善显式--thrust-trend-forecast，不改默认轨迹。没重跑RAW或测速。\n- 恒频零导数/来源隔离/保持幅值不作新观测/几何重置和独立数值积分检查通过。参考Faessler等式16：https://arxiv.org/html/1712.02402 ，该控制论文不证明本项目事件频率代理有效。\n- 报告output/thrust_rate_review_20261029/index.html；原实际RAW报告保留，30px目标未完成，无后台作业。\n'
    if marker not in text:log.write_text(text+'\n\n'+block,'utf-8')
    checkpoint=ROOT/'output/checkpoints/latest_handoff.json';handoff=json.loads(checkpoint.read_text('utf-8'))
    handoff['thrust_rate_review']=result;handoff['latest_diagnostic_report']=str(OUT/'index.html')
    handoff['current_work_state']='Thrust-rate source histories improved; no practical accuracy gain, remains optional; default RAW means34.6049/59.7896px unchanged'
    handoff['next_action']='Stabilize source-consistent periodicity-to-force mapping; combine thrust amplitude rate and normal rate without inventing observations; diagnose weak-clip reversal43.633s'
    checkpoint.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),'utf-8')
    entry=ROOT/'output/结果入口.md';text=entry.read_text('utf-8');line='- [2026-10-29 两段录像差异与推力变化率](thrust_rate_review_20261029/index.html) — 原默认轨迹不变，变化率历史4→56时刻但无实用精度收益。'
    if line not in text:entry.write_text(line+'\n\n'+text,'utf-8')
    print(json.dumps(dict(report=str(OUT/'index.html'),default_changed=False,side_rate_ticks=56,side_mean=side['scores']['source_aware_rate']['0.4']['mean'],other_mean=other['scores']['baseline']['0.4']['mean']),ensure_ascii=False))
if __name__=='__main__':main()
