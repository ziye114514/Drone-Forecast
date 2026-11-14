"""Summarize173 promoted gap fix and174 candidate rim recovery with negatives."""
from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/gap173_20261107'

def load(p):return json.loads(p.read_text('utf-8'))

if __name__=='__main__':
    gap=load(OUT/'integrated/scores.json');rim=load(ROOT.parent/'output/rim174_20261107/integrated/scores.json')
    props=load(ROOT.parent/'output/rim174_20261107/supported_rim_recovery.json');mechanism=load(OUT/'verified_shortgap/summary.json')['mechanism']
    summary=dict(promoted='ShortGapObserver for current rotor-only relative-observation route; NN unchanged',
        experimental_not_promoted='native-rim reconsideration and adaptive force memory',source_time_horizon_s=.4,
        historical_development_data=True,new_RAW_FPS_measured=False,gap=gap,rim=rim,mechanism=mechanism)
    (OUT/'final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name();plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(9,3.8),constrained_layout=True)
    for name,folder,file in [('旧172',OUT/'integrated','train2_control.jsonl'),('丢测一致0.35s',OUT/'verified_shortgap','train2_coherent_gap.jsonl'),('丢测短记忆0.1s',OUT/'integrated','train2_short.jsonl')]:
        if name=='丢测一致0.35s':score=load(OUT/'verified_shortgap/summary.json')['cases']['train2']['coherent_gap']['score']
        else:score=gap['train2']['control' if name=='旧172' else 'short']['score']
        rr=[r for r in score['records'] if r['valid']];ax.plot([r['t'] for r in rr],[r['error_px'] for r in rr],'o-',label=name)
    ax.set_xlabel('预测起点 / s');ax.set_ylabel('0.4s误差 / px');ax.legend();ax.grid(alpha=.2);ax.set_title('先修丢测过冲：44.2s从171.4降至69.2px；其他变向仍未解决')
    fig.savefig(OUT/'gap_fix_errors.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,6.4),constrained_layout=True)
    for col,r in enumerate(props['records']):
        q=round(r['t'],1);p=ROOT.parent/f'output/rim174_20261107/{q:.1f}.npz'
        with np.load(p) as z:on=z['on'].astype(float);off=z['off'].astype(float)
        x,y,X,Y=r['roi'];lim=max(1.,np.percentile((on+off)[y:Y,x:X],99.7));rgb=np.zeros(on.shape+(3,))
        rgb[:,:,0]=np.clip(np.log1p(on)/np.log1p(lim),0,1);rgb[:,:,2]=np.clip(np.log1p(off)/np.log1p(lim),0,1)
        for row,(key,label,color) in enumerate([('old','旧活动核','#ffcc55'),('new','原弧候选','#51dfc4')]):
            ax=axs[row,col];ax.imshow(rgb);ax.set_xlim(x,X);ax.set_ylim(Y,y);ax.set_title(f'{q:.1f}s / {label}')
            for g in r[key]:ax.add_patch(Ellipse((g['cx'],g['cy']),g['major'],g['minor'],angle=g['angle'],fill=False,ec=color,lw=1.8,ls='--'))
    fig.suptitle('同一原生20ms：两个小活动核不能自动视为已解析两个桨盘\n虚线为拟合/外推边界；这不是独立桨心真值或AP验证',fontsize=12)
    fig.savefig(OUT/'native_rim_recovery.png',dpi=150);plt.close(fig)
    table=[]
    for case,name in [('side','侧视17/19起点'),('train2','另一录像7/7起点')]:
        a=gap[case]['control']['score']['error']['mean'];b=gap[case]['short']['score']['error']['mean'];c=rim[case]['short']['score']['error']['mean']
        table.append(f'<tr><td>{name}</td><td>{a:.4f}</td><td>{b:.4f}</td><td>{c:.4f}</td></tr>')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>丢测过冲修复与桨盘原弧恢复</title>
<style>body{max-width:1100px;margin:35px auto;padding:0 22px;background:#f4f7fb;color:#20324a;font:17px/1.85 "Microsoft YaHei",sans-serif}h1{font-size:29px}h2{margin-top:32px}table{width:100%;border-collapse:collapse;background:white}td,th{border:1px solid #cad5e3;padding:10px}img{max-width:100%}a{color:#155da0}code{background:#e4ebf4;padding:2px 4px}.note{background:#fff0ca;padding:16px;border-left:5px solid #bf861d}</style>
<h1>先修丢测过冲，再处理桨盘原弧与位置参考</h1><p>2026-11-07 · 阶段173默认修复、174候选提取 · 纯桨叶前端，无NN/权重/灰度重建。</p>
<p class="note">当前默认：另一录像0.4s均值102.04→87.08px，侧视38.73px不变；44.2s的171.41→69.16px。保留全部固定起点与有效性，没有达到30px。桨盘原弧候选恢复有具体进展，但尚未稳定改善预测，暂不接管默认位置/力学输出。稳定RAW30Hz亦未验证。</p>
<h2>1. 已修并接入：丢测时的未知/净加速度延续</h2><p>旧代码有测量时估计运动，丢测后仍按恒定加速度推进内部状态；对外却用Singer指数衰减外推。同一丢测区间被拆成数次空更新时，内部状态不再遵循已发布的条件预测。44.2s前连续两次没有有效位置，旧负加速度会把竖直速度从正值推进到负值，随后再向未来延续而反向过冲。</p>
<p>新CoherentGapObserver在无有效位置时复用已有Singer转移，内部状态与该模式的发布曲线一致。ShortGapObserver进一步仅在位置证据缺失时使用项目既有0.1s短记忆先验，保留同一加速度稳态方差和更大的过程扩散。有位置时的更新方法不改；没有增加第二条轨迹、伪造新位置、改转速资格或把预测计作测量。</p>
<p><b>0.1s是工程先验，并非辨识出的风力/隐藏桨动力学常数。</b>纯一致0.35s已经使102.04→98.21px；短记忆降到87.08px。没有在这些标签上扫时间常数。首次丢测切换先验表示降低未被当前观测支持的延续信用；不能声称新模式与上一有测量时模式永远使用同一先验。</p>
<img src="gap_fix_errors.png" alt="修复前后各预测起点误差"><table><tr><th>相同源时刻0.4s查询</th><th>172旧模式 / px</th><th>173默认短丢测 / px</th><th>174再接管新原弧 / px</th></tr>TABLE</table>
<p>侧视133包、111当前区域、115有效预测；另一段67包、57当前区域、54有效预测，两个版本相同。侧视评分范围没有覆盖后面所有丢测时刻，不能据17点不变声称所有场景均不退化。两段仍是开发录像，不是新盲测。</p>
<h2>2. 已找到并实现候选修复：两个小核不代表两个完整桨盘</h2><p>旧recover只要父轮廓内已有两条合格弧就跳过。43.6s、44.0s已有小椭圆仅覆盖父轮廓当前前景约11%和13%，并没有解释大片原生桨叶支持。新选项先计算已有椭圆对父轮廓的覆盖；低于已有60%支持标准才重新拟合原开放弧，新组合须解释更多当前前景，并继续通过原来的膨胀/独立弧/角度/覆盖/不确定性检查。未新增4桨凑数规则。新候选也使用旋转椭圆在完整传感器上的真实范围标记出画，不把ROI边界当出画。</p>
<img src="native_rim_recovery.png" alt="同一原生画面的活动核与恢复原弧候选对照">
<p>43.6s左侧原10/13px量级小核恢复为约43/30px原弧椭圆；44.0s恢复为约47/36px。43.4s没有够用原弧，保留原输出。人工复核C的4桨恢复也保持。图中完整虚线含外推，不是标注真值，更不意味着转速或推力已准确。</p>
<h2>3. 为什么没有让新原弧直接接管预测</h2><p>实际整段对照中，新候选使侧视38.73→39.22px、另一段87.0806→87.0823px，未形成稳定收益。更重要的是112.033s首次新几何中心使位置参考向上移动约22.6px；虽然现代码同步移动了p/clone、没有直接给v加跳变，输出的“预测哪个点”仍改变了。原生图也显示桨盘平面点、原活动区域中心和包含机体/起落架的人工框中心不同，不能统一称为真实COM。</p>
<p>因此新提取以独立候选/API保留，FastStructure.reconsider_small_cores默认False。当前CLI默认仍是173的完整版本。下一个单点问题是<b>让新原弧先通过共享刚体/跨帧约束，再为同一个定义明确的位置参考提供位移，而非首次出现就替换参考点</b>。原整机框评分照常保留，不通过给坐标加固定20px或更换评分目标守分。</p>
<p>新提取另一录像的模型CPU P95约19.27→41.79ms，侧视约231→228ms（双臂顺序运行的模型部分，未计完整GPU/解码/写出链）。它也没有解决30FPS；不拿图变好或一次算子速度作为实时证明。</p>
<h2>4. 另一个保留的负例与验证范围</h2><p>复用既有AdaptiveForceMemory、只从到期的自身位置学习短/长记忆，加入已知参考变化rebase：另一段90.57px，但侧视40.15px，因此不接默认。新API只在显式native选项下改变旧资格，默认NN数值与元数据保持；未发明新推力公式或用标签在线校正。</p>
<p>已完成：丢测分段与发布曲线机制检查、可见贡献转未知项仍保持预测、丢测不改seen/观测数量、协方差数值PSD、两个入口控制选项、实际GPU前端与固定观察回放一致。当前样本潜在加速度均未触及原3000px/s²输出上限，机制等价不扩张到该上限外的所有状态。</p>
<p>第一份机制检查因测试用1/30浮点时间、现有转移逐段四舍五入到整数微秒而差0.000048px；保留记录，改为原生整数微秒测试，并未放宽状态等价容差。实际推理有开始源码快照和结束守卫；坐标标签只在输出落盘后评分。修改仅工作区，无训练、新依赖、作者代码复制、远端上传或RAW修改，NN发布目录和两处yolov10-main不动。</p>
<h2>5. 文献与下一步的复用入口</h2><p><a href="https://stonesoup.readthedocs.io/en/v1.6/_modules/stonesoup/models/transition/linear.html">Stone Soup的Singer实现</a>用于复查条件衰减模型；本项目复用自己的已有转移函数，没有复制其实现。<a href="https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb">既有自适应跟踪资料</a>说明机动时模型适配已有研究基础，本轮修复不当成新滤波理论。</p>
<p>另查到<a href="https://arxiv.org/html/2610.31384v1">CoHelix（2026-10-22）</a>：以共面多桨的共享投影结合独立旋转状态，区分几何变化与旋转。它有初始圆环/方向/频率设置条件，未直接解决我们的自动启动和侧视退化，不能把论文RPM锁定率当0.4s轨迹精度。<a href="https://github.com/radimspetlik/helixtrack">作者HelixTrack仓库</a>提供可研究的相位/几何耦合实现；尚未编译或接入本项目，作者初始化与本机SDK依赖也需核对。后续先查本机是否已下载，按许可和独立接口复用，避免再独立重写同类跟踪器。</p>
<p><a href="final_summary.json">所有最终数值</a> · <a href="checks.json">机制/入口检查</a> · <a href="integrated/protocol.json">173实际前端协议</a> · <a href="../rim174_20261107/integrated/protocol.json">174候选对照协议</a> · <a href="../rim174_20261107/supported_rim_recovery.json">原弧候选及覆盖证据</a> · <a href="../rim174_20261107/integrated/scores.json">候选接管的负例</a></p></html>'''.replace('TABLE',''.join(table))
    (OUT/'index.html').write_text(doc,'utf-8');print('Wrote173-174 report and two figures.')
