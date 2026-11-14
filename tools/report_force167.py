"""Explain the actual rotor/two-force blockers with complete saved results."""
from pathlib import Path
import sys,json,copy,hashlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from run_force167 import OUT
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams['axes.unicode_minus']=False
NAMES={'clear':'NN仅辅助观测：清晰段','side':'NN仅辅助观测：侧视','train2':'NN仅辅助观测：另一录像',
    'pure_side':'纯桨叶旧状态：侧视','pure_best_side':'纯桨叶较强状态：侧视','pure_best_side_2px':'纯桨叶相对位移先验一致：侧视','pure_train2':'纯桨叶：另一录像'}


if __name__=='__main__':
    scores=json.loads((OUT/'scores.json').read_text('utf-8'));summary=json.loads((OUT/'summary.json').read_text('utf-8'))
    rows_by_clip={c:[json.loads(s) for s in (OUT/f'{c}_full_learned.jsonl').read_text('utf-8').splitlines()] for c in NAMES}
    audit={}
    for clip,rows in rows_by_clip.items():
        fits=[r['visible_force_dynamics']['gain_fit'] for r in rows if r['visible_force_dynamics']['gain_fit'].get('candidate_gain_px_s2') is not None]
        diff=[]
        zero=[json.loads(s) for s in (OUT/f'{clip}_zero.jsonl').read_text('utf-8').splitlines()]
        for r,z in zip(rows,zero):
            if r['valid'] and r['physical']:diff.append(float(np.linalg.norm(np.asarray(r['physical']['0.4'])-z['physical']['0.4'])))
        reasons=Counter(f['reason'] for f in fits)
        audit[clip]=dict(solved=len(fits),fit_reason_counts=dict(reasons),
            below_2sigma=sum(f['gain_signal_to_noise']<2 for f in fits),
            numeric_force_changes_mean_px=float(np.mean(diff)) if diff else 0.,numeric_force_changes_max_px=max(diff,default=0.),
            updated_modes=dict(Counter(r['structure']['mode'] for r in rows if r['visible_force_dynamics']['gain_updated'])),
            true_RPM_or_Newton_force_established=False)
    (OUT/'force_audit.json').write_text(json.dumps(audit,indent=2),'utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for clip,ax in zip(['side','clear'],axes[0]):
        rows=rows_by_clip[clip];ax.plot([r['t'] for r in rows],[r['visible_force_dynamics']['effective_gain'] for r in rows],c='#a4671b',label='当前来源有效增益')
        updates=[r for r in rows if r['visible_force_dynamics']['gain_updated']]
        ax.scatter([r['t'] for r in updates],[r['visible_force_dynamics']['effective_gain'] for r in updates],c='#257a57',s=28,label='新合格拟合更新')
        ax.set_title(NAMES[clip]);ax.set_xlabel('源时刻 / s');ax.set_ylabel('图像有效增益（非真实推力系数）');ax.grid(alpha=.2);ax.legend(fontsize=9)
    for clip,ax in zip(['side','pure_best_side_2px'],axes[1]):
        for arm,color,label in [('zero','#3976bc','同模型，无桨叶趋势'),('full_learned','#ae6b24','完整桨叶趋势＋尺度学习')]:
            records=[r for r in scores[clip]['arms'][arm]['records'] if r['valid']]
            ax.plot([r['query_t'] for r in records],[r['error_px'] for r in records],'.-',c=color,label=label)
        ax.set_title(NAMES[clip]);ax.set_xlabel('查询时刻 / s');ax.set_ylabel('0.4s终点误差 / px');ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.savefig(OUT/'force_results.png',dpi=150);plt.close(fig)
    table=''
    for clip in ['clear','side','train2','pure_best_side_2px','pure_train2']:
        arms=scores[clip]['arms'];n=arms['zero'];values=''.join(f"<td>{arms[k]['error']['mean']:.3f}</td>" for k in ['zero','full_fixed','full_learned','old_full'])
        table+=f"<tr><th>{NAMES[clip]}</th><td>{n['available']}/{n['eligible']}</td>{values}</tr>"
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>阶段167：桨叶二力模型优化</title>
<style>body{{font:17px/1.8 system-ui,"Microsoft YaHei";max-width:1120px;margin:35px auto;padding:0 20px;color:#263246;background:#fafbfd}}h1,h2{{line-height:1.4}}table{{width:100%;border-collapse:collapse;font-size:15px}}th,td{{border:1px solid #d7deea;padding:8px;text-align:right}}th:first-child{{text-align:left}}pre{{padding:18px;background:#e9f1f9;white-space:pre-wrap}}img{{max-width:100%}}.note{{background:#fff1ce;padding:18px;border-left:5px solid #ad7919}}a{{color:#245e9e}}</style>
<h1>这轮优化的是桨叶／二力模型，NN只提供冻结观测</h1>
<p>2026-11-05 · 阶段167 · 保留用户的ON/OFF桨叶检测、五圆及四/三/二/侧视分工、当前可观测贡献＋未知合力。没有改NN、没有训练轨迹网络、没有混合旧NN完整预测来制造收益。</p>
<div class="note">修复了参考幅值、时间覆盖和尺度单位三个实现问题。NN观测辅助下新增14次侧视／2次清晰有效图像增益更新；纯桨叶仍0次。精度尚未稳定提升：清晰同模型27.965→27.873px；侧视43.098→43.850px；较强纯桨叶侧视35.635→35.643px。30px／30FPS目标未完成，不升级默认。</div>
<h2>二力模型究竟卡在哪里</h2>
<pre>当前二维模型：a = k u_visible + d
u_visible：当前桨叶频率平方与像面法向构成的代理。
d：隐藏桨力、重力、风、阻力及其他图像动力学误差。

只看当前运动，k增大时，d可以减去相同贡献：
    (k + Δk) u_visible + (d − Δk u_visible) = a。
因此“能追上当前位置”不等于“分清了推力与未知力”。</pre>
<p><b>最根本的困难是可辨识性。</b>需要连续的已知输入变化，并对未知项变化作合理约束。尤其隐藏桨也会随控制器快速变化：可以把其推力计入d，但不能据此认为整个d一定慢变。常未知项假设可能在变向时失败；学出的图像有效响应不能自动叫真实推力系数。</p>
<p><b>图像投影和频率归属又增加歧义。</b>侧视混合频率可能来自不同桨或不同谐波；换一组主导像素也会改变读数。它可以用作经过检验的预测代理，但不是测出了每个桨的RPM。两桨连线不足以唯一恢复完整三维平面，法线符号目前仍用像面向上先验。</p>
<h2>本轮实际改了什么</h2><ol>
<li><b>修复参考减法压小姿态作用的问题。</b>当前已知/未知分账仍防止可见数量切换制造假加速；未来法线转动改作用于当前可见的正幅值。侧视只保留一个混合代理，不乘4、不补隐藏RPM。</li>
<li><b>未来分别处理已知贡献变化和未知基线。</b>当前净加速度保持；只把已观测来源的幅值/法向趋势积分进未来，未知基线不跟机身旋转。不再仅把两项相加后使用同一净加速度外推。</li>
<li><b>复用已有积分速度辨识。</b>以v=v0+k∫u dt+d·t拟合，先消除初速与局部常未知项；增益学习用最多3s连续同来源历史，趋势仍.35s。只排除时间早于已知输入覆盖的速度midpoint，不臆造过去输入。</li>
<li><b>修复单位依赖的k上限。</b>旧固定k&lt;5000会因输入任意归一化而改变接受结果。新候选复用仓库已有的“3000px/s²／观测输入激励RMS”上界，仍检查正值、2倍标准误和速度残差。更换输入单位后，等效动力学接受结果保持一致。旧fit_gain调用默认5000不变。</li>
<li><b>来源分开学尺度。</b>可识别可见和、全局池化、局部一阶模板不互借有效系数。纯桨叶记录缺dt时，仅在相邻当前中心位移与保存dx/dy完全一致的情况下恢复dt，避免跨参考变化造速度。</li></ol>
<pre>新外推的条件模型：
a(s) = a_current·exp(−s/.35)
       + k Σ A_i [exp((r_i + jω)s) − 1] n_current
其中A_i只含当前可见贡献；r_i是其当前合格幅值趋势，ω是无头尾法向趋势。
当前d = a_current − k Σ A_i n_current。
稳态、r_i=0且ω=0时，增量为0；隐藏力没有被补进已知项。
它是条件图像预测先验，不是已知未来控制指令或真实牛顿力模型。</pre>
<h2>对照结果：保留所有预定起点</h2>
<p>源时刻未来0.4s终点均值px。NN辅助各组只共享原有位置/速度/当前净加速度观测；纯桨叶各组共享原有纯桨叶状态。此处未优化NN本身。旧完整入口另列，避免只赢弱控制就声称超越旧系统。</p>
<table><tr><th>输入路线</th><th>可评分/预定</th><th>同模型无桨叶趋势</th><th>固定尺度桨叶模型</th><th>学习尺度桨叶模型</th><th>旧完整入口</th></tr>{table}</table>
<p>只开法向趋势时，较强纯桨叶侧视为35.606px，相对35.635px约0.029px小幅改善；加入当前周期幅值趋势后为35.643px。两个候选都保留，不将这个小差异当独立统计结论，也不隐藏周期项的退化。另一纯桨叶旧状态对照38.647→38.689px。</p>
<img src="force_results.png" alt="实际图像增益更新与全部查询误差">
<h2>更新成功了，为什么预测还没有明显变好</h2>
<p>14次侧视更新全部来自局部混合代理（13次axis_only、1次edge_on）；清晰2次也来自混合代理。并没有学出四个独立电机的物理系数。清晰更新出现在固定评分起点之后，故没有改变那6个起点的学习版结果。</p>
<p>尺度增大让原来很弱的桨叶趋势开始影响预测，但过去短窗的相关性不保证未来0.4s仍成立。侧视均值43.098→43.850px、P95 79.303→76.205px：长尾略好，整体均值更差，不能宣称稳定提升。不能再简单通过放大k解决。</p>
<p>连续来源片段仍短：既有NN辅助侧视最长约0.97s、清晰0.60s；较强纯桨叶约0.40s。之前的输入激励和当前速度噪声不足以可靠分开k与d；纯桨叶仍没有合格尺度更新。最大3s学习窗不等于已经获得3s连续证据。最后另列纯相对位移观测器原有2px位移先验的一致性对照，避免误用两份6px绝对位置误差；仍0次更新、预测相同。2px也只是现有工程先验，不能把拒绝学习等同于原始录像绝无信息。</p>
<h2>下一步：继续攻自己的部分</h2><ol>
<li>把侧视的局部周期支持按空间连续区域和谐波假设续接；主导源改变时明确切换，不把换来源当转速变化。它直接针对短片段和混合读数，不能靠增加NN计算掩盖。</li>
<li>分别检验4/3/2/侧线的有效响应。隐藏桨与可见桨可能共同加速，需将其未观测贡献作为有不确定性的未知项；不能将“d局部常值”强套所有变向。有效耦合可按可见性条件学习，不能叫测得隐藏推力。</li>
<li>使用已到期的自身速度，检验新尺度对下一段速度的预测是否成立，再赋予其持续信用；区别“拟合过去成功”和“能预测后续”。这检验桨叶动力学参数，不是给完整NN预测再加万能融合门控。</li>
<li>待上述输入稳定后再接RAW实时可选入口。当前只是冻结输入原型，尚未接默认实时链路，不称30FPS；前端仍保留166约44–46ms的已知成本。</li></ol>
<h2>研究与实现边界</h2><p>增量力学借鉴<a href="https://arxiv.org/abs/1701.07254">INDI</a>及<a href="https://www.ezratal.net/files/CDC18_1876.pdf">作者的加速度/推力增量方法</a>，作者方法有机载测量和执行器条件；这里是外部事件图像预测，不能把控制论文当作本项目效果证明。积分速度拟合、无头尾法向等复用仓库已有代码，未复制新论文代码。可见性分工、未知分账与观测机制仍是项目重点，具体创新是否有效需由其消融收益支持。</p>
<p>机制检查通过：稳态不假动、参考变化不改变实际可见幅值的未来作用、正频率趋势、h=0不改当前点、正可辨识例k=700、常输入拒绝、输入单位改变时等效700/7000均接受。NN辅助各真实输入前缀重放一致，较强纯桨叶也做前缀检查。各轮主回放先存源码快照并检查结束哈希；补充纯桨叶probe记录源码哈希和前缀结果。标签在推理/控制输出落盘后读取。</p>
<p>这些数据是历史开发集，规则过拟合风险仍在。本轮未重读RAW、未改保护目录/权重、未训练、未上传、未替换旧默认或公开ZIP。</p>
<p><a href="scores.json">完整逐点评分及归因对照</a> · <a href="summary.json">输入/源码/资格统计</a> · <a href="force_audit.json">增益与推力数值作用审计</a> · <a href="../../../event_drone_forecast/docs/FORCE167_PROTOCOL_20261105.md">机制与各次修复协议</a> · <a href="../covered_gain/scores.json">固定5000上限的旧对照</a></p></html>'''
    (OUT/'index.html').write_text(page,'utf-8')
    parent=OUT.parent
    (parent/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=source_gain/index.html"><a href="source_gain/index.html">阶段167最终图文报告</a>','utf-8')
    (parent/'FAILED_ATTEMPT.json').write_text(json.dumps(dict(scope='Only the initial root source_snapshot and side outputs',status='aborted_numpy_int64_summary_serialization',scored=False,valid_final_folder='source_gain',earlier_complete_folder='final'),indent=2),'utf-8')
    print('Saved:',OUT/'index.html')
