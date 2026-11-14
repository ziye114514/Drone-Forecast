"""Make the bounded170-172 work report from saved actual outputs."""
from pathlib import Path
import os,sys,json,hashlib,html
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from score_rotor164 import score
OUT=ROOT.parent/'output/motion172_20261107'
BASE=ROOT.parent/'output/geometry170_20261106'

def load(p):return json.loads(p.read_text('utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text('utf-8').splitlines()]
def compare(a,b):
    aa=rows(a);bb=rows(b);assert len(aa)==len(bb);delta=0.
    for x,y in zip(aa,bb):
        assert x['t']==y['t'] and x['valid']==y['valid']
        assert (x['coefficient'] is None)==(y['coefficient'] is None)
        if x['coefficient'] is not None:
            for key in ('position','velocity','acceleration'):
                delta=max(delta,float(np.max(abs(np.array(x['coefficient'][key])-y['coefficient'][key]))))
    return dict(frames=len(aa),max_coefficient_delta=delta)

if __name__=='__main__':
    assert (BASE/'final172/completion.json').exists() and (BASE/'final_regional172/completion.json').exists()
    result={'source_time_horizon_s':.4,'NN_used':False,'new_RAW_FPS_measured':False,
            'historical_development_data_not_blind_test':True,'side':{},'train2':{}}
    for arm in ('legacy','fixed'):
        p=BASE/f'final172/side_{arm}.jsonl';lab=ROOT.parent/'output/dense_labels_side_110_1195/labels.json'
        result['side'][arm]=dict(fixed=score(p,lab,np.arange(110.4,112.0001,.2)),
            extension=score(p,lab,np.arange(112.2,114.0001,.2)),combined=score(p,lab,np.arange(110.4,114.0001,.2)))
        result['train2'][arm]=score(BASE/f'final_regional172/train2_{arm}.jsonl',
            ROOT.parent/'output/dense_labels_train2_40_46/labels.json',np.arange(43.2,44.4001,.2))
    result['batch_operator_full_run_parity']={a:compare(BASE/f'final_run/side_{a}.jsonl',BASE/f'batched_run/side_{a}.jsonl') for a in ('legacy','fixed')}
    result['final_side_vs_before_guard_parity']={a:compare(BASE/f'batched_run/side_{a}.jsonl',BASE/f'final172/side_{a}.jsonl') for a in ('legacy','fixed')}
    result['model_CPU_only_timing']=load(BASE/'final172/completion.json')['summary']['side']
    result['arc_operator']=load(ROOT.parent/'output/geometry171_20261107/arc_operator_check.json')
    result['mechanism']=load(OUT/'guard_mechanism.json')
    result['sources_unchanged_during_runs']=all(not load(BASE/f'{n}/completion.json')['source_changed'] for n in ('final172','final_regional172'))
    result['reviewed']=[]
    reviewed=load(BASE/'final172/reviewed.json')
    truth={'A':'用户：3桨盘可见','B':'用户：基本成线','C':'用户：4桨盘可见','D':'用户：基本成线'}
    for r in reviewed:
        result['reviewed'].append(dict(label=r['label'],t=r['t'],human=truth[r['label']],
            **{a:dict(detected=v['body'].get('detected',False),candidates=len(v['geometry'].get('rotors',[])),
                mode=v['structure'].get('mode'),view=v['structure'].get('view_evidence')) for a,v in r['arms'].items()}))
    (OUT/'final_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),'utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    r=next(r for r in reviewed if r['label']=='C')
    with np.load(BASE/'C.npz') as z:
        on=z['on'].astype(float);off=z['off'].astype(float);t=float(z['t'])
    x,y,X,Y=r['arms']['legacy']['geometry']['roi'];limit=max(1.,np.percentile((on+off)[y:Y,x:X],99.7))
    rgb=np.zeros(on.shape+(3,));rgb[:,:,0]=np.clip(np.log1p(on)/np.log1p(limit),0,1);rgb[:,:,2]=np.clip(np.log1p(off)/np.log1p(limit),0,1)
    fig,axs=plt.subplots(1,3,figsize=(15,3.8),constrained_layout=True)
    for ax,title in zip(axs,['原生 ON/OFF · 20 ms','旧路径：3候选 / 不足','修复：4桨证据 / 等待重锁']):
        ax.imshow(rgb);ax.set_xlim(x,X);ax.set_ylim(Y,y);ax.set_title(title);ax.set_xlabel('原始像素 x');ax.set_ylabel('y')
    for ax,arm in zip(axs[1:],['legacy','fixed']):
        for i,g in enumerate(r['arms'][arm]['geometry'].get('rotors',[]),1):
            color='#ffcf5c' if g.get('curvature_validated') else '#54e5c7'
            ax.add_patch(Ellipse((g['cx'],g['cy']),g['major'],g['minor'],angle=g['angle'],fill=False,ec=color,lw=1.7,ls='--'))
            ax.plot(g['cx'],g['cy'],'+',color=color,ms=7);ax.text(g['cx']+3,g['cy']-3,str(i),color='white')
    fig.suptitle(f'C：t={t:.2f}s，同一原始事件图；虚线为弧拟合外推边界，十字为估计桨心',fontsize=12)
    fig.savefig(OUT/'C_native_arc_fix.png',dpi=165);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,3.6),constrained_layout=True)
    for arm,label in [('legacy','旧路径'),('fixed','修复后')]:
        rr=result['train2'][arm]['records'];ts=[r['t'] for r in rr]
        axs[0].plot(ts,[r['current_position_error_px'] for r in rr],'o-',label=label)
        axs[1].plot(ts,[r['error_px'] for r in rr],'o-',label=label)
    for ax,title in zip(axs,['当前位置误差','0.4秒后的预测误差']):
        ax.set_title(title);ax.set_xlabel('预测起点 / s');ax.set_ylabel('px');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'position_vs_forecast.png',dpi=160);plt.close(fig)
    table=[]
    for title,scope in [('侧视原9点','fixed'),('同录像后段8/10点','extension')]:
        a=result['side']['legacy'][scope];b=result['side']['fixed'][scope]
        table.append(f'<tr><td>{title}</td><td>{a["error"]["mean"]:.2f}</td><td>{b["error"]["mean"]:.2f}</td><td>{b["available"]}/{b["eligible"]}</td></tr>')
    a=result['train2']['legacy'];b=result['train2']['fixed']
    table.append(f'<tr><td>另一录像7点</td><td>{a["error"]["mean"]:.2f}</td><td>{b["error"]["mean"]:.2f}</td><td>{b["available"]}/{b["eligible"]}</td></tr>')
    review_table=''.join(f'<tr><td>{r["label"]}</td><td>{r["human"]}</td><td>{r["legacy"]["candidates"]} / {r["legacy"]["mode"]}</td><td>{r["fixed"]["candidates"]} / {r["fixed"]["mode"]}<br>{r["fixed"]["view"]}</td></tr>' for r in result['reviewed'])
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>桨叶主线：卡点、实际修复与剩余问题</title>
<style>body{max-width:1100px;margin:36px auto;padding:0 22px;background:#f5f7fb;color:#192a40;font:17px/1.85 "Microsoft YaHei",sans-serif}h1{font-size:30px}h2{margin-top:38px}table{width:100%;border-collapse:collapse;background:white}td,th{border:1px solid #ccd6e4;padding:10px;text-align:left}img{max-width:100%;background:white}code{background:#e7edf5;padding:2px 5px}.note{padding:16px;background:#fff1cf;border-left:5px solid #cb8514}a{color:#155ca2}li{margin:8px 0}</style>
<h1>桨叶主线：先修观测，再研究推力预测</h1><p>2026-11-07 · 阶段170–172 · 实际改代码和短原生缓存回放。用户已恢复研究；11月5日的“冻结维护”不再约束当前研究。原NN发布包继续保留。</p>
<p class="note"><b>结论：</b>清晰C帧的粘连拆分已恢复4桨证据；坏位移绕回绝对位置的通路已切断；弧拟合算子加速且结果相同。但0.4秒预测没有达到30px，另一录像当前版本约102px；稳定30FPS也未达到。不会把几何修复或当前位置变准说成未来预测成功。</p>
<h2>1. 按依赖顺序处理卡点</h2><table><tr><th>卡点</th><th>这次实际做了什么</th><th>当前状态 / 下一步</th></tr>
<tr><td>粘连、侧视捷径抢先</td><td>保留原生双极性边缘支持；凹点划分原始开放弧；检验膨胀、覆盖、重叠、尺度/角度一致性；当前非扁平椭圆证据先于旧侧视判据。</td><td>C恢复4桨证据，B保持线投影。A仍有多余候选，D仍漏检；不能宣称总体四桨检出率已解决。</td></tr>
<tr><td>不可信四点把位置参考移动21px</td><td>四候选先检查同一投影平面形状一致性；首次有效几何改参考时同步移动状态参考，不伪造速度或力。</td><td>已挡住110.833s那次错误参考改写。真正侧线仍只提供无符号方向，不补当前四桨中心。</td></tr>
<tr><td>匹配失败却退回“绝对中心”</td><td>44.199983s的dx=40px撞搜索边界；禁用该位移，短时惯性仅作参考，前后状态保留交叉协方差。下一幅有效相对位移可重锁；预测参考不算新位置观测。</td><td>当前位置偏差明显下降，未来均值略变差；保留负例。还须解决旋转/形变被错当平移、以及初始桨区域中心偏置。</td></tr>
<tr><td>几何计算长尾</td><td>对同一批AMS候选一次批量算残差，种子、顺序、阈值和精修保留；65条真实弧20个接受结果全部数值一致。</td><td>算子约103.83→55.74ms；实际整段模型CPU尾延迟仍超过33.3ms。下一步复用有当前弧支持的轨迹，仅对新粘连/失锁区域重搜索。</td></tr>
<tr><td>周期有了，推力信息仍不可靠</td><td>保留现有可见贡献u与未知项d的单状态二力分工；不可见桨不补RPM，不强分电机头尾。</td><td>尚未打通。先稳定同一区域的谐波/尺度和方向，并验证它对随后净加速度有增量信息，再扩大力学分支权重。</td></tr></table>
<h2>2. 你的弧度拆分思路实际改变了什么</h2><img src="C_native_arc_fix.png" alt="同一原生C帧的旧新弧拟合对照">
<p>没有用NN框或未来事件初始化上述C结果。图是相同20ms原生ON/OFF计数；不是重建灰度。虚线完整椭圆包含外推边界，十字不是质量重心。C的当前4桨证据只进入重锁等待，没有直接跳成完整姿态/推力真值。</p>
<table><tr><th>历史人工核验帧</th><th>用户核验</th><th>旧候选数 / 模式</th><th>修复候选数 / 模式</th></tr>REVIEW_TABLE</table>
<p>当前候选数不等于真实可见桨数，更不等于电机数。全段提出四候选9→25次，只是算法输出次数；其中只有3次通过新增当前形状资格，不能当作召回率提升。相邻两凹点的快捷拆分曾把机架碎弧选成桨盘，已撤掉，保留失败结果。</p>
<h2>3. 改善当前位置，不等于改善0.4秒预测</h2><img src="position_vs_forecast.png" alt="当前位置误差与未来预测误差的分离对照">
<p>44.2s当前位置误差约51.99→31.98px，44.4s约51.49→23.95px；但44.4s未来误差33.87→55.89px。旧的错位曾抵消部分速度/加速度误差。这种抵消不能作为保留错误观测的理由，也不能反推当前推力已经估对。</p>
<table><tr><th>原固定查询</th><th>旧均值 / px</th><th>当前均值 / px</th><th>可评分 / 计划</th></tr>SCORE_TABLE</table>
<p>前两行是同一侧视录像的原段和延伸段；第三行恢复该录像既有regional-seed配方，不把初次漏配导致6/67检出当算法退化。两路共享同一GPU前端结果，固定查询、评分定义和可用性一起保留。推理阶段禁止读取坐标标签，人工框只在事后评分使用；全部仍是历史开发录像，不是新盲测或论文成绩。</p>
<h2>4. 二力模型具体卡在哪里</h2><p><code>a_image = k · u_visible + d</code>。u来自当前真正合格的周期平方与投影方向；d包含隐藏桨、重力、空气作用和当前图像模型未解释的部分。现有实现只称u为代理，并没有独立测出牛顿推力。</p>
<p>如果让d任意变化，任意k′都可用<code>d′ = d + (k − k′)u</code>解释同一运动。这是参数不可唯一辨识的问题，不能单靠“有偏差就同时修k和d”解决。只有在d的时间变化受合理约束、可见输入有可靠独立变化时，才可能学习有用的图像响应尺度。</p>
<p>侧视不需要知道头尾，也不一定需要知道具体电机编号；但仍需知道信号是否来自同一可见区域、当前合并/分离是否改变了信号尺度。可见贡献变化不一定是推力变化；线的法向还有符号与投影幅度歧义。不能仅用“周期代表桨”就把整个混合频谱当总推力。</p>
<p>下一步只做一条明确链路：<b>有当前原弧支持的区域续接 → 稳定周期来源 → 可见贡献变化 → 检查其能否比同资格置零更早反映净加速度</b>。若前一步不可靠，先不让它改变力学预测。当前后段仍有明显变向与旧速度延续误差；没有充分证据把所有未来误差都归为不可预见的新指令。</p>
<h2>5. 当前完整计算逻辑</h2><p>RAW或原生缓存 → ON/OFF计数、密度抑制 → 过去子窗发现周期、较晚子窗验证 → 周期种子触及的原生连通区域 → 原开放弧拟合/曲率拆分 → 当前4/3/2/侧线与未决资格 → 相对位移/合格几何中心 → 单一位置、速度、可见贡献尺度和未知项状态 → 同一状态外推0.1/0.2/0.4秒。</p>
<p>三桨完成几何不补隐藏转速；二桨/侧线不强制构造当前完整五圆中心；可见项丢失时把它转入未知项以免制造净力跃变。刚体几何变化和轨迹平移分开处理。NN旧主入口显式保留旧几何路径，发布目录和ZIP未改。</p>
<h2>6. 时间和证据边界</h2><p>本轮使用已保存的原始EventCD事件包，运行实际CUDA周期前端、几何、匹配和观测器；没有重跑RAW解码测速。侧视133包：111包当前区域存在，115包有效预测（含允许的短时延续），两者不是同一指标。当前模型部分CPU中位约20.33ms、P95约257.20ms，未包含整个GPU/解码链路；绝不能据中位数声称稳定30Hz。</p>
<p>完整输出、固定查询、源码快照和失败尝试均保留。数值同一性检查只用于加速不改结果、坏观测不再回灌等明确机制，不用大规模扫参数刷这两段录像。</p>
<h2>7. 参考与复用范围</h2><ul>
<li><a href="https://arxiv.org/abs/2008.00997">凹点检测与重叠物体椭圆分割</a>：支持“曲率异常→原弧分组→椭圆”的方法依据；本项目沿用已有147实现再改分组，没有复制作者实现。</li>
<li><a href="https://github.com/Li-Zhaoxi/AAMED">AAMED官方项目</a>：弧邻接分组值得复用；这次没有集成其二进制或声称完整复现。</li>
<li><a href="https://github.com/ialzugaray/haste">HASTE官方项目</a>：事件多假设特征跟踪；仍需初始化/丢失管理。未集成，不把它当成桨盘检测器。</li>
<li><a href="https://authors.library.caltech.edu/records/enrkd-zam24">Stochastic cloning原始论文记录</a>：已有相对位移观测器保留跨时刻相关性，本轮修复失败帧的参考续接，并非提出新的卡尔曼方法。</li></ul>
<p><a href="final_summary.json">本轮全部数值</a> · <a href="guard_mechanism.json">坏位移/续接机制检查</a> · <a href="../geometry170_20261106/final172/protocol.json">侧视运行协议</a> · <a href="../geometry170_20261106/final_regional172/protocol.json">另一录像协议</a> · <a href="../geometry171_20261107/arc_operator_check.json">65条弧同一性与算子时间</a></p>
</html>'''.replace('REVIEW_TABLE',review_table).replace('SCORE_TABLE',''.join(table))
    gap_path=OUT/'error_gap_breakdown.json'
    if gap_path.exists():
        gap=load(gap_path)['cases'];a=gap['side']['summary'];b=gap['train2']['summary']
        appendix=f'''<h2>3.1 为什么侧视约39px，另一录像却约102px？</h2>
<p>“侧视”是片段名称，不是误差大小的充分原因。另一录像也没有稳定建立完整四桨模板；同一录像43.8s可低至24.60px，43.6/44.0/44.2s则分别140.51/193.72/171.41px。</p>
<table><tr><th>相同定义</th><th>侧视17起点</th><th>另一录像7起点</th></tr>
<tr><td>当前位置误差均值 / px</td><td>{a['mean_position_error_px']:.2f}</td><td>{b['mean_position_error_px']:.2f}</td></tr>
<tr><td>目标人工框对角线均值 / px</td><td>{a['mean_box_diagonal_px']:.2f}</td><td>{b['mean_box_diagonal_px']:.2f}</td></tr>
<tr><td>0.4s误差 / 当前框对角线，逐起点平均</td><td>{100*a['mean_error_over_box_diagonal']:.1f}%</td><td>{100*b['mean_error_over_box_diagonal']:.1f}%</td></tr>
<tr><td>同一估计状态仅p+h·v的误差均值 / px</td><td>{a['mean_position_velocity_only_error_px']:.2f}</td><td>{b['mean_position_velocity_only_error_px']:.2f}</td></tr></table>
<p>最后一行只是冻结当前状态后的外推项诊断，不是重跑另一套基线；移除运动项并非独立因果贡献实验。标量误差不可直接相加。尺度归一化后仍有差距，目标大小不能解释全部问题。</p>
<img src="why_error_spikes.png" alt="同一录像三个起点的实际参考与预测轨迹">
<p>43.6s：当前估计竖直速度约−57px/s，未来0.4s参考平均约+278px/s，转弯后分叉。44.0s：估计竖直速度+513px/s，未来平均仅+128px/s，继续外推旧速度并附加旧加速度导致过冲。44.2s：当前p/v外推误差53.14px，未知/净加速度延续又贡献约120.41px的位移向量，最终误差171.41px，出现反向过冲；此时当前可见周期输入未获资格。这一处不能都推给未来指令。</p>
<p>这些查询的有效尺度k都没有学成，当前周期项的独立响应仍小；较大的外推主要来自桨叶区域运动和未知/净加速度状态。此处区分了变向、定位偏差和模型过冲，没有声称已经辨识真实推力，也没有证明参数不存在过拟合。两段均是历史开发数据。</p>
<p><a href="error_gap_breakdown.json">每起点精确向量分解与归一化数值</a></p>'''
        doc=doc.replace('<h2>4. 二力模型具体卡在哪里</h2>',appendix+'<h2>4. 二力模型具体卡在哪里</h2>')
    (OUT/'index.html').write_text(doc,'utf-8')
    print(json.dumps(dict(scores={c:{a:(r.get('combined',r)['error']) for a,r in result[c].items()} for c in ('side','train2')},
        operator_parity=result['batch_operator_full_run_parity'],side_parity=result['final_side_vs_before_guard_parity']),indent=2))
