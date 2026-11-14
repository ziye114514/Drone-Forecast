"""Report one root's completed actual-RAW implementation, not a new experiment."""
from pathlib import Path
import sys,json,html,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
from physical_publication import evaluate
from extend_inverse321 import references
from compare_final_pipeline import reference_at


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--memory',action='store_true');args=parser.parse_args()
    current_id='517' if args.memory else '512'
    base=ROOT.parent;actual=base/('output/raw_bg_memory517_20261113' if args.memory else 'output/raw_bg_selection512_20261113')
    result=json.loads((actual/'results.json').read_text('utf8'))
    assert result['all6_actual_RAW'] and result['original102_preserved'] and result['source_unchanged_during_batch']
    report=base/('output/background_root_final_20261113' if args.memory else 'output/background_root_20261113');report.mkdir(exist_ok=False)
    ns=['13','15','16','11','3','4'];before=result['pooled']['baseline487'];after=result['pooled']['background']
    assert before['n']==after['n']==102
    selection=json.loads((base/('output/background_memory516_20261113/results.json' if args.memory else 'output/background_evidence511_20261112/results.json')).read_text('utf8'))
    fig,ax=plt.subplots(figsize=(11,4.8));x=np.arange(6);width=.35
    a=[result['cases'][n]['baseline487']['original_available']['mean'] for n in ns]
    b=[result['cases'][n]['background']['original_available']['mean'] for n in ns]
    p=ax.bar(x-width/2,a,width,label='原桨叶入口487',color='#8b9baa')
    q=ax.bar(x+width/2,b,width,label=f'当前背景证据入口{current_id}',color='#167e9e')
    ax.bar_label(p,fmt='%.2f',fontsize=9);ax.bar_label(q,fmt='%.2f',fontsize=9)
    ax.axhline(30.5,color='#be6c00',ls='--',lw=1,label='阶段均值门槛30.5px')
    ax.set(xticks=x,xticklabels=[f'录像{n}\n{result["cases"][n]["background"]["original_available"]["n"]}起点' for n in ns],ylabel='0.4秒均FDE（px）',ylim=(0,74),title='同六段实际RAW、原102起点：没有换录像或删除困难点')
    ax.legend(loc='upper left',frameon=False);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(report/'per_source.png',dpi=160);plt.close(fig)

    # Example chosen in495 before candidate implementation, not best-case mining.
    n='4';t0=12.;oldrows=[json.loads(s) for s in (base/f'output/raw_local_contrast487_20261112/{n}_local_contrast/predictions.jsonl').read_text('utf8').splitlines()]
    newrows=[json.loads(s) for s in (actual/f'recording{n}/predictions.jsonl').read_text('utf8').splitlines()]
    old=[r for r in oldrows if r['t']<=t0+1e-9][-1];new=[r for r in newrows if r['t']<=t0+1e-9][-1]
    ref=json.loads((base/'output/reference323_20261109/references.json').read_text('utf8'))
    refs=sorted(ref['records'].values(),key=lambda r:r['t']);times=np.round(np.arange(11.6,12.6001,.2),1)
    targetpath=np.array([reference_at(refs,float(t)) for t in times]);hh=np.linspace(0,.4,41)
    aa=np.array(list(evaluate(old,(hh+t0-old['t']).tolist()).values()))
    bb=np.array(list(evaluate(new,(hh+t0-new['t']).tolist()).values()))
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(*targetpath.T,'o-',color='#626a73',label='弱桨叶轮廓参考（非重心真值）')
    ax.plot(*aa.T,color='#b44e4e',label='原入口：12.0发出的未来0.4秒')
    ax.plot(*bb.T,color='#167e9e',label='当前：12.0发出的未来0.4秒')
    ax.scatter(*reference_at(refs,t0+.4),marker='*',s=130,color='black',zorder=4,label='12.4秒参考')
    ax.scatter(*aa[-1],marker='x',s=70,color='#b44e4e');ax.scatter(*bb[-1],marker='x',s=70,color='#167e9e')
    ax.set(xlabel='图像x（px）',ylabel='图像y（px）',title='录像4@12.0秒：实现前已选定的早期误差案例')
    ax.invert_yaxis();ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=8,loc='best');fig.tight_layout()
    fig.savefig(report/'example4.png',dpi=170);plt.close(fig)

    rows=''.join(f'<tr><td>{n}</td><td>{result["cases"][n]["background"]["original_available"]["n"]}</td><td>{a[i]:.6f}</td><td>{b[i]:.6f}</td><td>{b[i]-a[i]:+.6f}</td></tr>' for i,n in enumerate(ns))
    gain=before['mean']-after['mean'];percent=100*gain/before['mean']
    gate_text=('30px开发均值门槛形式上达到，但余量仅'+f'{30-after["mean"]:.6f}px，不能称为稳定30px。' if after['mean']<=30 else '总目标30px尚未达到。')
    memory_text=('进一步让补偿资格按原力学状态0.35秒时间尺度衰减：取当前背景证据权重与上一资格衰减值的较大者，避免背景刚停下就忘掉历史坐标污染。它不是把相关的过去观测反复相乘，也不声称为校准后验。512的30.138842px因此变为517的29.997218px；这是同一根因的后续修复，无数值搜索。' if args.memory else '本版尚未引入力学状态时间尺度下的补偿资格记忆。')
    blind_text='保留新录像仍未解码，不宣称跨新录像泛化。'
    blind_path=base/'output/frozen_external518_20261113/results.json'
    if blind_path.exists():
        blind=json.loads(blind_path.read_text('utf8'));blind_text=html.escape(blind['plain_language_result'])
    elif (base/'output/frozen_external518_20261113/protocol.json').exists():
        blind_text='已按此前约定执行固定10–16秒的新录像冻结验证；未依据其结果调参，独立验收状态以盲测结果文件为准。'
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>先解决一个根因：背景参考与桨叶力学状态</title>
    <style>body{{max-width:1100px;margin:36px auto;padding:0 24px;font:17px/1.85 system-ui,sans-serif;color:#172a3b;background:#f5f7fa}}article{{background:white;border-radius:14px;padding:26px;margin:22px 0}}h1{{font-size:30px}}h2{{font-size:23px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:9px 14px;text-align:right;border-bottom:1px solid #dde3ea}}td:first-child,th:first-child{{text-align:left}}.metric{{font-size:29px;color:#126d84;font-weight:700}}code,pre{{background:#eaf0f5;padding:3px 7px;border-radius:4px;white-space:pre-wrap}}.note{{color:#765109}}a{{color:#126d84}}</style>
    <h1>沿一个根因深入：背景运动混入桨叶速度与不可测力</h1>
    <article><p class="metric">{before['mean']:.6f} → {after['mean']:.6f}px</p>
    <p>同六段实际RAW，原102个起点，预测源时间之后0.4秒的均FDE下降{gain:.6f}px（{percent:.2f}%）。中位{before['median']:.6f}→{after['median']:.6f}px；30.5px内{before['within30p5']}→{after['within30p5']}/102。</p>
    <p>阶段均值30.5px已达到；{gate_text}不是每段或每次预测都低于30.5px。{blind_text}</p></article>
    <article><h2>先查哪里，再改哪里</h2><p>先检查旧487的原102点，分开当前位置、当前速度、0.4秒延续和之后新变化。弱参考只能用于诊断，不能据此判定未来操纵原本可预测。录像4@12秒的早期误差被选作排查案例：原生桨叶掩码已基本排除电线，单纯继续修轮廓没有证据；画面背景却存在共同移动，旧力学状态会把这项表观加速度归给目标。</p>
    <p>随后只攻背景参考这一环：旧背景仿射在录像13的165个合格区间里有161个把目标位置放在跟踪点凸包之外，局部剪切外推会放大错误。因此复用现有背景LK与一致性检查，只估计共同二维平移。</p>
    <img src="../native_root496_20261112/evidence.png" alt="原始ON/OFF、现有桨叶支持与严格周期像素对照"><p>左：原ON/OFF；中：已有桨叶支持；右：严格周期像素。图中背景并非全进了检测，但它的运动仍影响图像坐标中的目标速度。周期像素稀疏的问题尚未解决。</p></article>
    <article><h2>改的是参考坐标，保留你的主线</h2><pre>RAW原生ON/OFF
├─ 原桨叶滤波 → 弧/椭圆/线 → 可见数量分工 → 原周期代理
│    → 可见贡献＋不可测合力 → 原有未来0.4秒力学预测
└─ 目标区域之外的背景事件 → 共同平移与协方差
     → 背景不动/背景共同运动两种坐标假设的当前证据
     → 对同一桨叶预测作坐标补偿与加权输出</pre>
    <p>图像位置 p<sub>I</sub> = 相对背景位置 p<sub>R</sub> + 背景参考位移 C。先在观测端减去C的增量，再估计原二力状态；预测后加回背景参考的未来位移。隐藏桨叶、重力、风与阻力仍由不可测项负责。背景不检测无人机，不制造桨心、RPM或推力。</p>
    <p>相邻位移共用一幅事件图，噪声相关，不能逐段当独立误差累加。新增背景端点误差状态：单段底噪4px²，相邻协方差−2px²；连续三段的端点底噪仍4px²，错误独立累加会变12px²。原前景噪声先保底再加背景噪声，避免把原始LK小方差混成过度自信。</p>
    <p>背景证据弱时自动靠近原入口。权重来自当前和过去三个连续背景观测区间的静止/共同速度与加速度似然；复用旧0.12秒支持窗、1000px/s及1000px/s²先验，先验赔率1，不搜索系数。两个状态使用同一桨叶前端，彼此相关；它们是参考坐标假设，不能当两路独立测量相加提高置信度。</p></article>
    <article><h2>同一根因的时间记忆</h2><p>{memory_text}</p></article>
    <article><h2>同源实际RAW结果</h2><img src="per_source.png" alt="六录像均误差逐段对照"><table><tr><th>录像</th><th>起点</th><th>487原入口px</th><th>512当前px</th><th>变化px</th></tr>{rows}</table>
    <p>13、15、16几乎保持原值；11改善不足0.001px，按噪声级处理。收益主要来自3和4，不能说六段都有实质提升。规则不读取录像编号、坐标或人工误差，六段用同一代码和原模型。</p>
    <img src="example4.png" alt="同一起点0.4秒预测路径对比"><p>示例只展示一个预先选定案例，验收采用完整102点；没有把未来新操纵的失效预测排除。</p></article>
    <article><h2>哪些检查必要，哪些没有继续做</h2><table><tr><th>同根因环节</th><th>原102均FDEpx</th></tr>
    <tr><td>487原桨叶入口</td><td>31.471473</td></tr><tr><td>500完整仿射，仍会无支持外推（不采用）</td><td>39.009869</td></tr>
    <tr><td>504共同平移</td><td>30.782700</td></tr><tr><td>506d/509保留共享背景端点相关性</td><td>30.600825</td></tr>
    <tr><td>同协方差、去掉背景位移均值的消融</td><td>32.372458</td></tr><tr><td>511回放 / 512实际RAW，背景证据选择</td><td>{after['mean']:.6f}</td></tr></table>
    <p>共同位移补偿优于只加噪声，说明收益并非仅靠调低信任度。回放写完六段预测才读取参考；{current_id}实际RAW的每段181包、零跳包，原前端事件数/桨叶区域/位移/有效性与回放相同，预测最大差异{max(result['replay_parity'].values()):.3g}px。完整批次源码哈希保持不变。</p>
    <p>没有重训、门槛搜索、录像专属补丁、NN替换、SDK/RAW/权重改动或FPS优化。本次速度日志只是离线运行附带数据，不是30Hz验收。旧基线、失败尝试及单段消融均保留。</p></article>
    <article><h2>这个根因尚未完全解决的边界</h2><p>共同平移只是最低维的背景参考；强透视、旋转、景深差、少量纹理与背景间断时仍不充分。当前权重是工程先验下的概率，不是真实相机运动置信度。不能为了剩余0.139px扫权重或时间常数。下一步仍先诊断这些失效是否主导剩余误差，只有背景事件提供空间支持时才考虑更复杂参考模型。</p>
    <p>另一处尚未解决的硬约束是“频率代理到有效推力”的可辨识性：本次没有增加逐桨周期观测，未获得真实机械RPM/牛顿推力或在线增益稳定学习。二力结构保留，隐藏桨推力归未知项；单靠位置偏差不能唯一分离隐藏推力、风与质量。</p>
    <p class="note">六段已反复参与开发选择；训练排除它们也不等于独立验证。参考是事件轮廓中心，非真实重心/姿态/飞控真值。本次是源时间离线预测精度，尚未包含墙钟延迟。</p></article>
    <article><h2>相关已有工作与复用范围</h2><p><a href="https://arxiv.org/abs/1904.01293">Event-Based Motion Segmentation by Motion Compensation</a>研究分开背景和独立目标运动；<a href="https://arxiv.org/abs/2103.05903">FAST-Dynamic-Vision</a>利用自运动补偿与额外深度信息追踪动态目标。本项目未复现其完整系统，也没有它们的深度/相机姿态输入，复用的是本机已有OpenCV LK与运动一致性代码。本轮工程修复不单独宣称新算法或前沿性能。</p>
    <p><a href="../{actual.name}/results.json">实际RAW完整结果</a> · <a href="../{actual.name}/protocol.json">冻结协议与源码哈希</a> · <a href="../background_evidence511_20261112/results.json">证据权重回放</a></p></article></html>'''
    (report/'index.html').write_text(text,'utf8')
    (report/'summary.json').write_text(json.dumps(dict(before=before,after=after,gain_px=gain,gain_percent=percent,
        actual_RAW=True,original102=True,new_video_validated=False,target30p5_met=after['mean']<=30.5,
        target30_met=after['mean']<=30,report_script=str(Path(__file__))),indent=2),'utf8')
    print(json.dumps(dict(report=str(report/'index.html'),before=before,after=after,gain_px=gain),indent=2))
