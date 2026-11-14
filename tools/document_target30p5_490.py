"""Consolidate the completed accuracy batch, preserving blind-test boundary."""
from pathlib import Path
from datetime import datetime
import json,html,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
NS=('13','15','16','11','3','4')

def load(folder):return json.loads((WORK/'output'/folder/'results.json').read_text('utf8'))

if __name__=='__main__':
    out=WORK/'output/target30p5_20261112';out.mkdir(exist_ok=True)
    specs=[('473','raw_mean473_20261111','joint_only','run_rotor_joint_only_candidate.py','实际RAW'),
        ('478','raw_symmetric478_20261112','symmetric','run_rotor_symmetric_candidate.py','实际RAW'),
        ('480','raw_supported_plane480_20261112','supported_plane','run_rotor_supported_plane_candidate.py','实际RAW'),
        ('484','raw_joint_cost484_20261112','joint_cost','run_rotor_symmetric_surface_candidate.py','实际RAW'),
        ('485','raw_complete_cost485_20261112','complete_cost','run_rotor_complete_surface_candidate.py','实际RAW'),
        ('486','shared_axis486b_20261112','shared',None,'冻结观测回放'),
        ('487','raw_local_contrast487_20261112','local_contrast','run_rotor_local_contrast_candidate.py','实际RAW'),
        ('488','raw_contrast_symmetric488_20261112','contrast_symmetric','run_rotor_contrast_symmetric_candidate.py','实际RAW')]
    plane_folder='raw_contrast_plane492_20261112'
    if (WORK/'output'/plane_folder/'results.json').exists():
        q=load(plane_folder)
        if q['pooled']['contrast_plane']['n']==102 and not q['missing_original']['contrast_plane']:
            specs.append(('492',plane_folder,'contrast_plane','run_rotor_contrast_plane_candidate.py','实际RAW'))
    trials=[]
    for stage,folder,arm,entry,kind in specs:
        d=load(folder);p=d['pooled'][arm];assert p['n']==102
        assert not d.get('missing_original',{}).get(arm,[])
        trials.append(dict(stage=stage,folder=folder,arm=arm,entry=entry,kind=kind,summary=p,data=d))
    trainfolder='contrast_training489_20261112'
    if (WORK/'output'/trainfolder/'results.json').exists():
        d=load(trainfolder);assert d['pooled']['matched']['n']==102
        trials.append(dict(stage='489',folder=trainfolder,arm='matched',entry='run_rotor_local_contrast_candidate.py',
            kind='实际新前端＋重校准回放',summary=d['pooled']['matched'],data=d))
    base=trials[0]
    # A changed model is not promoted here without its RAW factory reproduction.
    best=min((q for q in trials if q['stage'] not in ('486','489')),key=lambda q:q['summary']['mean'])
    orig=load('reference429_20261110');scores={};records={};delta=[];improved=0
    for n in NS:
        ids={r['t'] for r in orig['cases'][n]['control410']['records'] if r['valid']}
        aa={r['t']:r for r in base['data']['cases'][n][base['arm']]['records'] if r['valid'] and r['t'] in ids}
        bb={r['t']:r for r in best['data']['cases'][n][best['arm']]['records'] if r['valid'] and r['t'] in ids}
        assert aa.keys()==bb.keys()==ids
        scores[n]=dict(n=len(ids),baseline_mean=float(np.mean([r['error_px'] for r in aa.values()])),
            candidate_mean=float(np.mean([r['error_px'] for r in bb.values()])))
        scores[n]['delta']=scores[n]['candidate_mean']-scores[n]['baseline_mean']
        improved+=scores[n]['delta']<0
        records[n]=[dict(t=t,baseline=aa[t]['error_px'],candidate=bb[t]['error_px']) for t in sorted(ids)]
        delta.extend(bb[t]['error_px']-aa[t]['error_px'] for t in ids)
    current=best['summary'];macro=float(np.mean([s['delta'] for s in scores.values()]));gain=base['summary']['mean']-current['mean']
    within=sum(r['candidate']<=30.5 for rr in records.values() for r in rr)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(10.5,4.7),layout='constrained');x=np.arange(6)
    ax.bar(x-.19,[scores[n]['baseline_mean'] for n in NS],.38,label='上一轮473')
    ax.bar(x+.19,[scores[n]['candidate_mean'] for n in NS],.38,label=f'当前{best["stage"]}')
    ax.axhline(30.5,color='#777',linestyle='--',label='30.5px阶段目标');ax.set_xticks(x,NS)
    ax.set_xlabel('已暴露开发录像');ax.set_ylabel('原起点0.4秒平均误差 / px');ax.legend()
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
    for ax,n in zip(axes.flat,NS):
        rr=records[n];tt=[r['t'] for r in rr]
        ax.plot(tt,[r['baseline'] for r in rr],'.-',label='473');ax.plot(tt,[r['candidate'] for r in rr],'.-',label=best['stage'])
        ax.axhline(30.5,color='#aaa',linestyle='--');ax.set_title(f'录像{n}，原{len(rr)}起点全部保留')
        ax.set_xlabel('发布起点 / s');ax.set_ylabel('0.4秒误差 / px');ax.legend()
    fig.savefig(out/'all_origins.png',dpi=150);plt.close(fig)
    rows='\n'.join(f'|{n}|{scores[n]["n"]}|{scores[n]["baseline_mean"]:.3f}|{scores[n]["candidate_mean"]:.3f}|{scores[n]["delta"]:+.3f}|' for n in NS)
    table='\n'.join(f'|{q["stage"]}|{q["summary"]["mean"]:.6f}|{q["kind"]}|' for q in trials)
    note=f'''# 当前权威交接：30.5px阶段目标，478–489（2026-11-12）

当前推荐实验入口：`event_drone_forecast/{best['entry']}`，原13参数模型`models/native_response_plane_v1.json`保留。
同六段实际RAW、原102起点，源时间0.4秒均值 **{base['summary']['mean']:.6f}→{current['mean']:.6f}px**，改善{gain:.6f}px；中位{base['summary']['median']:.6f}→{current['median']:.6f}px。
30.5px阶段目标{'已达到' if current['mean']<=30.5 else '未达到'}。30.5内{within}/102；这不代表每次预测都低于30.5。
{improved}/6录像平均改善，录像等权平均变化{macro:+.6f}px，不能称每段均改善。

|录像|原起点|473均值px|当前均值px|变化px|
|---|---:|---:|---:|---:|
{rows}

## 本轮实际改动与边界

487在既有桨叶支持内，对位移匹配用的log事件能量做31px局部均值/方差归一化；沿用既有局部统计尺度，没有搜索窗口或阈值。背景/单一恒定区域不产生纹理。原始ON/OFF、几何和频率分析输入不作这项变换。目标是降低各桨事件能量变化对整机平移的干扰，随后原力学状态和预测响应接收新位移。不是用NN替换五圆，不是灰度G重建；不是转速或推力真值改进证据。

已知平移合成控制中，各区域能量分别改变后，旧匹配2.767px、归一化0.226px；这只证明机制。真实收益以本表全102点为准。归一化是已有视觉方法的工程应用，不宣称新颖算法；参见[匹配代价与局部光度变化研究](https://www.cs.middlebury.edu/~schar/papers/evalCosts_cvpr07.pdf)。

478双向一致位移单独31.885724px，较473仅0.068871px，但3/6回退，录像等权略退0.016646。487已超过它。488将双向与归一化统一叠加31.585981，劣于487，不能按录像拼两者优点。

480当前点在跟踪凸包内的平面仿射仍回退。482时间戳去模糊丢16个原起点，只剩86/102；子集52.963不能当完整102分数，拒绝。484联合正反匹配代价、485再加完整支持资格均未超过473。486保留相邻法线误差相关性，已知幅值合成方差不再无界增长，但真实冻结观测31.998822回退；不推广。第一次绝对PSD审计在原控制也因混合单位/近奇异克隆状态失败，第二次采用相对谱尺度数值检查，控制全部六源预测差0；没有裁协方差或改预测来通过。

|实验|原102均值px|实际执行类型|
|---|---:|---|
{table}

489只在既定13个其他训练录像10–40秒，重跑新前端后重新拟合同一13参数/原目标函数和先验；188弱教师来自7个非空源，六个评测源整段排除。全13源各901包、无跳包，原始数据/sidecar不改。弱教师是本机桨叶轮廓而非人工COM真值。新模型回放34.855904回退，原模型复用控制六源差0；不扫描参数、不推广，原模型保留。

492沿用12个双向LK点、75%一致性、2px重投影、形状和覆盖门槛；仅把已有局部特征图改为31px归一化，并要求参考点在平面跟踪凸包内，才用共同仿射移动。线/低秩/局部不一致时回退487匹配，未降低侧视资格。已知平面仿射＋不等能量合成误差0.020px，但实际六源32.452010回退，未推广。局部亮度归一化并不能让高速桨叶内部纹理变成可靠的刚体点，不能仅以RANSAC一致就声称找到了真实平面运动。这是同一个“事件能量变化与平移混淆”问题的局部运动解法，不是新造五圆或外接NN。

## 用户关于盲测的新决定

2026-11-12用户明确：**“保留盲测，继续优化到门槛。”** 原来30px＋冻结新录像通过后才做FPS的顺序保持；30.5只是中间目标。
预留`E:/dataset/recordings/rec_20261004_162506/event.raw`固定10–16秒不许提前解码。483启动在CreateProcess前被自动审核拒绝，未产生预测/图像，`output/frozen_external483_20261112`不存在；其脚本已无条件退出，不可绕过。元数据核对表明与19个实验采集不同，不等于穷尽证明从未暴露。

## 还卡在哪里，下一步

当前不是边框中心差几像素的主问题；部分起点当前位置已较准，但桨叶外观变化会使位移/速度偏移，并被积分到0.4秒。487确实减轻了这一项，仍有录像16/3轻微回退，不能据它们坐标/时间写补丁。
更深层仍是可见周期代理与有效力增益的可辨识性：487六源直接检查gain_learned/gain_ever_learned均无更新（不是取不存在的gain_updates字段作0）。13/15/16/11/4最长合格输入历史3/2/1/1/0；3号21但激励最大0.015<旧0.08。旧307已经做过消除常外力和隐藏法向项的单位无关激励检验，16允许11次但FDE约1.8e-6px，不重复降门槛。不能把事件频率称真实机械RPM或牛顿推力。隐藏桨推力、风/重力/阻力仍归不可测合力。后续聚焦当前合格位移的空间非刚性与旋转外观误差，不继续扫默认推力、Singer时间常数或姿态噪声；不重复本轮失败的同参数重校准。

下一次只攻“当前位移是否来自平移，而非旋转外观”这一件事：先读189早半窗速度、279–282补偿失败及350关键帧匹配负例。可检查原始时间分割后的位移一致性是否能为全窗匹配提供独立的当前质量证据；不是继续把去模糊图输入几何，也不是用未来参考挑位移。先明确与旧方法的差别、真实机动允许的时间差和噪声相关性，再决定是否实现；不得在没有新证据时再次扫仿射门槛、局部窗口或模型系数。

全六源已经用于开发选择；排除模型拟合不等于独立泛化验证。参考是弱事件轮廓，不是真实重心。所有原102点保留，包括预测发出后未来新操纵；新观测更新新预测，不倒改旧预测。没有FPS优化或30Hz验收，也不是实际发布延迟精度。
保护两处yolov10、原RAW/SDK/权重及已发布NN包；没有上传、安装、新代理或额度操作。当前作业和下一入口以latest_handoff.json为准。
'''
    doc=ROOT/'docs/HANDOFF_20261112_TARGET30P5.md';doc.write_text(note,'utf8');(out/'handoff.md').write_text(note,'utf8')
    htmlrows=''.join(f'<tr><td>{n}</td><td>{s["n"]}</td><td>{s["baseline_mean"]:.3f}</td><td>{s["candidate_mean"]:.3f}</td><td>{s["delta"]:+.3f}</td></tr>' for n,s in scores.items())
    trialrows=''.join(f'<tr><td>{q["stage"]}</td><td>{q["summary"]["mean"]:.3f}</td><td>{q["kind"]}</td></tr>' for q in trials)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>30.5px目标：桨叶位移改善</title><style>body{{max-width:1080px;margin:32px auto;padding:0 22px;background:#f3f6fa;color:#24384c;font:17px/1.8 system-ui,"Microsoft YaHei"}}section{{background:white;padding:20px 27px;border-radius:12px;margin:20px 0}}img{{max-width:100%}}td,th{{padding:7px 18px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}</style>
<h1>30.5px目标：实测改善，盲测保留</h1><section><h2>{base['summary']['mean']:.3f} → {current['mean']:.3f}px</h2><p>同六段实际RAW、原102起点，0.4秒平均改善{gain:.3f}px。中位{current['median']:.3f}px，30.5内{within}/102。阶段目标尚未达到。{improved}/6录像均值改善；没有删点或分录像拼接。</p><img src="comparison.png"><table><tr><th>录像</th><th>原起点</th><th>上一轮px</th><th>当前px</th><th>变化px</th></tr>{htmlrows}</table></section>
<section><h2>这次改了什么</h2><p>只在位移匹配时归一化局部事件能量，减轻各桨亮暗变化被误判为平移。原始ON/OFF和周期/几何输入保留；纯桨叶→五圆可见性分工→可见贡献＋未知合力→0.4秒预测主线保留，没有NN或G。</p><p>该优化是已有光度匹配思想的工程应用，<a href="https://www.cs.middlebury.edu/~schar/papers/evalCosts_cvpr07.pdf">相关原论文</a>讨论匹配代价对局部能量变化的敏感性；本项目没有声称完整复现该文或测准真实推力。</p></section>
<section><h2>全部起点，回退也保留</h2><img src="all_origins.png"><p>未删除预测后发生新操纵的起点。只能用新观测更新下一条预测，不能改过去已发出的结果。</p></section>
<section><h2>本轮结果</h2><table><tr><th>实验</th><th>均值px</th><th>执行类型</th></tr>{trialrows}<tr><td>482</td><td>无可比完整分数</td><td>丢16点，仅86/102，拒绝</td></tr></table><p>当前统一入口：<code>{best['entry']}</code>。双向校正叠加不优，未按录像拼优。489新模型若有回放收益，还须RAW入口复现。</p></section>
<section><h2>限制与下一步</h2><p>30.5px目标未到；继续先精度。用户已明确保留盲测，原30px＋新录像验证后的FPS门槛保持。六段均已暴露开发，参考是弱轮廓而非COM真值，尚不能宣称新录像泛化或实时30Hz。</p><p><a href="handoff.md">完整交接与下一步</a> · <a href="../{best['folder']}/results.json">逐点评分</a> · <a href="summary.json">机器可读汇总</a></p></section></html>'''
    (out/'index.html').write_text(page,'utf8')
    summary=dict(best_stage=best['stage'],best=current,baseline473=base['summary'],gain_px=gain,
        target30p5_achieved=current['mean']<=30.5,all_original102_preserved=True,
        source_improved_count=improved,source_equal_macro_delta_px=macro,scores=scores,records=records,
        trial_summaries={q['stage']:q['summary'] for q in trials},new_video_validated=False,FPS_work=False,
        reserved_video_user_chose_preserve=True,entry=str(ROOT/best['entry']))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf8')
    print('SAVED',out/'index.html',best['stage'],current,flush=True)
