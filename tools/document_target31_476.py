"""Record the complete target31 batch without hiding failed hypotheses."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent

if __name__=='__main__':
    out=WORK/'output/target31_20261111';out.mkdir(exist_ok=True)
    baseline=json.loads((WORK/'output/positive_normal457_20261110/results.json').read_text('utf8'))
    specifications=[('470','raw_polarity470_20261111','polarity470','run_rotor_polarity_candidate.py'),
        ('473','raw_mean473_20261111','joint_only','run_rotor_joint_only_candidate.py'),
        ('474','raw_full_joint474_20261111','full_joint','run_rotor_full_joint_candidate.py'),
        ('475','raw_multichannel475_20261111','multichannel','run_rotor_multichannel_candidate.py')]
    candidates=[]
    for stage,folder,arm,entry in specifications:
        d=json.loads((WORK/'output'/folder/'results.json').read_text('utf8'));p=d['pooled'][arm]
        assert p['n']==102 and not d.get('missing_original',{}).get(arm,[])
        candidates.append(dict(stage=stage,folder=folder,arm=arm,entry=entry,summary=p,results=d))
    best=min(candidates,key=lambda d:d['summary']['mean']);current=best['summary'];old=baseline['pooled']['positive'];achieved=current['mean']<=31
    ns=('13','15','16','11','3','4');scores={n:best['results']['cases'][n][best['arm']]['original_available'] for n in ns}
    table='\n'.join(f'|{n}|{scores[n]["n"]}|{baseline["cases"][n]["positive"]["original_available"]["mean"]:.3f}|{scores[n]["mean"]:.3f}|' for n in ns)
    entry=ROOT/best['entry'];stamp=datetime.now(timezone(timedelta(hours=8))).isoformat()
    failed={'461 endpoint trend':32.61794520802857,'464 likelihood mode credit':33.05898688285517,
        '465 observed Singer consistency':34.93166934089867,'466 matched prior moments':32.416767974625074,
        '472 aperture covariance':34.32481490655817,'472 covariance plus joint mean':34.28622064934548,
        '474 full sampled-neighborhood vertex':32.34126640403351,
        '475 separate ON/OFF matching':32.02790747637339,
        '477 horizon-aware matured force credit':32.06825347889552}
    horizon=json.loads((WORK/'output/horizon_credit477_20261111/results.json').read_text('utf8'))
    assert horizon['original102_preserved'] and all(v==0 for v in horizon['control_parity'].values())
    assert abs(horizon['pooled']['control']['mean']-current['mean'])<1e-9
    assert abs(horizon['pooled']['horizon']['mean']-failed['477 horizon-aware matured force credit'])<1e-9
    note=f'''# 当前权威交接：目标31px，461–477（2026-11-11）

当前入口：`event_drone_forecast/{best['entry']}`。同六源实际RAW、原102起点，0.4s源时间平均 **{old['mean']:.6f}→{current['mean']:.6f}px**，改善{old['mean']-current['mean']:.6f}px；中位{old['median']:.6f}→{current['median']:.6f}。31内{current['within31']}/102。31目标{'已达到' if achieved else '未达到'}，不等于每次误差都低于31。所有原起点保留，不能改分母、删除未来新操纵误差或按录像拼接。

|录像|原起点|457均值px|当前均值px|
|---|---:|---:|---:|
{table}

## 已完成，不能再按旧checkpoint重复

上一轮458真实16号181包与457预测差0；459原186教师重训退到33.925，原13参数模型保留。461旧角趋势重用略退；462初试未初始化诊断字段失败，未评分，463修复后原控制全6差0。463把增益尺度先验与可见事件频率及原正向响应先验对应，均32.374；不是质量/CT/真实RPM辨识。

467无序、互不重叠、各有当前周期证据的区域组不必等四桨模板，只计可见部分；隐藏桨仍归未知项，头尾/物理桨ID不需要。平均32.302。468匹配增益二阶矩、保留旧均值200的控制32.280，几乎复制收益；说明主要来自不确定度传播，不能宣称推力均值算准。该保守输入不确定度路线接入partial入口。469真实16号181包复现差0。

470复用已有ON/OFF独立位移分歧，只增加外观误差R，保留联合位移、body和频率输入；避免对已有分歧字段重复加噪声。六源真实RAW全部181包，无跳包，平均32.086852；11号改善1.34px。473只改联合二维亚像素峰值，保留470方差，平均31.954595。472把峰值曲率当协方差回退，未推广。

474修正合成已知峰中二维顶点可位于3x3采样域内而超出半像素格的情况，事先声明、不搜索阈值；合成误差0.398→0.160→0.019px，但真实32.341回退，保留实验。合成成功不得冒充真实预测收益。475保持ON/OFF双通道联合相关，结果{candidates[-1]['summary']['mean']:.6f}px。最终只选择一个全局候选{best['stage']}，没有任何录像ID条件。

477只改变已经到期的0.1/0.2/0.4s误差如何更新后续模型信用，使用原Singer过程方差，没有更改已发预测或评分，也没有删除突变样本。六源473测量回放控制差全0，试验32.068253px，未超过473；保留为负例，不接入推荐入口。该项是冻结测量回放，不额外冒称六源新RAW实跑。

当前470/472/473/474/475所有六源RAW实跑完成，全部输出后才读取参考评分。SOURCE_FILES在每批运行中冻结；不能拿单源结果拼接。当前场景几何/周期输入仍有限，推力增益在线更新仍未得到可靠辨识，很多收益来自桨叶位移与置信度的改进。仍是纯ON/OFF桨叶、五圆4/3/2/线、可测代理＋不可测合力；没有NN/G、没有隐藏转速补测或实际牛顿力标定。

剩余大误差中当前位置可基本准确，而延续明显错误：3号15.6s为151.861px、当前位置3.690px；4号12.0s为109.131px、当前位置5.943px；15号11.4s为101.670px、当前位置7.670px。这只是事后误差分解证据，不能直接据此认定未来机动不可预测，更不能删掉这些原起点。下一步需要区分起点已有速度/力估计偏差和发布后才出现的控制变化；后者不倒灌旧预测。继续抠边界不是本轮优先项。

## 实验边界与后续顺序

6个评测源均已暴露开发；原模型拟合排除整段6源，但本轮算法选择仍使用这组开发数据。没有按录像写补丁不等于已证明无过拟合。原参考是独立于预测的事件轮廓代理，非独立人工/COM真值。当前是离线源时间0.4s误差，不是实际发布时间的实时精度；未做FPS优化或验收。

用户本轮目标31；原长期30px＋冻结新录像通过后才优化FPS的顺序保持。预留`E:/dataset/recordings/rec_20261004_162506/event.raw`10–16s仍未解码；不能提前看并调参。未来新操纵不倒灌旧预测，使用新观测发出新的t+0.4s预测。P95只记录，不继续抠轮廓/IoU。

下一步优先解决当前转速/推力代理与真实运动的可辨识联系、持续可见来源的频率一致性；复用既有HelixTrack/SharedPhaseFFT/匿名单环桥，先读其203/231/235/279–282/334–337/395负例。不要再扫本批失败的角趋势、全局Singer、初始方差或峰值曲率噪声。也不要继续加大默认推力均值。当前已看到增益均值消融近似等效，应将力学测量价值与观测不确定度收益分开报告。

保护两处yolov10、原RAW/SDK/权重及已发布NN包；不上传、不安装新依赖、不购买/重置额度。E盘在普通沙箱中拒绝读取，已授权的提升命令可正常读；这不是硬盘未接。无新代理或外部消息。

## 可复核来源

[Markov切换模型原论文](https://ieeexplore.ieee.org/document/1299/)用于464思路核验；本地实现没有状态交互，不是完整IMM，且未改善均值，未采用。[Lucas–Kanade原文](https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf)用于配准与方向弱观测背景；本地相关峰协方差是工程近似，真实验证失败，不称论文完整复现或新算法。

报告`output/target31_20261111/index.html`；当前逐点结果`output/{best['folder']}/results.json`。所有本批作业已退出。当前代码/模型/报告指针见latest_handoff.json；旧formal goal状态不是账户实时额度。
'''
    doc=ROOT/'docs/HANDOFF_20261111_TARGET31.md';doc.write_text(note,'utf8');(out/'handoff.md').write_text(note,'utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');x=np.arange(6)
    ax.bar(x-.18,[baseline['cases'][n]['positive']['original_available']['mean'] for n in ns],.36,label='上一轮457')
    ax.bar(x+.18,[scores[n]['mean'] for n in ns],.36,label=f'当前{best["stage"]}');ax.set_xticks(x,ns);ax.axhline(31,color='grey',linestyle='--',label='本轮31px目标');ax.set_ylabel('固定原起点0.4s均值 / px');ax.set_xlabel('已暴露开发录像');ax.legend();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    rows=''.join(f'<tr><td>{n}</td><td>{scores[n]["n"]}</td><td>{baseline["cases"][n]["positive"]["original_available"]["mean"]:.3f}</td><td>{scores[n]["mean"]:.3f}</td></tr>' for n in ns)
    failures=''.join(f'<li>{k}: {v:.3f}px</li>' for k,v in failed.items())
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>31px目标优化结果</title><style>body{{font:17px/1.8 system-ui,"Microsoft YaHei";max-width:1050px;margin:30px auto;padding:0 20px;background:#f4f6fa;color:#243249}}section{{background:white;border-radius:12px;padding:20px 28px;margin:18px 0}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 22px;border-bottom:1px solid #ddd}}</style><h1>31px目标：实际RAW累计优化</h1><p>2026-11-11 · {'目标均值已达到' if achieved else '目标均值尚未达到'} · 纯桨叶主线 · 新录像未验证 · FPS后置</p><section><h2>{old['mean']:.2f} → {current['mean']:.2f}px</h2><p>相同原102起点、六源实际RAW。改善{old['mean']-current['mean']:.3f}px，中位{current['median']:.2f}px，31px内{current['within31']}/102。保留回退和未来新操纵误差，没有删点或分录像切换。</p><img src="comparison.png"><table><tr><th>录像</th><th>原起点</th><th>上一轮px</th><th>当前px</th></tr>{rows}</table></section><section><h2>改了什么</h2><p>无序可见周期区域不等完整四桨模板；隐藏桨留未知项。按事件频率尺度传播输入增益的不确定度；用ON/OFF位移分歧识别外观误差；二维匹配峰细化修正分轴估计。当前全局候选为{best['stage']}。仍无整机NN/G，没有新模型拟合。</p><p>匹配二阶矩的均值消融几乎复制了收益，说明主要是置信度和位移改善，不能说质量、转速或推力已经测准。</p></section><section><h2>失败尝试保留</h2><ul>{failures}</ul><p>合成峰值定位更准不保证真实无人机预测更准；失败机制没有叠进当前推荐入口。</p></section><section><h2>复现和限制</h2><p>入口：<code>event_drone_forecast/{best['entry']}</code>。原始输入只有RAW及本机SDK/GPU路径。当前参考是弱事件轮廓，不是真实重心；六源已用于开发选择，泛化尚未验证。这里不是实时发布时间精度或30Hz证书。</p><p><a href="handoff.md">完整交接与下一步</a> · <a href="../{best['folder']}/results.json">逐点评分</a> · <a href="../raw_partial469_20261111/reproduction.json">RAW入口复现证明</a> · <a href="../checkpoints/native_joint_peak_synthetic_20261111.json">合成已知位移检查</a></p></section></html>'''
    (out/'index.html').write_text(html,'utf8');(out/'summary.json').write_text(json.dumps(dict(best_stage=best['stage'],best= current,baseline457=old,target31_achieved=achieved,entry=str(entry),candidate_summaries={c['stage']:c['summary'] for c in candidates},failures=failed,new_video_validated=False,FPS_work=False),indent=2),'utf8')
    p=ROOT/'README.md';s=p.read_text('utf8');start='<!-- target31-current:start -->';end='<!-- target31-current:end -->'
    if start in s:s=s[s.index(end)+len(end):].lstrip()
    header=f'''{start}
> **当前研究（2026-11-11）**：同原102起点0.4s均值 **32.600→{current['mean']:.3f}px**，中位{current['median']:.3f}；31目标{'已达到' if achieved else '未达到'}。推荐`{best['entry']}`，纯ON/OFF桨叶＋五圆＋可测/未知力，不用NN/G。收益主要来自位移和不确定度校正，不称推力已辨识。六源均已暴露开发，新录像未验证，FPS后置。见[报告](../output/target31_20261111/index.html)与[当前交接](docs/HANDOFF_20261111_TARGET31.md)。以下为历史。
{end}

''';p.write_text(header+s,'utf8')
    p=WORK/'WORKLOG.md';s=p.read_text('utf8');mark='<!-- target31-handoff:start -->';last='<!-- target31-handoff:end -->'
    if mark in s:s=s[s.index(last)+len(last):].lstrip()
    p.write_text(f'{mark}\n当前交接：[目标31](event_drone_forecast/docs/HANDOFF_20261111_TARGET31.md)；当前均值{current["mean"]:.6f}px，原102起点。入口{best["entry"]}。所有本批作业已退出。31目标未达时不可标完成；原30＋新录像验证后的FPS门槛保留。\n{last}\n\n'+s+'\n\n## 2026-11-11 476收尾\n\n'+note,'utf8')
    p=WORK/'output/checkpoints/latest_handoff.json';j=json.loads(p.read_text('utf8'));j.update(timestamp=stamp,in_progress=False,running_jobs=[],background_jobs=[],experiments_running=[],objective_completed=False,
        current_precision_mean_px=current['mean'],aggregate_same102=current,candidate_scores=scores,candidate_entry=str(entry),latest_report=str(out/'index.html'),current_status_document=str(doc),
        current_target_FDE_px=31,next_target_FDE_px=31,milestone31_achieved=achieved,
        status=f'477 complete: original102 actualRAW mean{current["mean"]:.6f}; target31 achieved={achieved}; new video unvalidated;FPS deferred.',
        current_user_stage='461-477 complete; best473 all6 actualRAW;477 frozen-measurement replay',project_work_state='accuracy_batch_finished_research_goal_not_completed',
        current_pipeline_uses_body_NN=False,raw_current_all6_complete=True,FPS_optimization_allowed=False,
        no_recording_specific_inference_rules=True,all6_exposed_development=True,
        next_action=['Read current477 handoff; do not rerun failures461/464/465/466/472/474/475/477.','Prioritize observable periodic-source continuity and distinguish force-mean information from uncertainty benefits.','Keep reserved video unread until30px gate, then freeze and verify before FPS.']);p.write_text(json.dumps(j,ensure_ascii=False,indent=2),'utf8')
    print('SAVED',str(out/'index.html'),best['stage'],current,'target31',achieved)
