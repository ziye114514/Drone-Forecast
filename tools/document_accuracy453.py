"""Save the actual accuracy result, limits, failures and reproducible handoff."""
from pathlib import Path
from datetime import datetime
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent

if __name__=='__main__':
    out=WORK/'output/accuracy_progress453_20261110';out.mkdir(exist_ok=True)
    d=json.loads((WORK/'output/accuracy451_20261110/results.json').read_text('utf8'));local=json.loads((WORK/'output/local_geometry454_20261110/results.json').read_text('utf8'))
    p=WORK/'output/anonymous_group456_20261110/results.json';anonymous=json.loads(p.read_text('utf8')) if p.exists() else None
    primary=d['pooled']['accuracy451'];control=d['pooled']['complete447'];names=('13','15','16','11','3','4')
    extra=('匿名区域组456：'+json.dumps(anonymous['pooled'],ensure_ascii=False)+'。只读重放，尚未接真实RAW工厂。' if anonymous else '匿名区域组456尚未结束；不能引用455为有效改进。')
    table='\n'.join(f'|{n}|{d["cases"][n]["accuracy451"]["original_available"]["n"]}|{d["cases"][n]["complete447"]["original_available"]["mean"]:.3f}|{d["cases"][n]["accuracy451"]["original_available"]["mean"]:.3f}|' for n in names)
    note=f'''# 当前权威交接：441–456（2026-11-10）

先0.4s平均误差30px，在冻结新录像通过后再FPS；P95只记录。原102评分起点不删、不改目标时间；未来新操纵不倒灌旧预测。保护两处yolov10、RAW/SDK/权重/公开NN包。不购买或用重置券。账户限额通过get_usage_limits核验；旧formal goal的usageLimited不是账户当前额度。未完成30px目标。

## 当前入口和准确分母

推荐研究入口：`event_drone_forecast/run_rotor_accuracy_candidate.py`。RAW→纯ON/OFF周期像素→原生桨叶支撑及当前位移→五圆4/3/2/线→可测部分＋未知力→共享响应先验→20ms原生可见包络参考预测。无机体NN/G。共享原模型`models/native_response_plane_v1.json`未换，186教师；6评测源整段排除拟合，但均已暴露开发。弱事件轮廓参考不是独立人工/COM真值。

**相同实际RAW网格、原102全部输出：完整447为{control['mean']:.6f}→累计451为{primary['mean']:.6f}px，改善{control['mean']-primary['mean']:.6f}px。**中位26.706→27.684（回退）；30内54→55。4/6录像改善、2/6回退，不称全面稳定或泛化成功。

|录像|原起点|完整447均值px|累计451均值px|
|---|---:|---:|---:|
{table}

旧436原缓存102为34.295，旧实际RAW101为33.396，不能跨网格/分母直接计算增益。451与旧实际436共同101为33.475（略退）；恢复的原11@12.0为5.304px。必须同时披露，不能用补上易点冒充共同点普遍改善。当前30未达，预留新录像仍未解码，未开始FPS优化或验收。

## 已落地和已实跑

441/444：当前严格像素＋双向匹配确认更早出生，出生位移仅给图像搜索先验；实体速度仍等待后续实测。修复0.12s图像身份先失锁的问题，保留5次观测资格。六段真实包均181、无跳包；原缺失11@12.0恢复。444原102均33.611846，共同101较旧436回退0.1507，未隐藏。

447/448：几何参考初始化与位置可用性分开；只在前5个合格完整参考前，以3样本medoid稳定几何偏移，当前位置校正仍即时使用。之后逐字回原读出。实际RAW11共181包与冻结447预测差0。不是长期框平滑、没有改内部力学。

449：33.333ms来源与20ms参考窗按中心差6.6665ms显式换算；不搜索延迟、不改时间戳/参考终点、不读未来事件。仅是非均匀事件的近似观测模型，不是瞬时COM。完整102单项33.361241→33.317655。

450：进入目标初始速度先验从100px/s放宽为近似弥散1e6px/s，后两帧真正位移决定速度，不把首次搜索位移当物理速度。主/备用观测器同时改变，原类保持。合成600px/s匀速消除了第二帧虚假加速度；真实102为33.237485。没有扫初值、重训或按录像切换。

451/452：全6统一叠加两项为33.199150；实际RAW4共181包，当前输入/状态一致，0.4s预测与冻结451差0，跳包0。入口精度优先unpaced，这不是实时30Hz证书。旧410/429/432/436/447入口及模型均保留。

## 未推广的尝试和真实缺口

442/443/445：永久参考平滑、收缩保持、位置/几何拆分保持均跨源回退，保留负例，不继续扫保持门槛。446：167个连续注册材料位移教师替代186候选标签仍回退，旧模型保留；材料点不等于真实重心。

454：局部厚圆弧可在全局扁长布局下继续提议，关闭侧视快捷预算，未放宽原弧拟合质量。6段RAW全部实跑，102均33.866553，四桨时刻/当前逐桨输入次数不增，未推广。新增弧不等于独立真实桨盘，不能让更多曲线直接变成更多可信姿态。`curvature_hz=0`表示不限制频率，之前“被关闭”的猜测已排除。可选入口`run_rotor_local_geometry_candidate.py`仅实验，推荐入口不启用它。

{extra}

455首试未显式剥离输出窗元数据、未做控制一致性且力学禁更行处理不足，标记superseded；456修复、先验证六源原控制差<1e-7后评分。不能将455的小数变化算有效收益。重放工具新增清除派生读出元数据，旧无该字段输入不受影响。

当前6段各181包，原逐桨slot输入只有0/10/1/0/6/0包，推力增益在线更新均0。其余多为混合频率变化代理；不等于真实RPM，更不是总推力。主要精度收益来自桨叶定位、位移和状态估计。训练源h后验sigma约986–1029，接近初始1000，不能据此虚构高置信隐藏推力先验。二力模型已实现，独立推力增益的预测价值仍弱。

## 下一步，按根因推进

1. 先读456结果和控制parity；仅当同102真正改善才接实际RAW，任何新收益都用统一策略，不按录像拼接。匿名组只统计分离且当前合格的区域，不填隐藏桨、不猜头尾；极端侧线仍只有匿名周期代理。
2. 优先攻“当前局部周期→同一可见来源的连续变化”，复用已有`local_temporal_phase.py`、`local_period_tracks.py`、`SharedPhaseFFT`和已编译HelixTrack桥，不从头造模块。旧143/180/203/231/235/279–282/334–337/395已尝试过局部/相位/补偿，必须先读负例。当前fixed-pixel阶段仍受平移、谐波、混合来源影响；补偿只改善周期观测时不得更改原生几何像素。不能简单给混合峰开放增益：单位下变化远小于工程噪声且增益/未知力可混淆。
3. 不再靠更复杂推力公式、恒定k扫参或高增益ESO掩盖缺观测；先让新局部周期对可测部分有稳定贡献，再检查0.4s增量。不强求提前知道未来新操纵。
4. 30px通过后再冻结预留`E:/dataset/recordings/rec_20261004_162506/event.raw`10–16s，核历史重叠；当前仍未解码，不能提前看并调参。通过后才FPS。

## 本次复核的主来源（不是新发现或完整复现）

- [原论文2604.18289](https://arxiv.org/html/2604.18289v1)：频率经Cf与质量进加速度；世界位置还使用已知机架尺寸、相机内参与观察者位姿。其条件不能直接搬成未知大小单目px误差。
- [HelixTrack](https://arxiv.org/html/2603.09235v1)与[作者代码](https://github.com/radimspetlik/helixtrack)：联合相位/几何有帮助；原文限制含粗初始化、可见环、侧视退化、无自动复锁。本机已多次编译/调用，不宣称首次接入或现成解决侧视。
- [EHGO未知轨迹](https://arxiv.org/abs/2003.06390)：估计扰动和参考运动用于自机控制，假设自机位置/姿态及参考位置；不能等价外部RAW四旋翼未来操纵可知。
- [Cascade ESO](https://arxiv.org/abs/2004.01483)：针对高增益放大测量噪声的问题；当前不盲目提高带宽，未宣称照搬会降低px。

报告`output/accuracy_progress453_20261110/index.html`，主逐点评分`output/accuracy451_20261110/results.json`，RAW证明448/452；454/456各自快照/逐点结果保留。额度和后台状态以latest_handoff.json最新记录为准。
'''
    (ROOT/'docs/HANDOFF_20261110_ACCURACY453.md').write_text(note,'utf8');(out/'handoff.md').write_text(note,'utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained');x=np.arange(6)
    a=[d['cases'][n]['complete447']['original_available']['mean'] for n in names];b=[d['cases'][n]['accuracy451']['original_available']['mean'] for n in names]
    axes[0].bar(x-.18,a,.36,label='完整447');axes[0].bar(x+.18,b,.36,label='累计451');axes[0].set_xticks(x,names);axes[0].axhline(30,color='grey',linestyle='--');axes[0].set_ylabel('0.4s原起点均值 / px');axes[0].set_xlabel('开发录像');axes[0].legend()
    for path,label in [('reference_bootstrap447_20261110/temporal_11.jsonl','原初始速度先验'),('diffuse_startup450_20261110/11_diffuse.jsonl','弥散初始速度先验')]:
        rr=[json.loads(s) for s in (WORK/'output'/path).read_text('utf8').splitlines()];rr=[r for r in rr if 11.76<=r['t']<=12.04 and r.get('coefficient')]
        axes[1].plot([r['t'] for r in rr],[r['coefficient']['acceleration'][0] for r in rr],marker='o',label=label)
    axes[1].set_ylabel('估计横向加速度 / px/s²');axes[1].set_xlabel('11号RAW时间 / s');axes[1].set_title('初始化机制示例，不是真值加速度');axes[1].legend();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    rows=''.join(f'<tr><td>{n}</td><td>{d["cases"][n]["accuracy451"]["original_available"]["n"]}</td><td>{u:.3f}</td><td>{v:.3f}</td></tr>' for n,u,v in zip(names,a,b))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>桨叶预测累计精度与剩余卡点</title><style>body{{font:17px/1.8 system-ui,"Microsoft YaHei";max-width:1050px;margin:30px auto;background:#f4f6fa;color:#233249;padding:0 20px}}section{{background:white;padding:22px 28px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse}}td,th{{padding:8px 25px;border-bottom:1px solid #ddd}}img{{width:100%}}</style><h1>桨叶预测：保留小收益，也明确未达标</h1><p>2026-11-10 · 纯ON/OFF桨叶主线 · 30px未达 · 新录像未验证 · FPS后置</p><section><h2>同102个原定起点，33.361 → 33.199px</h2><p>改善0.162px，中位26.706→27.684（回退），30px内54→55。四段改善、两段回退。旧实际101点不能直接与当前102点比；双方共同101为33.396→33.475，新增恢复点11@12.0为5.304px，完整披露。</p><img src="comparison.png" alt="六源均值及初始加速度示例"><table><tr><th>录像</th><th>原起点</th><th>完整447</th><th>累计451</th></tr>{rows}</table></section><section><h2>已接入完整RAW</h2><p>更早的当前桨叶身份确认与搜索续接、参考初始化与当前位置分工、弥散初始速度、固定20ms观测窗换算已累计。4号真实RAW181包与冻结预测差0，无跳包。无NN/G、无新权重拟合、不分录像开关。目标为原生可见桨叶包络中心，不是瞬时真实重心。</p></section><section><h2>还没有解决的核心问题</h2><p>六段1086包只有17包提供原逐桨slot输入，推力增益在线更新全为0。混合周期不等于可靠逐桨转速，已有未知力估计也没有把隐藏推力幅值辨识清楚。当前收益主要来自定位和运动状态，不能说已靠推力成功提前预测变向。</p><p>局部弧度提取454实际六段33.867px，四桨/逐桨输入次数没增，未接默认。{extra}</p></section><section><h2>可复现与交接</h2><p>推荐入口 <code>event_drone_forecast/run_rotor_accuracy_candidate.py</code>。原基线及失败实验全部保留。<a href="handoff.md">完整交接、下一步及文献边界</a> · <a href="../accuracy451_20261110/results.json">102逐点评分</a> · <a href="../raw_accuracy452_20261110/reproduction.json">实际RAW复现</a> · <a href="../local_geometry454_20261110/results.json">局部几何负例</a></p></section></html>'''
    (out/'index.html').write_text(page,'utf8')
    p=ROOT/'README.md';s=p.read_text('utf8');start='<!-- accuracy453-current:start -->';end='<!-- accuracy453-current:end -->'
    if start in s:s=s[s.index(end)+len(end):].lstrip()
    p.write_text(f'''{start}
> **当前研究状态（2026-11-10）**：`run_rotor_accuracy_candidate.py`为原生桨叶＋五圆＋可测/未知力主线，无机体NN/G。同实际网格原102起点完整447 **33.361→33.199px**；中位回退，4/6改善，30px未达，新录像未验证，FPS后置。RAW4共181包复现差0。当前逐桨输入稀少，推力增益在线更新仍0，不称推力创新已成功。局部几何454退化未接默认；原入口和公开NN发布包保留。先读[权威交接](docs/HANDOFF_20261110_ACCURACY453.md)与[图文报告](../output/accuracy_progress453_20261110/index.html)，下面为历史。
{end}

'''+s,'utf8')
    p=WORK/'output/checkpoints/latest_handoff.json';j=json.loads(p.read_text('utf8'));j.update(timestamp=datetime.now().isoformat(),objective_completed=False,
        in_progress=anonymous is None,background_jobs=[] if anonymous else [dict(session_id=12852,task='anonymous456 CPU')],running_jobs=[] if anonymous else [12852],experiments_running=[] if anonymous else ['anonymous456'],
        status='Actual-grid original102 all available, mean33.199150;30px/new-video gate unmet. Local geometry454 regressed. No FPS work.',
        candidate_entry=str(ROOT/'run_rotor_accuracy_candidate.py'),current_status_document=str(ROOT/'docs/HANDOFF_20261110_ACCURACY453.md'),latest_report=str(out/'index.html'),
        current_precision_mean_px=primary['mean'],aggregate_same102=primary,candidate_scores={n:d['cases'][n]['accuracy451']['original_available'] for n in names},
        current_user_stage='441-456 accuracy-first; original102 retained; actual proof452 passed',
        current_precision_p95_px=float(np.percentile([q['error_px'] for n in names for q in d['cases'][n]['accuracy451']['records'] if q['valid'] and q['t'] in {r['t'] for r in json.loads((WORK/'output/reference429_20261110/results.json').read_text('utf8'))['cases'][n]['control410']['records'] if r['valid']}],95)),
        actual_grid_complete447=control,accuracy451=primary,raw_latest_reproduction=json.loads((WORK/'output/raw_accuracy452_20261110/reproduction.json').read_text('utf8')),
        local_geometry454=dict(promoted=False,mean_px=local['pooled']['local454']['mean'],all6_actual_RAW=True,actual_gain_updates=0),
        anonymous456=None if anonymous is None else dict(pooled=anonymous['pooled'],evidence=anonymous['evidence'],control_parity=anonymous['control_parity'],actual_RAW_connected=False),
        next_action=['Read456; do not use superseded455 as a validated effect. Only integrate if actual same102 accuracy helps.',
            'Prioritize current periodic-source continuity and identifiable image response; avoid repeating prior local phase/compensation negatives.',
            'Keep original102 and reserved video unread until30px gate, then verify frozen video before FPS.'],
        project_work_state='accuracy_work_handoff_saved_not_completed',GT_recall_newly_measured=False,FPS_optimization_allowed=False)
    p.write_text(json.dumps(j,ensure_ascii=False,indent=2),'utf8')
    p=WORK/'WORKLOG.md';s=p.read_text('utf8');head='## 2026-11-10 441–456：完整102恢复、初始速度与观测窗累计'
    if head not in s:p.write_text(s+'\n\n'+head+'\n\n'+note+'\n','utf8')
    (out/'summary.json').write_text(json.dumps(dict(primary=primary,full_control447=control,local454=local['pooled'],anonymous456=None if anonymous is None else anonymous['pooled'],thirty_px_achieved=False,new_video_validated=False,FPS_work=False),indent=2),'utf8')
    print('SAVED',str(out/'index.html'),primary)
