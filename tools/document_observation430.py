"""Refresh the accuracy handoff from completed, unchanged-origin results."""
from pathlib import Path
import json,html,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent


if __name__=='__main__':
    result=json.loads((WORK/'output/reference429_20261110/results.json').read_text('utf8'))
    pooled=result['pooled_same_original_origins'];old=pooled['control410'];now=pooled['qualified429']
    assert now['n']==old['n']==102
    parity=json.loads((WORK/'output/raw_observation430_20261110/reproduction.json').read_text('utf8'));assert parity['raw_reproduction_passed']
    out=WORK/'output/observation_accuracy_20261110';out.mkdir(exist_ok=True)
    order=('13','15','16','11','3','4')
    table=[];md=[]
    for n in order:
        case=result['cases'][n];scores=[case[k]['same_original_origins'] for k in ('control410','bounded426','qualified429')]
        table.append('<tr><td>'+n+'</td><td>'+str(scores[0]['n'])+'</td>'+''.join(f'<td>{s["mean"]:.3f}</td>' for s in scores)+'</tr>')
        md.append('|'+n+'|'+str(scores[0]['n'])+'|'+ '|'.join(f'{s["mean"]:.3f}' for s in scores)+'|')
    summary=dict(primary_metric='Mean Euclidean endpoint error at0.4s, same original102 origins',old410=old,
        integrated426=pooled['bounded426'],integrated429=now,improvement_px=old['mean']-now['mean'],
        improvement_percent=100*(1-now['mean']/old['mean']),unchanged_reference_files=True,
        original_origins_retained=102,additional_outputs_separate=True,
        independent_new_video_validated=False,target30_achieved=False,FPS_work_deferred=True,
        P95_is_descriptive_not_optimization_target=True,all6_exposed_development=True,
        all6_recordings_excluded_from_model_fit=True,reference_not_independent_COM_truth=True,raw_reproduction=parity)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),'utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(6);w=.37
    a=[result['cases'][n]['control410']['same_original_origins']['mean'] for n in order]
    b=[result['cases'][n]['qualified429']['same_original_origins']['mean'] for n in order]
    ax.bar(x-w/2,a,w,label='Preserved410');ax.bar(x+w/2,b,w,label='Observation429')
    ax.axhline(30,color='black',linestyle='--',label='Target30px');ax.set_xticks(x,order)
    ax.set_xlabel('Recording (fixed original origins)');ax.set_ylabel('Mean0.4s endpoint error /px')
    ax.set_ylim(0,70);ax.legend();fig.tight_layout();fig.savefig(out/'mean_comparison.png',dpi=160);plt.close(fig)
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>桨叶当前观测与位置精度</title>
<style>body{{font:17px/1.7 system-ui,"Microsoft YaHei";max-width:1050px;margin:36px auto;padding:0 24px;color:#243248;background:#f5f7fa}}h1{{font-size:30px}}section{{background:white;padding:20px 25px;margin:18px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}code{{overflow-wrap:anywhere}}.metric{{font-size:29px;font-weight:700;color:#165b7a}}</style>
<h1>接住当前桨叶位移，平均预测误差34.61px</h1><p>2026-11-10 · 精度优先 · 仍未达到30px · 尚未运行保留的新录像验证</p>
<section><div class="metric">36.463 → 34.607px（改善{summary['improvement_percent']:.2f}%）</div><p>同6段录像、同102个原起点、同0.4秒终点及原参考。30px内51→56个。另增5个可输出起点单独记录，不混入精度分母。P95只作附带记录，不以它驱动边界精修。</p></section>
<section><h2>这轮实际改了什么</h2><ol><li>冷启动仍需原严格周期资格；已有桨叶身份时，允许当前双极支撑与双向匹配提供位移。全局周期空缺时最多沿用120ms身份，空缺包不能刷新身份时钟。</li><li>存在、位移和转速分开授权：弱续踪不更新电机频率bank，不伪造可测推力。隐藏桨叶仍在未知力中。</li><li>弱当前观测沿用本来缺测时的传播先验后再更新。零创新不会因“从无框变有框”凭空改变加速度。</li><li>弱支撑不再自动重定机体参考点。静止材料参考点＋支撑框摆动的控制例，人工输出摆动8.99→0px。这是合成机制检查，不是真实机体COM精度。</li><li>已有完整原生支撑提供视野裁剪的输出映射，内部运动与推力不变；未针对IoU重新修边。</li></ol><p>主线仍为原生ON/OFF桨叶、4/3/2/线几何分工、可观测项＋不可测项；无机体NN、无G通道。没有测出真实RPM/牛顿推力。</p></section>
<section><h2>逐录像：回退也保留</h2><table><tr><th>录像</th><th>原起点数</th><th>410原基线</th><th>426续踪</th><th>429参考资格</th></tr>{''.join(table)}</table><img src="mean_comparison.png"><p>11号比原基线退步；429在4号又比426退步1.71px。统一算法全段使用，没有按录像挑最优开关。429比426总体只改善0.084px，不声称显著突破。</p></section>
<section><h2>你指出的“当前明明可见却漏接”案例</h2><p>4号13.2s：原算法7个严格周期像素未过12像素冷启动门槛；原生支撑实际存在。新规则接收当前匹配位移，同时拒绝将它当电机转速。这个起点0.4秒误差117.78→74.19px；426为67.32px，429的参考点限制在这里有回退。整段当前位移观测48→75包，原10个评分起点平均59.97→47.25px。输出数量不是GT检出率。</p><p><a href="../difficult_clips_20261110/index.html">原生画面与旧困难案例</a> · <a href="../raw_observation430_20261110/reproduction.json">原RAW接入证明</a> · <a href="../reference429_20261110/results.json">所有逐点评分</a></p><p>实际RAW4共181包，未跳帧，与冻结429预测最大差0px。此项只证明实际入口复现了精度算法；不作为30FPS测速或验收。</p></section>
<section><h2>仍卡在哪里，接下来只攻什么</h2><p>当前漏接已部分修好，但当前运动／未来未知输入的延续仍不准。例如4号12.0s当前检测有效仍116.8px；15号11.4s当前定位约11px，未来0.4秒却104px，其中包含发出后变向。不能把后一类强制变成“提前猜中未来操纵”的优化目标。</p><p>已试更精细四椭圆裁剪反而变差；因果背景运动补偿也未降平均误差，均未并入入口，不继续扫参数。431独立检查“已有平面转动是否能为未来未知力先验增加信息”，原186教师、冻结旧12权重，仅2维增量：34.607→34.574px、30内56→55，4号又回退。源码／结果完整保留为可累计的研究候选，未声称已解决主要卡点，也没有因这0.034px去反复调参或重跑六段GPU。</p><p>保持原102起点评分，不删除突变点；另记录可观察机动后的滚动更新。下一步先确认当前同一材料参考点的位移／净加速度在可见性切换时有无错误，再实施有证据的修正。先平均30px，再冻结代码去保留的新录像验证；通过之后才做FPS。</p></section>
<section><h2>可运行入口与证据边界</h2><p><code>event_drone_forecast/run_rotor_observation_candidate.py</code>：原RAW、精度优先、全包处理。旧<code>run_rotor_learned_candidate.py</code>及NN发布包保留。</p><p>模型由7个非评测来源186个原生教师训练；6段评测录像整段排除拟合，但已反复暴露开发，因此不叫盲测。参考是原生轮廓中心代理，非独立COM真值。新录像只查元数据，没有解码、评测或用于调参。</p></section></html>'''
    (out/'index.html').write_text(doc,'utf8')
    handoff=f'''# 当前权威交接：419–430（2026-11-10）

本节优先于下方历史阶段。用户要求：先第2类当前可见漏接与总体0.4s平均误差30px，在新录像冻结验证通过后才优化FPS。P95只保留记录，不修IoU/细边界。未观测到的未来操纵不强求提前猜中，旧预测冻结，新观测立刻更新新t+0.4。保留所有原102起点，不删难点刷分。

## 当前可运行状态

`run_rotor_observation_candidate.py` → `ReferenceGuardedObservedPredictor`：429统一算法，精度优先unpaced、零跳包，不声称实时。旧`run_rotor_learned_candidate.py`保留410。共享模型仍`models/native_response_plane_v1.json`：186原生教师、12残差系数＋1未来驱动增益，6评测录像整段排除拟合，全部已暴露开发，弱轮廓参考非独立COM。

同102点平均36.4630267→**34.6074859px**（−{summary['improvement_percent']:.2f}%），30内51→56；426为34.6913082、30内55。原生ON/OFF桨叶、五圆4/3/2/线、可测＋未知力主线，无NN/G。**30未达，新录像未验证，FPS不开展。**

|录像|原点数|410|426|429|
|---|---:|---:|---:|---:|
{chr(10).join(md)}

11比410回退，4号429比426回退1.71px；没有按录像拼优。429总体仅比426改善0.084px，是小增量而非显著突破。额外5可输出起点独立记录，不掺原102均值。

## 具体改动与负例

-419：恢复逻辑原有regional_only_motor_frequency_unobserved标记被旧continuation丢失。新observation_roles不更新弱续踪的频率bank、不造已知推力，但接收真实当前位移。单开419同10237.038，未保留为主入口。
-420：弱观测不能把原缺测传播F/Q切成强观测recipe。仅弱帧沿用现有.1s缺测先验后更新；强帧/.35常数不动。零创新新状态差0，旧切换约274px/s²假变化。420同10236.525。
-421：完整原生轮廓教师与裁剪可见轮廓评分的观测点不同。已有完整支撑extent提供输出映射，内部p/v/a/力不变；420+421为34.927720。422更精细四conic模板35.589，未采用；按用户要求停止边界路线。
-426：严格周期为空时也可在原120ms身份期限内凭当前双极图＋双向匹配续踪，零周期不能刷新时钟、不能造RPM；冷启动12门槛不变。原10234.691308，额外5均33.491589。4号当前region52→79、位移观测48→75、有效预测60→76，不当GT召回。
-429：弱支撑可测相对位移，不够资格重新校准材料点到轮廓的偏移/完整尺寸先验；合成静止材料点＋弱框摆动，输出假移8.993635→0px。实际4号13.2误差117.7839→74.1883（426为67.3161），原10点59.9694→47.2457。
-424背景诊断表明3/4有明显背景共同位移，非独立相机姿态真值。428只用过去背景拟合未知变化，34.814384不胜426，未接入口、不可继续扫β/窗/时间常数。

实际RAW425复现421、427复现426、430复现429均181包零跳包，全部对应预测最大差0px。`output/raw_observation430_20261110/reproduction.json`是最新证据。自动日志包含耗时，未进行FPS优化/验收，不能把unpaced结果当30Hz成功。

## 仍待解决

1. 当前漏接部分解决，不等于全场景已找到桨叶。冷启动仍依赖严格周期；靠近出画的弱支撑恢复可能跟到背景，需基于真实负例评估，不把输出数量当检出率。
2. 当前速度/加速度与未来未知控制延续仍主要决定误差。4号12.0当前有效仍116.8px；15号11.4发出后转向。不得为了旧终点倒推未来命令或删除难点。
3. 431已完成：冻结原12系数＋gamma，原186教师，仅增加2个当前无头平面投影导数特征来条件化未来未知率。回放34.607486→34.573586px，30内56→55，4号47.245653→47.501954；其余均27.546606/32.063529/35.785172/32.168665/58.285818。2权重[0.0510891333,0.1710581096]，无向反号/静止保护/当前p-v-a-力不变检查通过。源码模型结果保留output/plane_transition431_20261110，不继续扫参，尚未并入实际RAW入口；不要把34.574报成实际入口精度。微小收益不能证明转动已解决主要卡点。
4. 保留新录像E:/dataset/recordings/rec_20261004_162506/event.raw，10–16s，不先看图/选好窗口。先原均值30，冻结模型/代码，再验证；还需核历史重叠。指纹不同不证明不同飞行。

报告：`output/observation_accuracy_20261110/index.html`。保护D:/yolov10-main、E:/dataset/software/yolov10-main、原RAW/SDK/权重/NN发布包未改。旧formal goal曾usageLimited不等于核验账户耗尽，新工作来自后续用户授权，不把目标标完成。

---

'''
    p=ROOT/'docs/HANDOFF_20261110_CONDITIONAL_RESPONSE.md';text=p.read_text('utf8');marker='# 当前权威交接：419–430（2026-11-10）'
    if text.startswith(marker):text=text.split('\n---\n\n',1)[1]
    p.write_text(handoff+text,'utf8')
    p=ROOT/'README.md';text=p.read_text('utf8');start='<!-- observation430-latest:start -->';end='<!-- observation430-latest:end -->'
    block=f'''{start}
> **当前原生桨叶精度候选（2026-11-10）：** 同6段、同102原起点0.4秒平均误差36.463→**34.607px**，30内51→56。新增已有桨叶身份的当前位移续接、弱观测传播一致性、弱支撑参考点资格分工；不更新隐藏RPM/可测推力。4号原RAW181包零跳包复现差0px。11号有退步，4号参考资格小修相对426有回退，逐段全部公开；无按录像选择。主线仍为ON/OFF桨叶、五圆可见性与可测/未知力，无机体NN/G。**30px未达，新录像未验证，先精度后FPS，P95仅附带记录。** 新入口`run_rotor_observation_candidate.py`，旧入口/公开NN包保留。见[当前报告](../output/observation_accuracy_20261110/index.html)与[权威交接](docs/HANDOFF_20261110_CONDITIONAL_RESPONSE.md)。以下阶段为历史。
{end}

'''
    if start in text:a=text.index(start);b=text.index(end,a)+len(end);text=text[:a]+text[b:].lstrip()
    p.write_text(block+text,'utf8')
    p=WORK/'output/checkpoints/latest_handoff.json';state=json.loads(p.read_text('utf8'))
    backup=p.with_name('handoff_before_observation430_20261110.json')
    if not backup.exists():backup.write_bytes(p.read_bytes())
    state.update(timestamp='2026-11-10',objective_completed=False,account_quota_exhaustion_verified=False,
        status='Current user-directed accuracy batch: integrated429 mean34.607px,30px/new-video unmet. FPS deferred. Old formal goal usageLimited is not an account-quota claim.',
        project_work_state='accuracy_first_user_directed_work',latest_report=str(out/'index.html'),
        candidate_entry=str(ROOT/'run_rotor_observation_candidate.py'),preserved_candidate_entry=str(ROOT/'run_rotor_learned_candidate.py'),
        current_precision_mean_px=now['mean'],current_precision_p95_px=now['p95'],aggregate_same102=now,
        candidate_scores={n:dict(result['cases'][n]['qualified429']['same_original_origins']) for n in order},
        detection_front_changed_in_this_batch=True,FPS_optimization_allowed=False,FPS_work_deferred_until_30px_and_new_video=True,
        P95_optimization_priority='descriptive_only_per_user',all_original_origins_preserved=102,
        no_retroactive_future_manoeuvre_or_point_exclusion=True,raw_latest_reproduction=parity,
        in_progress=False,experiments_running=[],background_jobs=[],running_jobs=[],
        current_user_stage='419-431 complete; actual integrated42934.607px.30px/new-video accuracy gate unmet; FPS deferred.',
        latest_user_priority='Mean0.4s position error30px then frozen new-video verification before FPS. P95 is descriptive; no boundary/IoU focus.',
        next_action=['Preserve current qualified weak displacement and rotor-only force ownership. Investigate a demonstrated current material-reference/motion bias before another change.',
            'No further conic-border, background beta, or two-column431 parameter scan.',
            'After original102 mean reaches30px, freeze code/model and verify reserved distinct RAW10-16s without tuning; only then optimize FPS.'],
        FPS_gate=dict(allowed=False,requires=['original102 mean <=30px','frozen distinct-video validation passed']),
        observation419=dict(completed=True,superseded_by='combined429',raw_reproduced430=True),
        plane_transition431=dict(mean_px=34.573585912207285,baseline429_px=now['mean'],within30=55,
            actual_RAW_entry_integrated=False,model_and_source_retained=True,parameter_scan=False))
    for key in ('native_output_counts_not_recall',):
        state[key]={}
        for n in order:
            rows=[json.loads(s) for s in (WORK/f'output/reference429_20261110/{n}.jsonl').read_text('utf8').splitlines()]
            state[key][n]=dict(frames=len(rows),current_native_rotor_region=sum(bool(r['body'].get('detected')) for r in rows),
                current_displacement_observation=sum(bool(r['motion'].get('current_observation')) for r in rows),
                valid_forecasts=sum(r['valid'] for r in rows),note='Counts, not GT recall or true visible motor count')
    p.write_text(json.dumps(state,ensure_ascii=False,indent=2),'utf8')
    p=WORK/'WORKLOG.md';log=p.read_text('utf8');heading='## 2026-11-10 419–430：按当前桨叶位移资格分工，平均34.607px'
    if heading not in log:
        p.write_text(log+'\n\n'+heading+f'''\n\n用户明确P95不作为主攻，不再修IoU/精细边界；先0.4s均值30px＋新录像冻结验证再FPS。不预知发出后才发生的操纵，保留原102起点另看新观测滚动响应。\n\n419修弱续踪频率角色丢失，真实当前位移与电机RPM/可测力分开；420修弱观测引起传播先验切换，零创新状态假变274→0。421处理完整/可见轮廓输出观测点，426允许严格周期空白但身份不超过原120ms的当前双极＋双向匹配，不能自刷新身份/更新motorbank。429弱支撑不重定义机体参考点，静止材料点支撑摆动控制假移8.99→0。代码入口run_rotor_observation_candidate.py，不换NN/G，不改186教师12+1模型。\n\n原102均36.463027→34.607486（−{summary['improvement_percent']:.2f}%），30内51→56；42634.691308，新增5输出分开。六段均27.473/32.111/36.023/32.266/58.285/47.246，11退步；429侧视4号比426又退1.71，统一算法不按录像拼优。4号当前位移48→75包、13.2误差117.78→74.19（42667.32），仍有缺口；数量不是GT召回。422四conic精细边界35.589不胜矩形观测映射，428因果背景34.814不胜426，均未采用不扫参。\n\n实际RAW425/427/430分别复现421/426/429，181包零跳包，对应预测差0。FPS未优化/验收。新视频仅元数据保留E:/dataset/recordings/rec_20261004_162506/event.raw10–16s未解码，不证明历史无重叠。30px未达。报告output/observation_accuracy_20261110，交接HANDOFF_20261110_CONDITIONAL_RESPONSE.md。保护目录/RAW/SDK/权重/NN发布包未动，无上传/安装/新代理。\n''','utf8')
    log=p.read_text('utf8');heading431='## 2026-11-10 431：当前平面转动的未知率增量，保留研究结果'
    if heading431 not in log:
        p.write_text(log+'\n\n'+heading431+'\n\n复查Torrente残差动力学、RPM-modulatedKF与原桨叶相对估计论文，既有RPM调Q不当本项目新发明。一次预先限定试验：冻结12权重/gamma，原7源186教师，仅2个无头投影导数列。全102回放34.607486→34.573586，30内56→55，4号退47.245653→47.501954，未取得主要卡点的实际突破；源码/模型/逐点结果保留，不按视频选开关、不扫描窗口或加大权重。新率仍统计先验，不把隐藏推力或世界相机运动说成测量。没有为这0.034px反复GPU验证，实际入口仍42934.607。所有本批作业已完成，checkpoint无后台会话。\n','utf8')
    print(json.dumps(summary,ensure_ascii=False))
