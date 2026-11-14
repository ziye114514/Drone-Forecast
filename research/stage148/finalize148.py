"""Publish observations and negative forecast results without promoting them."""
from pathlib import Path
from datetime import datetime,timezone
import json,html,re,shutil
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent;OUT=WORK/'output/stage148_temporal_rotors_20261024'
summary=json.loads((OUT/'summary148.json').read_text());scores=json.loads((OUT/'scores148.json').read_text());checks=json.loads((OUT/'checks148.json').read_text());controls=json.loads((OUT/'controls148.json').read_text())
assert all(controls.values())
assert all(all(v is True or type(v)is int and v>0 for v in c.values()) for c in checks.values())
report=OUT/'原弧引导与跨帧桨盘跟踪实验.html'
fmt=lambda x:'—' if x is None else f'{x:.3f}'
table='';mdrows=[];score_rows=''
for name,v in summary.items():
    s=scores[name];b=s['baseline']['0.4'];new=s['fused148']['0.4']
    match=f'{v["matching"]}/{v["matching"]+v["new_observations"]}'
    vals=[name,v['frames'],f'{v["guided_arc_used_frames"]}/{v["fragment_merged_frames"]}',match,f'{v["conditional_feature_ticks"]}/{v["previous147_conditional_ticks"]}',
          f'{v["longest_continuous_feature_s"]:.2f}s',f'{fmt(b["mean_px"])} → {fmt(new["mean_px"])}',f'{new["n"]}/缺{new["missing"]}',f'{v["processing_median_ms"]:.1f}ms']
    table+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in vals)+'</tr>'
    mdrows.append('| '+' | '.join(map(str,vals))+' |')
    for h in ('0.1','0.2','0.4'):
        vals=[name,h,fmt(s['baseline'][h]['mean_px']),fmt(s['previous147'][h]['mean_px']),fmt(s['fused148'][h]['mean_px']),fmt(s['bias_only_fused'][h]['mean_px']),s['fused148'][h]['n'],s['fused148'][h]['missing']]
        score_rows+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in vals)+'</tr>'
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原弧引导与跨帧桨盘跟踪实验</title>
<style>body{font:17px/1.8 "Microsoft YaHei",sans-serif;color:#203449;background:#f3f6fa;max-width:1380px;margin:25px auto;padding:0 20px}section{background:white;padding:24px;margin:18px 0;border-radius:12px}h1{font-size:30px}h2{font-size:23px}img,video{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #c7d3de;padding:8px;text-align:left}.scroll{overflow:auto}.note{padding:15px;background:#fff3dd;border-left:5px solid #cb9626}code,pre{background:#eef3f7}pre{padding:14px;overflow:auto}a{color:#2263a1}</style>
<h1>从原始弧段到连续桨盘轨迹：新增148实验</h1><p>2026-10-24 · 三轮共3975个端到端RAW时刻，最终*_joint 1325时刻；另550时刻观测探针不算端到端 · 默认144保留</p>
<section><h2>已经实际解决了哪一部分</h2><p>把近期四桨模型前移到轮廓提取：先独立做147凹点拆分；失败时，近期可信四桨模型仅用于将当前轮廓点分配给可能的两片原始弧。两弧分别满足覆盖、残差、中心近似不确定度、组合区域解释和先验相容性后才拟合；仍不将历史位置当成新观测。四桨模型超过120ms、缺当前弧支持或已退化成线时拒绝该恢复。</p>
<p>除粘连拆分，还新增同桨碎块归并：多个实际分量的外弧若共同支持近期模型中的同一椭圆，且合并后的轮廓覆盖、拟合不确定度及区域解释通过，才合成一个部分弧候选。26.84秒旧147也报告四桨，却把左下桨盘分成两个小候选、把右侧两桨保留成一个大候选（该帧同为阈值4）；最终148将左下碎块合回一个椭圆、右侧粘连拆成两个。不能用四桨数量相同就认为几何同样正确，当前仍无独立轴心真值。</p>
<figure><img src="arcs_guided_comparison.png"><figcaption>同一真实26.84秒画面：左为ON/OFF显示，中为147原候选，右为148拆分与归并；绿色椭圆为新引导结果，蓝点为当前真实支持点。显示来源是同一已输出预览帧，不参与推理或评分。</figcaption></figure>
<figure><img src="prior_guided_arcs.png"><figcaption>完整输出中，青色为局部视觉轨迹ID；橙色为本轮实际融合预测，绿色为144基线。</figcaption></figure>
<p>身份关联直接复用本地RotorTrack及SciPy匹配，加入机体平移、形状门槛和歧义拒绝，滤波修正量有上限。只输出当前测得的桨盘，丢失轨迹不补成“当前四桨”；连续机体丢失超过0.5秒重建身份段。视觉ID会断裂，不是物理电机编号，也不能承诺一帧建模后永不丢失。</p>
<p>周期继续使用已装SDK的原始时间戳引擎。仅在不重叠的像素上统计，每个稳定ID累积80ms短期证据；至少两次一致、当前至少4个支持像素才输出。当前没有证据就拒绝，不用历史频率冒充当前测量；半/倍频转换至少三帧支持，但仍未辨识真实基频、桨叶数或机械RPM。</p></section>
<section><h2>连续RAW结果与预测</h2><div class="scroll"><table><tr><th>窗口</th><th>时刻</th><th>引导拆分/碎块归并帧</th><th>续接/当前观测</th><th>条件特征148/147</th><th>最长连续特征</th><th>0.4s基线→试验px</th><th>有效/缺失</th><th>处理中位</th></tr>'''+table+'''</table></div>
<p>续接比例是算法自身关联统计，没有轴心ID真值，不能叫跟踪正确率。条件特征变少包含拒绝当前证据不足、身份歧义或谐波候选的作用，未证明筛掉的一定是错误信号。角速度与推力正确性仍缺独立验证。</p>
<p class="note">预测试验未通过升级标准。0.4秒人工框参考：侧视'''+fmt(scores['side']['baseline']['0.4']['mean_px'])+' → '+fmt(scores['side']['fused148']['0.4']['mean_px'])+'px；第二录像'+fmt(scores['train2']['baseline']['0.4']['mean_px'])+' → '+fmt(scores['train2']['fused148']['0.4']['mean_px'])+'''px，均略差。所有窗口、所有预测时域的新融合都与“纯位置偏差校正”消融完全一致，事件特征没有获得独立权重。没有将这次残差预测升为默认。</p>
<img src="forecast148.png"><p>图中空心绿点为人工框中心，灰点为144预测，橙点为148试验预测。框中心不是物理重心。clear仅未来检测器自洽；FRED仅按大目标选的短窗，误差仍大、缺失多，不能代表全FRED基准。</p>
<div class="scroll"><table><tr><th>窗口</th><th>预测秒</th><th>144基线px</th><th>147融合px</th><th>148试验px</th><th>纯偏差消融px</th><th>有效</th><th>缺失</th></tr>'''+score_rows+'''</table></div></section>
<section><h2>为什么仍卡在推力到位置</h2><p>旧的绝对加速度模型受到未知投影尺度、扰动与特征间歇影响。本轮另做“基线预测+像素残差”模型：每个时域只用自身已到期目标校正；偏差校正与事件增量分开。只有事件模型在相同历史起点上同时胜过基线和偏差消融至少5%，才允许获得小权重。全部四窗口未满足该条件，不能说推力帮助了预测。</p>
<p>当前问题已具体到两个环节：清晰段的居中事件特征变化小，未显示额外预测信息；侧视合格特征短且断续，无法提供连续可靠的学习输入。重叠像素在本轮仍选择排除，没有完成多电机解混。对身份延续和平滑的改进，并不会自动补出这些缺失信息。</p>
<p>下一步应围绕“连续原弧—转速/相位一致性”继续：在近期四桨模型约束下直接跟踪原弧证据，解决每次轮廓合并后的身份重建；用公开独立周期真值验证半/倍频及混合信号处理；再处理投影尺度变化下的像素动力学增益。不能仅降低门槛、增加四桨假设或混合权重来宣布成功。当前默认仍138几何＋140运动＋144阻尼。</p></section>
<section><h2>复用、运行与可检查文件</h2><p>复用现有RotorTrack、椭圆工具、147凹点分割、79 SDK周期引擎；不安装新依赖，不访问或修改yolov10-main，不使用G灰度重建。<a href="https://arxiv.org/abs/2610.31384">CoHelix</a>的共享几何/独立旋转状态思路与本方向相关；本实现只是二维近期模型引导和频率一致性，不是其完整单应/相位滤波器。<a href="https://arxiv.org/abs/2604.18289">原始相对状态估计论文</a>也采用桨区关联和周期输入。<a href="https://github.com/JackPieCZ/EE3P">EE3P</a>及本地EEPPR可作后续独立周期验证来源，本轮未把已测试过的作者数据当成新实飞结果。</p>
<pre>python event_drone_forecast/run_optimized_forecast.py --raw "E:/dataset/train/6/media/event.raw" --output "D:/drone forecast/output/new148_trial" --start 22 --duration 8 --rotor-temporal --preview</pre>
<p>上面是复跑研究试验，不是推荐的精度升级开关；要用现有已保留的默认预测，去掉--rotor-temporal。输出目录必须不存在。新JSONL包含rotor_fused（试验融合）、rotor_bias_fused（纯偏差消融）、rotor_temporal_absolute（原绝对模型对照）；rotor_physics在148里是物理特征引导的像素残差预测，不能按字面当作牛顿推力模型。</p>
<p><a href="summary148.json">轨迹、覆盖与特征统计</a> · <a href="scores148.json">全部端点与消融评分</a> · <a href="checks148.json">源码/时序/视频检查</a> · <a href="controls148.json">12项关键控制</a> · <a href="pilot_summary148.json">早期观测探针（非端到端）</a>。首版clear/side/train2/fred0与postfit_only_analysis、第二版*_guided与guided_split_only_analysis保留，最终为*_joint。所有原144预测坐标最大变化0。</p>
<p>最终视频：<a href="clear_joint/predictions.mp4">清晰</a> · <a href="side_joint/predictions.mp4">侧视</a> · <a href="train2_joint/predictions.mp4">第二录像</a> · <a href="fred0_joint/predictions.mp4">FRED0</a>。50fps播放不等于实时计算。</p></section></html>'''
report.write_text(page,encoding='utf-8')
readme='''# 148：原始弧段引导、局部视觉ID、周期一致性及残差预测试验

2026-10-24。研究开关`--rotor-temporal`自动启用147/146；默认144未变。**新预测未通过升级标准，勿把本开关当作已证实的精度优化。**

模块：guided148.py在独立凹点拆分失败后，用最近120ms且至少3次观测、尺度相容的四桨假设为当前轮廓分配弧段；必须有当前至少173°弧、受限拟合残差/中心不确定度、联合区域解释，不能输出历史桨盘。另将共同支持同一椭圆的当前碎块归并，使用与候选一致的110/124开运算分量，联合区域IoU≥.72；独立小块拒绝。保留147同源两子块成对规则，归并出的单桨另有来源标记。
temporal148.py复用RotorTrack与SciPy关联，加入机体平移、尺寸与歧义门槛；有界平滑，漏检轨迹不作为当前观测；侧视活动组与桨盘分族，机体丢失0.5s重建身份段。ID只在连续段内有效，不是物理电机号。
频率复用79 SDK：唯一像素、排除重叠、80ms短期一致、当前支持、半/倍频转变3帧确认；不能称机械RPM已正确恢复或多电机信号已分离。
residual148.py：逐预测时域已到期目标更新“纯偏差+居中事件增量”，事件只有在相同过去起点胜过基线及纯偏差5%才获权重。所有四窗口事件分量均未获权重，融合等于纯偏差消融，0.4秒两人工窗口均略差，未升默认。

最终真实RAW1325时刻位于*_joint；首轮1325在clear/side/train2/fred0，其分析保存在postfit_only_analysis；第二轮1325在*_guided及guided_split_only_analysis，第二轮guided源码另存guided148_split_only.py。另550时刻pilot复用当前147框、只作观测探针，不算端到端。三轮协议均记录源指纹；首轮源码另存temporal148_postfit_only.py/run_optimized148_postfit_only.py；最终源码快照保存在本输出目录。

|窗口|时刻|引导拆分/碎块归并帧|续接/当前|条件特征148/147|最长连续|0.4s基线→试验px|有效/缺失|处理中位|
|---|---:|---:|---|---|---|---|---|---|
'''+ '\n'.join(mdrows)+'''

clear仅未来检测器自洽；side/train2人工框中心，非重心/RPM/姿态真值；FRED提供框且仅按大目标选窗。续接是算法自身统计，无身份真值不能称正确率。原基线坐标保持0最大差；最终视频全解码和源/当前证据/事件cutoff检查通过。

复跑：`.venv-event311/Scripts/python.exe event_drone_forecast/run_optimized_forecast.py --raw 你的RAW --output 新目录 --start 22 --duration 8 --rotor-temporal --preview`。研究试验而非已证明升级；正常默认不加此开关。整组用batch148.py --suffix 新后缀，分析analyse148.py --suffix 相同后缀。GPU串行，推理时不读旧输出/标签；分析阶段再读人工框。main预览橙色是真实rotor_fused，绿色144，青色ID。

关键控制12项见controls148.json：身份顺序/机体位移、同时间回退不重复计数、漏检不补测、当前周期与短期一致、三帧谐波转换、完全重叠拒绝、失效轨迹重置、静止机体与桨面波动、偏差消融归属、无目标不更新、先验不得替代当前弧段、同桨碎裂归并与无关小块拒绝。

来源：[CoHelix](https://arxiv.org/abs/2610.31384)、[原始论文](https://arxiv.org/abs/2604.18289)、[EE3P](https://github.com/JackPieCZ/EE3P)。复用本地实现，不声称完整作者系统复现或新算法原创。

下一步：先做连续原弧证据关联和转速/相位一致性，处理合并后身份重建；对照独立公开周期真值，再解投影尺度变化。保留未证实/负结果，不因拟合数增多声称推力预测有效。
'''
(ROOT/'research/stage148/README.md').write_text(readme,encoding='utf-8')
entry='''

## 153. 148：原弧模型引导与连续关联落地，残差预测未采用

2026-10-24。用户要求自行解决后续卡点。本轮实际增加guided148.py、temporal148.py、residual148.py，主入口新增可选`--rotor-temporal`。报告：[原弧引导与跨帧桨盘跟踪](<'''+report.as_posix()+'''>)；复跑：[stage148 README](<'''+(ROOT/'research/stage148/README.md').as_posix()+'''>)。默认仍138+140+144，147源码未修改。

先查CoHelix2610.31384、2604.18289及EE3P作者项目，复用本地RotorTrack、SciPy匹配、147真实弧段及79 SDK。首次只在拟合后平滑关联，无整体预测收益；随后把近期四桨假设前移到原弧分组，新增120ms期限、至少3观测、当前弧覆盖/残差/区域解释与FOV门槛，不能以先验补观测。第二轮引导已生效，但26.84s旧147虽报四桨，实际是同阈值4下右侧未拆、左下碎成两块；因此最终同时做粘连拆分和同桨碎块归并。归并必须使用与候选一致的110/124真实开运算分量，不能误用闭运算父轮廓。控制及真实图像中，26.84s得到约70–84px长轴的四个候选，仍非独立轴心真值。

**当前证据关联/周期**：机体平移与形状门槛、有界滤波和歧义拒绝；当前只输出可见观测，0.5s机体失联重建ID段，视觉ID非物理电机。周期在独占像素上积累80ms一致，至少2帧且当前≥4像素，零当前支持拒绝，半/倍频转换3帧确认；未解决基频/桨叶数或混合信号本身。先验引导未提升本轮四桨假设数。

**预测试验**：逐0.1/0.2/0.4s成熟目标学习“基线+纯偏差+居中事件增量”。事件需在相同过去起点同时胜过基线和纯偏差5%方能获权重；最终所有窗口均无独立事件权重，融合与纯偏差消融逐坐标相同。不能把位置校正叫推力增益；0.4s两个人工窗口均略差，不升默认。

|窗口|时刻|引导拆分/碎块归并帧|续接/当前|条件特征148/147|最长连续|0.4s基线→试验px|有效/缺失|处理中位|
|---|---:|---:|---|---|---|---|---|---|
'''+ '\n'.join(mdrows)+'''

clear为未来检测器自洽，side/train2为人工框中心，FRED官方框只选单窗。续接无ID真值，不是正确率。最终每窗口原144坐标最大变化0，源码指纹、事件cutoff、当前频率证据和视频全解码通过。关键控制12项通过，含机体不动桨面波动不自行制造平移、无目标不更新、先验不能生成隐藏桨盘、同桨碎块归并与独立小块拒绝。

**运行范围**：三轮各1325＝本轮3975实际端到端RAW时刻；另550时刻开发探针复用当前检测框，不算新增NN/端到端。首轮目录clear/side/train2/fred0与postfit_only_analysis、第二轮*_guided与guided_split_only_analysis保留；最终*_joint。源码快照与每次protocol保留。新预览已绘真实试验融合橙线，绿线144、青色局部ID。无G，无新依赖，保护项目未访问/修改，E盘RAW只读，无运行中作业。旧归档zip不含本轮。

**交接**：已修复模型只在拟合之后起作用的问题，但连续物理特征仍不足，未解决稳定推力预测。下一步围绕合并后的原弧身份与转速/相位一致性，公开周期独立真值，再处理投影尺度变化；不要继续只加融合权重或改成“多拟合几桨=成功”。148研究开关不是推荐精度升级；144继续为默认。
'''
log=WORK/'WORKLOG.md';existing=log.read_text(encoding='utf-8')
if '## 153. 148：' in existing:
    start=existing.index('\n\n## 153. 148：');tail=existing.find('\n## ',start+3)
    existing=existing[:start]+entry+(existing[tail:] if tail>=0 else '')
else:existing+=entry
log.write_text(existing,encoding='utf-8')
prefix='''## 148：原弧引导与连续关联（2026-10-24）

[本轮报告](<'''+report.as_posix()+'''>) · [实现与复跑](<'''+(ROOT/'research/stage148/README.md').as_posix()+'''>)。
新增`--rotor-temporal`研究试验：近期四桨假设引导当前原弧、粘连拆分与碎块归并、轨迹关联、独占周期一致性、逐时域残差及纯偏差消融。三轮3975实际RAW时刻；最终1325在*_joint。观测处理已生效，但预测试验未通过，未获独立事件增益；默认仍138＋140＋144。详见WORKLOG§153。以下为历史。

'''
for path in (WORK/'PROJECT_STATUS_20261022.md',ROOT/'README.md'):
    existing=path.read_text(encoding='utf-8')
    if existing.startswith('## 148：'):
        end=existing.find('\n## ',1);existing=existing[end+1:] if end>=0 else ''
    path.write_text(prefix+existing,encoding='utf-8')
handoff=WORK/'output/checkpoints/latest_handoff.json';state=json.loads(handoff.read_text(encoding='utf-8'))
state.update(utc=datetime.now(timezone.utc).isoformat(),execution_status='checkpoint_ready',last_completed_stage=148,
    latest_report=str(report),previous_report=str(WORK/'output/stage147_curvature_split_20261024/曲率拆分粘连桨盘与预测优化.html'),
    latest_checks=str(OUT/'checks148.json'),latest_full_forecast_stage=148,latest_full_forecast_report=str(report),
    experimental_RAW_stage=148,algorithm_stage=144,optional_temporal_stage=148,latest_RAW_geometry_stage=148,
    latest_completed_stages=[148],in_progress=[],experiments_running=[],running_jobs=[],
    optimization_this_stage_actual_RAW_frames=3975,optimization_current_full_frames=1325,
    stable_high_accuracy_realtime_achieved=False,independent_RPM_gain_proven=False,
    latest_stage148=dict(final_suffix='_joint',summary=summary,scores_file=str(OUT/'scores148.json'),
        default_promoted=False,forecast_trial_rejected=True,event_gain_over_bias_ablation=False,real_RPM_pose_ID_truth=False),
    next_action='148只作观测/研究开关；先解决原弧合并后的身份与相位/周期一致性，用公开独立周期真值，再解投影尺度；144仍默认。')
if 148 not in state['completed']:state['completed'].append(148)
state['counts_by_stage']['148']=3975
handoff.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
relative=report.relative_to(WORK/'output').as_posix()
(WORK/'output/latest_optimized_forecast.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='+relative+'"><a href="'+relative+'">打开原弧引导与真实预测对照</a>',encoding='utf-8')
for name in ['temporal148.py','residual148.py','guided148.py']:shutil.copyfile(ROOT/'research/stage148'/name,OUT/(name[:-3]+'_final.py'))
shutil.copyfile(ROOT/'run_optimized_forecast.py',OUT/'run_optimized148_final.py')
for target in re.findall(r'(?:href|src)="([^"]+)"',page):
    if not target.startswith(('http:','https:','#')):assert (OUT/target).exists(),target
print(json.dumps(dict(report=str(report),worklog_section=153,local_links_ok=True,default_algorithm=144,running_jobs=[]),ensure_ascii=False))
