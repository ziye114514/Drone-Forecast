"""Publish the completed paired147 experiment and update live handoff documents."""
from pathlib import Path
from datetime import datetime,timezone
import json,html,re
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parent
OUT=WORK/'output/stage147_curvature_split_20261024'
summary=json.loads((OUT/'summary147.json').read_text());scores=json.loads((OUT/'scores147.json').read_text())
checks=json.loads((OUT/'checks147.json').read_text());controls=json.loads((OUT/'negative_controls147.json').read_text())
assert all(all(v is True or isinstance(v,int) and v>0 for v in d.values()) for d in checks.values()),checks
assert controls['single_masks']==controls['rejected']==80 and not controls['false_splits']
report=OUT/'曲率拆分粘连桨盘与预测优化.html'
fmt=lambda x:'—' if x is None else f'{x:.3f}'
tr='';lines=[]
for name,v in summary.items():
    s=scores[name];b=s['baseline']['0.4'];f=s['fused']['0.4']
    values=[name,v['frames'],f'{v["old146_states"].get("four",0)} → {v["states"].get("four",0)}',
            v['paired_used_frames'],v['orphan_frames'],f'{fmt(b["mean_px"])} → {fmt(f["mean_px"])}',
            f'{f["n"]} / 缺{f["missing"]}',f'{v["processing_median_ms"]:.1f} ms']
    tr+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in values)+'</tr>'
    lines.append(f'| {name} | {v["frames"]} | {v["old146_states"].get("four",0)}→{v["states"].get("four",0)} | {v["paired_used_frames"]} | {fmt(b["mean_px"])}→{fmt(f["mean_px"])} | {f["n"]}/缺{f["missing"]} | {v["processing_median_ms"]:.1f}ms |')
side=scores['side'];train=scores['train2'];clear=scores['clear']
selected=json.loads((OUT/'selected_frames147.json').read_text());merge=next(r for r in selected if r['case']=='clear')
recovered=[r for r in merge['new147']['geometry']['observed_rotors'] if r.get('curvature_validated')]
centres='；'.join(f'({r["cx"]:.1f}, {r["cy"]:.1f})，长轴{r["major"]:.1f}px，弧覆盖{r["visible_arc_fraction"]:.0%}' for r in recovered)
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>曲率拆分粘连桨盘与预测优化</title>
<style>body{font:17px/1.8 "Microsoft YaHei",sans-serif;background:#f4f7fb;color:#1c2f42;max-width:1370px;margin:24px auto;padding:0 20px}section{background:#fff;border-radius:12px;padding:22px;margin:18px 0}h1{font-size:30px}h2{font-size:23px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #cbd5df;padding:9px;text-align:left}img,video{max-width:100%;height:auto}figure{margin:20px 0}figcaption{font-size:15px;color:#40556a}pre{overflow:auto;background:#edf3f8;padding:15px}a{color:#2164a0}.note{border-left:5px solid #d69a20;padding:12px;background:#fff6e3}</style>
<h1>曲率拆分粘连桨盘：已经接入 RAW 预测项目</h1><p>2026-10-24 · 阶段147 · 四个窗口共1325时刻的最终成对版本；另保留首版1325时刻 · 默认预测仍为144</p>
<section><h2>改动和实际效果</h2><p>新增可选开关 <code>--rotor-curvature</code>。阈值4的连通轮廓先找多尺度凹曲率与凸包凹陷，成对切出原始弧段；复用现有椭圆拟合、边界距离、弧覆盖和中心不确定度计算。切割产生的直线不参与拟合，也不为凑四桨生成边界。两条弧分别支持椭圆、组合区域能解释原轮廓、且比单椭圆残差降低足够时才采用。</p>
<p>26.66秒的真实样例：右侧两个桨盘在阈值4下粘连；146通过阈值8取得较小的活动核心，147在原阈值4轮廓上拟合出两个接近外边界的椭圆。它们仍是部分弧段反推的中心，未升级为完整物理桨盘或已验证转轴真值。</p>
<figure><img src="comparison_merge.png"><figcaption>三列使用同一 RAW 显示帧：左为原生ON/OFF及既有基线覆盖；中为146的高阈值核心；右为147原边界拟合。绿色为新拆分椭圆，蓝点为实际用于拟合的轮廓点，红点为切分凹点，橙色为其他原有候选。图像含原预览的预测线，未用于推理或评分。</figcaption></figure>
<p>本帧新拟合：'''+html.escape(centres)+'''。这些中心没有独立轴心标注，不能把与旧核心的差异称为轴心误差下降。</p></section>
<section><h2>连续 RAW 对照</h2><p>参数用已暴露开发样例选择，随后冻结运行。下面“四桨”只是满足当前模型门槛的假设次数，并非真实桨盘检测正确率；数量增加本身不证明更准确。</p><div style="overflow:auto"><table><tr><th>窗口</th><th>时刻</th><th>146→147四桨假设</th><th>成对拆分被采用</th><th>孤立子块</th><th>0.4s基线→新融合 px</th><th>有效/缺失</th><th>处理中位</th></tr>'''+tr+'''</table></div>
<p>side、train2用既有人工框中心；clear用未来检测器框中心，只衡量自洽；fred0使用FRED原框，但仅为按目标大小选出的108–111秒短窗。不是统一基准、独立测试集或真实重心/姿态/转速真值。0.1/0.2/0.4秒的全部评分、缺失数与逐端点坐标见评分JSON。</p>
<p class="note">检测局部问题有所修复，预测收益仍不足以升为默认。侧视0.4秒：'''+fmt(side['baseline']['0.4']['mean_px'])+' → '+fmt(side['fused']['0.4']['mean_px'])+''' px；第二段：'''+fmt(train['baseline']['0.4']['mean_px'])+' → '+fmt(train['fused']['0.4']['mean_px'])+''' px。清晰段检测器参考：'''+fmt(clear['baseline']['0.4']['mean_px'])+' → '+fmt(clear['fused']['0.4']['mean_px'])+''' px。原144基线在四窗口的预测坐标最大变化均为0。</p>
<p>不能用融合回退证明推力正确。side同一批有物理预测的0.4秒起点上，基线'''+fmt(side['baseline_same_physical_origins']['0.4']['mean_px'])+' px，纯物理分支 '+fmt(side['physical']['0.4']['mean_px'])+''' px；完整记录见scores147.json。仍未完成从频率代理到稳定位置收益的闭环。处理耗时也大于20ms，视频50fps播放不代表实时计算。</p></section>
<section><h2>避免误拆和副作用</h2><p>椭圆自身两端曲率高并不是粘连证据，使用有符号凹曲率并在两种平滑尺度核对。拟合至少约173°的可观测弧、中心近似标准差≤3px、双椭圆联合区域IoU≥0.78，并检查方向/尺度相容与单模型残差降低；这些是工程门槛，不是经真值校准的置信区间。</p>
<p>首轮连续测试发现后续分组有时只选拆分后的一块：清晰段35帧、侧视6帧、FRED2帧。已修正为同源两块必须成对入模，否则恢复原父轮廓。最终四窗口孤立拆分子块均为0，首版文件和负结果保留在未加_paired的目录及unpaired_analysis。拆分失败仍沿用146高阈值局部恢复、三桨有条件补点或薄线侧视分支。</p>
<p>合成控制：9个已知双椭圆重叠中接受6个，4个基础反例和80个单椭圆缺口/噪声/裁剪反例均未误拆；部分弧不得越过FOV资格检查；成对分组回退控制通过。有限合成控制不是实飞误报率。所有最终视频全解码、推理时未读标签/旧输出、事件早于预测时刻、源码指纹匹配。</p>
<figure><img src="comparison_clear_C.png"><figcaption>人工确认可见四桨的C帧，112.02秒。新方法不要求把原本分离的四轮廓再拆分。</figcaption></figure>
<figure><img src="comparison_side_line.png"><figcaption>118.12秒侧视。凹点缺少可靠的独立桨盘弧时保留退化状态，不用拆分强行恢复四桨。</figcaption></figure></section>
<section><h2>复用来源、局限和下一步</h2><p>“凹点→弧段→分别拟合”已有研究：<a href="https://arxiv.org/abs/2008.00997">Improving concave point detection…</a>及<a href="https://github.com/explainingAI/overlapped-objects">作者代码</a>；<a href="https://arxiv.org/abs/1906.01049">Concavity Analysis and Gaussian Process</a>也从凹点提取轮廓证据。弧支撑椭圆检测可参考<a href="https://github.com/AlanLuSun/High-quality-ellipse-detection">Arc-support Line Segments作者项目</a>。本轮实现采用已安装OpenCV与本项目几何工具，没有复制外部源码或安装其MATLAB依赖，不能把凹点切分本身当作新方法。</p>
<p>当前优先改进：保持原轮廓的同源弧分组，并结合短期共面形状跟踪，减少逐帧跳变；其次解决重叠区域内的频率归属与谐波，避免两块活动区域被当作两根独立电机。完全遮挡、包含式重叠或极扁投影未必存在可用凹点，仍需侧视分支和模型不确定度。FRED本窗预测差、没有稳定物理支持，不把新增四桨假设称为成功复原。</p>
<pre>python event_drone_forecast/run_optimized_forecast.py --raw "E:/dataset/train/6/media/event.raw" --output "D:/drone forecast/output/my_curvature_run" --start 22 --duration 8 --rotor-curvature --preview</pre>
<p>输出目录必须不存在。该开关自动启用146分支，默认关闭；原始时间戳仍直接供周期引擎使用。未使用G重建；保护项目未访问/修改。</p>
<p><a href="summary147.json">运行汇总</a> · <a href="scores147.json">所有预测评分</a> · <a href="checks147.json">最终输出检查</a> · <a href="controls147.json">双椭圆控制</a> · <a href="negative_controls147.json">单椭圆与分组控制</a> · <a href="selected_frames147.json">所示帧的全部测量</a></p>
<p>最终视频：<a href="clear_paired/predictions.mp4">清晰段</a> · <a href="side_paired/predictions.mp4">侧视段</a> · <a href="train2_paired/predictions.mp4">第二录像</a> · <a href="fred0_paired/predictions.mp4">FRED0</a>。视频轨迹仍显示原144基线；新融合坐标在各目录predictions.jsonl的rotor_fused中，单独物理分支在rotor_physics中，不能把视频中的基线当作新推力预测。</p></section></html>'''
report.write_text(page,encoding='utf-8')
readme='''# 147：凹曲率拆分粘连桨盘

2026-10-24。实现 `curvature147.py`，主入口 `--rotor-curvature` 自动启用146。
原144预测不变；新增路径可选、默认关闭。使用现有OpenCV、geometry._fit/_distance/_coverage/_uncertainty，外部论文只参考思路，没有复制源码或安装新依赖。

流程：当前阈值4轮廓→双尺度有符号凹曲率+凸包凹陷→凹点对→原始开弧的确定性稳健椭圆拟合→覆盖/残差/近似中心协方差→联合支持区域IoU与相容性→同源两子块成对分组，否则恢复父轮廓。没有将切割直线作为桨叶边界，也不强制四个分量。部分弧推断边界保持mode=arc/inferred_shape=True，单独允许合格弧中心建模，不能越过FOV检查；置信度降为原0.8倍。主阈值4/侧视阈值2/原146阈值8回退均保留。

最终四窗口1325实际RAW时刻在 *_paired；首版另1325时刻在 clear/side/train2/fred0，首版分析保留unpaired_analysis；原146对照保持。原main146快照run_optimized_before147.py，首版模块快照curvature147_unpaired.py。最终hash包含主入口与147模块。观测探针probe147/bench147读取当前ROI缓存或保存轮廓，不算新增端到端RAW。

|窗口|时刻|146→147四桨假设|成对拆分采用帧|0.4s基线→融合px|有效/缺失|处理中位|
|---|---:|---|---:|---|---|---|
'''+ '\n'.join(lines)+'''

四桨次数不是独立检测正确率。clear仅未来检测器自洽；side/train2人工框、FRED提供框均非COM/RPM/姿态真值。预测得分不支持将147升为默认；侧视物理分支仍逊于相同起点的144。FRED0仅按大目标选窗，不能外推整个数据集。

新增控制：9双椭圆接受6，4基础反例与80单椭圆反例无误拆；FOV/孤立子块回退控制通过。首版发现清晰35、侧视6、FRED2帧只取一个子块，最终每窗口均0。没有独立转速/姿态/轴心真值。

运行：`.venv-event311/Scripts/python.exe event_drone_forecast/run_optimized_forecast.py --raw "E:/dataset/train/6/media/event.raw" --output "D:/drone forecast/output/new147" --start 22 --duration 8 --rotor-curvature --preview`

复跑整个实验用batch147.py --suffix 一个新后缀（GPU串行）；分析用analyse147.py --suffix 相同后缀；controls147.py单独做几何控制。不要在GPU worker运行期间开另一套NN/大型分析进程。推理入口不读标签/旧输出。主预览视频只绘原144基线，新融合坐标看JSONL；报告图为同帧测量重叠。

来源：[凹点检测](https://arxiv.org/abs/2008.00997)、[作者项目](https://github.com/explainingAI/overlapped-objects)、[凹点轮廓分组](https://arxiv.org/abs/1906.01049)、[弧支撑椭圆项目](https://github.com/AlanLuSun/High-quality-ellipse-detection)。切分本身不声称创新。

接下来：结合短期共面结构，对同源原始弧分组稳定跟踪；优先解决重叠区域频率归属与谐波，不把活动组当电机ID。保留144默认，146/147实验开关分开。无运行中作业。
'''
(ROOT/'research/stage147/README.md').write_text(readme,encoding='utf-8')
worklog=WORK/'WORKLOG.md'
entry='''

## 152. 147：按凹曲率拆分粘连桨盘，修正分组孤立子块

2026-10-24。用户要求依据粘连处曲率异常拆分并优化，已实际实现并重跑RAW。
报告：[曲率拆分与预测](<'''+report.as_posix()+'''>)；实现/复跑：[stage147](<'''+(ROOT/'research/stage147/README.md').as_posix()+'''>)。

**代码改动**：新增curvature147.py，可选`--rotor-curvature`自动启用146分支。双尺度凹曲率+凸包凹陷找切点→不含人工切线的原始弧段→复用已有AMS拟合/边界距离/覆盖/近似不确定度→区域联合IoU与双模型残差验证。每轮廓最多6对、每弧16个固定随机假设，优质单模型提前结束。已保存轮廓重放的新增拆分耗时约0.4–22ms，不是端到端速度。146原模块未修改，默认144未变，G及yolov10-main未访问/修改，无新依赖。

**具体修复**：26.66s的右侧粘连由阈值4原始弧段拟合成两椭圆，不再只依赖阈值8缩小核心。部分弧边界始终mode=arc/inferred_shape=True，单独降低置信度准许合格中心参加几何，FOV越界仍拒绝。首版分组会只选一个子块（clear35/side6/FRED2帧）；修正同源两子块必须成对入组，否则恢复原父候选。最终四窗口均无孤立子块；首版目录、快照和负结果保留。

**真实RAW**：首版1325+成对最终1325=本轮2650时刻；最终clear 400、side 475、train2 300、FRED0 150。参数冻结、GPU串行，RAW入口不读标签/缓存，视频全解码、时序及全部源指纹通过；所有原144基线坐标相对146最大变化0。

|窗口|时刻|146→147四桨假设|成对拆分采用帧|0.4s基线→融合px|有效/缺失|处理中位|
|---|---:|---|---:|---|---|---|
'''+ '\n'.join(lines)+'''

四桨是算法假设数，不是人工正确率。side/train2参考为人工框；clear为未来检测器自洽；FRED原框只选一个大目标窗，预测依然差。纯物理分支必须看同起点对照，side0.4s基线'''+fmt(side['baseline_same_physical_origins']['0.4']['mean_px'])+'px，物理'+fmt(side['physical']['0.4']['mean_px'])+'''px；不证明推力预测提升，不升为默认。不是稳定20ms实时。

**控制**：9双椭圆接受6；4基础反例、80单椭圆缺口/噪声/裁剪均无误拆；FOV和分组回退通过；有限合成不等实飞误报率。外部来源：arXiv2008.00997+作者explainingAI/overlapped-objects、1906.01049、AlanLuSun/High-quality-ellipse-detection；本次独立实现并复用本地工具，凹点分割本身非新。完全包含/重叠及极扁投影未必有可观测凹点。

**交接/下一步**：147保持可选；先用短期共面结构稳定同源弧分组，再处理重叠区域转速归属和谐波。最终目录*_paired；全部scores147/checks147/selected_frames147有端点、缺失与弧来源。旧146默认开关行为保留；原main146在run_optimized_before147.py，首版147模块在curvature147_unpaired.py。现有旧归档zip不含143–147，不能称已打包本轮。无运行中作业。
'''
old=worklog.read_text(encoding='utf-8')
if '## 152. 147：' not in old:worklog.write_text(old+entry,encoding='utf-8')
prefix='''## 147：曲率拆分粘连桨盘（2026-10-24）

[最新报告](<'''+report.as_posix()+'''>) · [实现与复跑](<'''+(ROOT/'research/stage147/README.md').as_posix()+'''>)。
新增`--rotor-curvature`：原阈值4凹点分割、原始弧拟合、同源子块成对入模。最终四窗口1325RAW时刻，另保留初版1325；局部粘连已能拆开，但预测收益不足、未升默认。当前默认仍138几何＋140运动＋144阻尼；146/147为可选实验，尚未证明独立推力收益或实时。详见WORKLOG§152。以下为历史。

'''
for path in [WORK/'PROJECT_STATUS_20261022.md',ROOT/'README.md']:
    old=path.read_text(encoding='utf-8')
    if not old.startswith('## 147：'):path.write_text(prefix+old,encoding='utf-8')
handoff=WORK/'output/checkpoints/latest_handoff.json';state=json.loads(handoff.read_text(encoding='utf-8'))
state.update(utc=datetime.now(timezone.utc).isoformat(),execution_status='checkpoint_ready',last_completed_stage=147,
    latest_report=str(report),previous_report=str(WORK/'output/stage146_threshold_branches_20261024/阈值抑制与四三二桨预测实验.html'),
    latest_checks=str(OUT/'checks147.json'),algorithm_stage=144,optional_rotor_branch_stage=146,optional_curvature_stage=147,
    latest_full_forecast_stage=147,latest_full_forecast_report=str(report),latest_diagnostic_stage=147,
    latest_RAW_geometry_stage=147,experimental_RAW_stage=147,in_progress=[],experiments_running=[],running_jobs=[],
    stable_high_accuracy_realtime_achieved=False,independent_RPM_gain_proven=False,
    optimization_this_stage_actual_RAW_frames=2650,optimization_current_full_frames=1325,
    latest_stage147=dict(final_suffix='_paired',summary=summary,scores_file=str(OUT/'scores147.json'),
                         baseline_144_preserved=True,real_hub_RPM_pose_truth=False,default_promoted=False),
    next_action='147保持可选，结合短期共面结构稳定同源弧分组，再解重叠区域周期归属/谐波；144默认保持。')
if 147 not in state['completed']:state['completed'].append(147)
state['latest_completed_stages']=[147]
state['counts_by_stage']['147']=2650
handoff.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
relative=report.relative_to(WORK/'output').as_posix()
(WORK/'output/latest_optimized_forecast.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='+relative+'"><a href="'+relative+'">打开曲率拆分与预测结果</a>',encoding='utf-8')
# Check only local links and media belonging to the new report.
missing=[]
for target in re.findall(r'(?:href|src)="([^"]+)"',page):
    if not target.startswith(('https:','http:','#')) and not (OUT/target).exists():missing.append(target)
assert not missing,missing
print(json.dumps(dict(report=str(report),local_links_ok=True,worklog='152',running_jobs=[]),ensure_ascii=False))
