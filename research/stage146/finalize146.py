"""Record final outcomes without replacing prior experiments or pretending success."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]; WORK = ROOT.parent
OUT = WORK/'output/stage146_threshold_branches_20261024'
summary = json.loads((OUT/'summary146.json').read_text(encoding='utf-8'))
scores = json.loads((OUT/'scores146.json').read_text(encoding='utf-8'))

def metric(case, key='fused', horizon='0.4'):
    q = scores[case][key][horizon]
    return f'{q["mean_px"]:.3f} px（n={q["n"]}，缺失{q["missing"]}）' if q['mean_px'] is not None else '无可评分预测'

table = '| 窗口 | 帧数 | 四桨/三桨补点/二桨假设 | 拆分恢复 | 条件推力 | 融合启用 | 处理中位 |\n|---|---:|---|---:|---:|---:|---:|\n'
for name, q in summary.items():
    s = q['states']
    table += f'| {name} | {q["frames"]} | {s.get("four",0)}/{s.get("three_completed",0)}/{s.get("two_side",0)} | {q["split_recovery_ticks"]} | {q["conditional_thrust_ticks"]} | {q["fused_active_ticks"]} | {q["processing_median_ms"]:.1f} ms |\n'
detail = f'''2026-10-24。按用户“ON/OFF阈值抑制→四/三/二桨分支→周期推力代理→未来位置”实际改代码并跑RAW；报告：[阈值抑制与四三二桨预测实验](<D:/drone forecast/output/stage146_threshold_branches_20261024/阈值抑制与四三二桨预测实验.html>)。

**实现**：新 `research/stage146/branches146.py`，主入口新增 `--rotor-branches --rotor-threshold 4`（默认不启用新物理分支）。复用110/124合并保护、61先判线与79已装SDK周期引擎；不使用G，不访问/修改yolov10-main。FRED从 `E:/FRED/0.zip` 仅提取RAW和框到本轮input/fred0，推理进程不读标注/缓存；GPU串行。

- 阈值单位是20ms内经原邻域平滑的min(ON,OFF)计数，非0—255灰度。主阈值4；三轮廓时尝试8，只在四个实际高阈值轮廓能映射成原三块的1/1/2局部分裂时采用，未按k=4强制聚类；薄线回退阈值2且仅允许通过细长/方向一致检查的二分量。高阈值轮廓是活动核心，**并不保证完整物理桨盘边界**。
- 四点不强求图像平行四边形：共面矩形在透视下可成梯形，使用对角线交点；椭圆中心仍只是物理轴心近似。C112.02s从低阈值左侧粘连，变为阈值4的四个分离候选。26.66s的双桨粘连先被误作三桨补点，检查图片后修正为高阈值四个独立分量；旧版本及负结果保留。
- 三桨：近期模型匹配+局部仿射更新，检查预测椭圆尺寸一致性；无历史模型时只在三个非共线、形状相容的椭圆能提供弱透视矩形度量时补点。补出的点**不作为观测**，不虚构第四转速。形状不相容（可能双桨粘连）拒绝补点。
- 二桨：两活动组细长且轴线与中心连线一致后输出无向法线；两点本身不唯一确定3D平面。活动组可能混有四电机，不认为已辨识两根物理转轴。合推力沿共法线；转速差主要产生力矩，不直接沿桨心连线转向。
- 周期基于原始时间戳，唯一像素投票并排除多椭圆重叠像素。仅输出条件Σf²相对推力代理；缺转速可短时保持120ms或明确等速假设，**无独立RPM或牛顿单位推力真值**。有效像素增益/综合扰动由已到期0.2s位置残差更新。0.1/0.2/0.4s各自用已到期预测误差定权重，不能用0.2s表现为0.4s背书；上限25%×几何置信度。

**最终真实RAW四窗口1325时刻**（目录 *_final；旧版两轮在原目录保留；本轮全部主入口真实RAW共3355时刻）。表中是算法合格假设次数，**不是人工桨盘正确率**。

{table}
**预测结果**（发射坐标、同一参考）：side4基线 {metric('side4','baseline')} → 融合 {metric('side4')}；train2基线 {metric('train2_4','baseline')} → 融合 {metric('train2_4')}。两条人工参考的原基线坐标相对144最大差均为0。

清晰段只有未来检测器参考：基线 {metric('clear4','baseline')} → 融合 {metric('clear4')}，不能称独立准确率。FRED0选较大目标108–111s：基线 {metric('fred0_4','baseline')} → 融合 {metric('fred0_4')}；只有一个短窗口，覆盖/精度均不足，非FRED全基准。

**结论与保留**：ON/OFF抑制、分支与局部拆分已经落地，但本轮没有预测精度收益。侧视物理分支的15个同起点上，0.4s基线32.845px，纯条件推力52.722px，更差；融合总体持平包含门控回退的作用，不能当作推力估计正确的证据。默认仍138几何＋140运动＋144阻尼；146保持可选。scores146.json列出物理分支与**相同起点基线**，防止只挑容易帧造成假增益。source指纹、因果/几何控制和视频检查单独保存。

**局限/下一步**：优先把阈值产生的活动核心扩回原始弧段，处理双桨粘连、单桨碎裂和转轴身份；再处理两活动组多电机频谱混合及差分转速/姿态变化。真实四电机总推力、质量/cf/风速不能从当前二维观测分别声称已恢复。0/1分量在真实检测中确实出现，按观测不足处理，不把四旋翼物理上有四桨当作四个实测分量。没有运行中的GPU/RAW作业。
'''
readme = '''# 146：ON/OFF阈值、四/三/二桨分支实验\n\n'''+detail+'''
复跑（输出目录必须不存在；现有3.11 SDK环境）：

```powershell
& 'D:/drone forecast/.venv-event311/Scripts/python.exe' 'D:/drone forecast/event_drone_forecast/run_optimized_forecast.py' --raw 'E:/dataset/train/6/media/event.raw' --output 'D:/drone forecast/output/my_rotor_branch_run' --start 110 --duration 9.5 --rotor-branches --rotor-threshold 4 --preview
```

`optimized_geometry` 是既有默认基线；`rotor_physics` 是条件推力模型；`rotor_fused` 是按各时域过去误差小权重融合。`rotor_branch` 内保留实测/补点、阈值来源、频率保持/等速假设及分支退化原因。旧protocol的forecasts字段列基线三项，新增预测由rotor_branches=true和逐行额外键标识；completion.only_three_forecasts=false。基线API/旧54入口未替换。
'''
(ROOT/'research/stage146/README.md').write_text(readme, encoding='utf-8')
worklog = WORK/'WORKLOG.md'; old = worklog.read_text(encoding='utf-8')
marker = '146：ON/OFF阈值与四三二桨分支落地'
if marker not in old:
    headings = [int(v) for v in re.findall(r'^## (\d+)\.', old, flags=re.M)]
    number = max(headings, default=150)+1
    worklog.write_text(old+f'\n\n## {number}. {marker}\n\n'+detail, encoding='utf-8')
intro = '''## 146：阈值抑制与四／三／二桨实验（2026-10-24）

[最新实验报告](<D:/drone forecast/output/stage146_threshold_branches_20261024/阈值抑制与四三二桨预测实验.html>) · [实现与真实结果](<D:/drone forecast/event_drone_forecast/research/stage146/README.md>)。
当前推荐入口默认仍为138几何＋140运动＋144阻尼；新增146仅通过`--rotor-branches`启用，不能声称已用转速/推力提高预测。阈值4／粘连拆分8／薄线2，三桨补点与二桨侧视均保留条件和歧义，真实RAW含E盘FRED已跑。以下旧阶段条目作为历史。

'''
for path in (ROOT/'README.md', WORK/'PROJECT_STATUS_20261022.md'):
    text = path.read_text(encoding='utf-8')
    if not text.startswith('## 146：'): path.write_text(intro+text, encoding='utf-8')
handoff_path = WORK/'output/checkpoints/latest_handoff.json'
handoff = json.loads(handoff_path.read_text(encoding='utf-8'))
handoff.update(last_completed_stage=146, algorithm_stage=144, geometry_execution_stage=138,
               latest_diagnostic_stage=146, latest_full_forecast_stage=146,
               latest_full_forecast_report=str(OUT/'阈值抑制与四三二桨预测实验.html'),
               recommended_forecast_research_stage=144, running_jobs=[],
               archive_scope='Existing verified archive covers through142 only; 143–146 sources/results are live workspace files.',
               next_action='Refine thresholded rotor activity cores back to original arcs; fix merged/split components and motor identities.146 remains optional, baseline144 unchanged. Read stage146 README and scores before proposing physics gains.')
handoff['latest_completed_stages'] = sorted(set(handoff.get('latest_completed_stages', [])+[143,144,145,146]))
handoff['latest_RAW_pilot_stage'] = 146
handoff['latest_RAW_geometry_stage'] = 146
handoff['optional_rotor_branch_stage'] = 146
handoff['latest_checks'] = str(OUT/'checks146.json')
handoff['source_summary_files'] = list(dict.fromkeys(handoff.get('source_summary_files', [])+
    [str(ROOT/'research/stage146/README.md'), str(OUT/'summary146.json'), str(OUT/'scores146.json')]))
handoff['latest_stage146'] = dict(report=str(OUT/'阈值抑制与四三二桨预测实验.html'),
    source=str(ROOT/'research/stage146/branches146.py'), experimental_switch='--rotor-branches',
    baseline='138 geometry +140 motion +144 damping', experimental_enabled_by_default=False,
    actual_RAW_final_frames=1325, this_turn_actual_RAW_main_frames=3355, summary=summary,
    scores_file=str(OUT/'scores146.json'), independent_RPM_truth=False,
    physical_force_identified=False, realtime=False)
handoff_path.write_text(json.dumps(handoff, ensure_ascii=False, indent=2), encoding='utf-8')
(WORK/'output/latest_optimized_forecast.html').write_text('''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=stage146_threshold_branches_20261024/阈值抑制与四三二桨预测实验.html"><a href="stage146_threshold_branches_20261024/阈值抑制与四三二桨预测实验.html">打开最新实验与默认基线对照</a>''', encoding='utf-8')
report_path = OUT/'阈值抑制与四三二桨预测实验.html'
page = report_path.read_text(encoding='utf-8')
if '同一起点的物理分支对照' not in page:
    paired = scores['side4']['baseline_same_physical_origins']['0.4']
    physical = scores['side4']['physical_supported_only']['0.4']
    insertion = (f'<p><strong>同一起点的物理分支对照：</strong>侧视窗口仅在物理分支有输出的{physical["n"]}个0.4s起点上，基线{paired["mean_px"]:.2f}px，纯条件推力模型{physical["mean_px"]:.2f}px。物理分支当前更差；融合总体持平包含门控回退的作用，不能当作推力估计正确的证据。</p>'
       '<p>实际运行视频：<a href="clear4_final/predictions.mp4">清晰窗口</a> · <a href="side4_final/predictions.mp4">侧视窗口</a> · <a href="fred0_4_final/predictions.mp4">FRED0窗口</a>。这些视频显示原生画面及既有基线预测；实验桨盘分支看上方图片，预测数值看scores146.json。</p>')
    page = page.replace('<section><h2>预测对照</h2>', '<section><h2>预测对照</h2>'+insertion)
    report_path.write_text(page, encoding='utf-8')
print(json.dumps(dict(worklog_updated=True, handoff_stage=146, readme=str(ROOT/'research/stage146/README.md')), ensure_ascii=False))
