"""Publish local pilot notes and checkpoint; no remote action or ZIP update."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast,re,html
ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'event_drone_forecast'
OUT=ROOT/'output/ddhf_review_20261027';VIS=ROOT/'output/visibility_period_20261027'


def main():
    raw=json.loads((OUT/'native_pilot/summary.json').read_text('utf-8'))
    null=json.loads((OUT/'null_calibration/null_control.json').read_text('utf-8'))
    calibrated=json.loads((OUT/'null_calibrated_summary.json').read_text('utf-8'))
    visibility=json.loads((VIS/'final_slots/summary.json').read_text('utf-8'))
    smoke=json.loads((VIS/'raw_slots/summary.json').read_text('utf-8'))
    panels=''.join('<section><h2>'+html.escape(c['name'])+' · '+str(c['t'])+'秒</h2><img src="native_pilot/'+c['name']+'/calibrated_comparison.jpg" alt="原生事件与计数/谐波候选对照"></section>' for c in raw['cases'])
    table=''.join('<tr><td>'+r['name']+'</td><td>'+str(r['paper_default_pixels'])+'</td><td>'+str(r['null_calibrated_pixels'])+'</td></tr>' for r in calibrated['records'])
    report='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DDHF论文：逐像素谐波与五圆模型</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;line-height:1.8;max-width:1100px;margin:36px auto;padding:0 20px;background:#f3f6f8;color:#203044}section{background:white;padding:22px 26px;border-radius:10px;margin:18px 0}h1{font-size:29px}h2{font-size:21px}img{max-width:100%;height:auto}td,th{text-align:left;padding:8px 16px;border-bottom:1px solid #ddd}table{border-collapse:collapse}a{color:#176696}.note{background:#fff3d8;padding:12px}code{background:#eef2f5}</style>
<h1>DDHF论文：有价值的检测前端，尚未证明本机收益</h1><p>2026-10-27 · 实际读取Zotero内9页原文 · 自采train6三个RAW窗口 · 未改Zotero、RAW或公开ZIP。</p>
<section><h2>适合借鉴的部分</h2><p>逐像素原始时间/正负极性NDFT → 谱平坦度 → 多阶谐波梳状得分 → 桨叶活动候选。它补的是当前“强事件不等于桨叶”的前端；之后仍由五圆模型判断四/三/两点与侧影信息，不把每个活动块直接叫作一个电机。</p><p>同一像素可能有两个桨的周期。论文用窗外功率中位数抑制另一组谐波对背景估计的影响。这对粘连值得研究，但还没有恢复隐藏桨心或分离真实转速。</p><p>论文最终合并为整机框，本项目应保留局部候选与多峰，让共享几何组装。脚注2指出，低仰角的一部分漏检来自局部框太小、合并失败；并非每次都没有桨叶信号。</p></section>
<section><h2>本次小试与反例</h2><p>按公式6–11写CPU参考，采用论文表2的标量门槛0.89/1.5。K=128、候选模式4..31是本机明确选择的未公开细节，故不是完整作者复现。每窗33.333ms，约30Hz频率网格。</p><p class="note">原门槛下3帧的谐波掩码分别等于“像素事件数≥6”的全部候选；随机时刻/极性的256像素也全部通过。不能把这个掩码写成已证明的周期选择性，更不能据此否定作者完整实现。</p><p>只做一次独立随机反例的99%经验分位校准，得到8.403门槛；没有根据这些真实帧标签反复调阈值。复用已保存频谱后：</p><table><tr><th>窗口</th><th>原门槛像素</th><th>经验校准后</th></tr>'''+table+'''</table><p>清晰帧四个区域突出；侧视仍碎裂。没有逐桨分割/中心真值，不能把像素少了或落在旧框里当准确率。CPU每窗1.48–3.70秒，未做GPU优化，未接默认预测，不能声称30FPS。</p><p>图中第一行：原生ON/OFF20ms与原20ms密度阈值；第二行：按窗口时长归一化的33ms计数阈值与DDHF式原门槛；第三行：反例校准后掩码与候选谐波基频颜色。颜色可能跨半倍频/谐波，不代表真实电机RPM。</p></section>
'''+panels+'''
<section><h2>与当前优化的衔接</h2><p>另已实现模型槽归一化：恒速不等速4→3→2→4的最大假相对偏差24.6%→0，保留共同真实频率变化。仅已识别槽位使用新输入，未建模侧视保留旧控制。独立混合周期侧视改法均值52.718→53.910px变差，作为可选实验保留，没有混入槽位修复。</p><p>新开关<code>--visibility-period</code>已跑通1.2秒完整RAW：36时刻、32当前整机观测、28有效预测。当前本机短测计算能力16.26FPS，尚未达到30FPS；异步观测与运行负载不同，不用于和历史短片直接归因比较。</p></section>
<section><h2>下一步做什么</h2><ol><li>核对频谱参数和计分，用不同事件数量的随机时序/单次边缘扫过检验选择性。复用已保存频谱，避免重复长RAW。</li><li>谐波图作候选权重，与当前弧/外轮廓合用，不能把只剩内圈的活动区当物理桨盘边界。</li><li>保留多个周期峰，由五圆稳定槽位关联；侧视保持局部/混合证据与不确定性。</li><li>确认几何信息确有改善后，再做有限ROI的GPU批处理；二力反馈更新另行进行，不把普通位置反馈收益归给频率模块。</li></ol></section>
<section><h2>原文、开源调查和可检查记录</h2><p><a href="https://arxiv.org/abs/2603.08386">DDHF原文</a> · <a href="https://github.com/aitsam12/EBFM_Vibration">其参考工作EBFM作者库</a>（本次未复制代码；README与根许可标识不一致，复用需核对具体文件/提交）。未查到可核实的DDHF作者完整实现链接，不等于断言没有任何代码。</p><p><a href="../../event_drone_forecast/docs/DDHF_REVIEW.md">完整阅读与改进方案</a> · <a href="native_pilot/summary.json">三个RAW窗口记录</a> · <a href="null_calibration/null_control.json">固定随机反例</a> · <a href="null_calibrated_summary.json">单次经验校准</a> · <a href="../visibility_period_20261027/final_slots/summary.json">桨位代理同输入对照</a> · <a href="../../event_drone_forecast/docs/VISIBILITY_PERIOD.md">新开关说明</a></p><p>作者表1为整机检测F1@IoU0.5及L40-8Q帧处理时间；不等于逐桨中心/RPM/未来轨迹的指标，本项目未宣称复现其90.89%或2.39ms。</p></section></html>'''
    (OUT/'index.html').write_text(report,'utf-8')
    archive=ROOT/'output/github_release_final_20261027/event-drone-forecast-v0.4.0-alpha.1.zip'
    archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive_hash=='ff1172b3f2d2e1752d8ebc6d299c6d972adfaaa1776f8773240e44aa3eb58abb'
    source=ROOT/'output/github_release_final_20261027/event-drone-forecast/src/event_drone_forecast/position'
    unchanged=[p.name for p in source.glob('*.py') if p.read_bytes()==(CODE/'src/event_drone_forecast/position'/p.name).read_bytes()]
    assert len(unchanged)==7
    for name in ['visibility_period.py','harmonic_pixels.py','run_fast_structure.py','tools/replay_visibility_period.py','tools/probe_harmonic_pixels.py','tools/check_harmonic_null.py']:
        ast.parse((CODE/name).read_text('utf-8'),filename=name)
    targets=re.findall(r'(?:href|src)="([^"#]+)"',report)
    local=[p for p in targets if '://' not in p]
    assert all((OUT/p).is_file() for p in local)
    equivalence=json.loads((VIS/'raw_slots/latest_source_equivalence.json').read_text('utf-8'))
    assert equivalence['latest_relock_guard_saved_RAW_replay_max_prediction_difference_px']<1e-8
    checks=dict(local_links_checked=len(local),position_core_files_unchanged=unchanged,
        latest_source_RAW_replay_equivalence=equivalence,
        frozen_GitHub_ZIP_sha256=archive_hash,source_syntax_checked=True,
        all_RAW_unchanged=all(r['raw_unchanged'] for r in raw['cases']) and smoke['raw_unchanged'],
        author_DDHF_code_copied=False,default_period_selector='LegacyPeriodProxy',
        current_jobs=[],no_live_GitHub_upload=True,Zotero_library_or_PDF_modified=False)
    (OUT/'artifact_checks.json').write_text(json.dumps(checks,indent=2),'utf-8')
    stamp=datetime.now(timezone.utc).isoformat();handoff=ROOT/'output/checkpoints/latest_handoff.json'
    h=json.loads(handoff.read_text('utf-8'))
    h.update(updated_utc=stamp,current_work_state='Completed visibility-period controls/integration and user-requested DDHF paper review plus3RAW windows; no current jobs',current_jobs=[],current_running_jobs=[])
    h['visibility_period']=dict(report=str(CODE/'docs/VISIBILITY_PERIOD.md'),output=str(VIS),
        default_changed=False,new_opt_in='--visibility-period',legacy_identical_on_saved_inputs=True,
        synthetic_false_relative_bias_before_after=[.24617996604414272,0.],
        slot_only_side_forecasts_identical=True,pooled_side_mean04px_before_after=[52.71846778264579,53.91024177921393],
        RAW_ticks=36,valid_forecasts=28,modelled_clear_difference_not_accuracy_gain=.3079570194648227)
    h['DDHF_review']=dict(report=str(OUT/'index.html'),source='User Zotero9-page2603.08386v1',
        local_prototype='harmonic_pixels.py; self-written incomplete CPU reference',
        author_code_found=False,author_code_absence_not_proven=True,raw_windows=3,
        scalar_thresholds_not_discriminative_in_local_reference=True,random_null_accepted=256,
        one_empirical_null_gate=8.402957916259766,model_prediction_default_changed=False,
        next='Audit unspecified spectral settings against EBFM and fixed count-conditioned null/edge controls; then use harmonic evidence as soft support for current arcs, preserve multiple fundamentals and stable5circle slots. GPU only after useful evidence is established.')
    h['latest_research_report']=str(OUT/'index.html');handoff.write_text(json.dumps(h,ensure_ascii=False,indent=2),'utf-8')
    with (ROOT/'WORKLOG.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 2026-10-27 可见性周期归一化与DDHF阅读小试\n\n'
            '- 用户要求继续优化，随后指定Zotero论文Real-Time Drone Detection in Event Cameras via Per-Pixel Frequency Analysis。只读原库选定PDF，临时库查找副本已清理；实际读9页、看方法/图表，未改Zotero。\n'
            '- visibility_period.py新增稳定模型槽f²基准，避免恒定不等速桨叶4/3/2可见集合改变造成假变化；受控最大相对偏差24.618%→0，隐藏转速不伪造。旧控制同输入预测差0；槽位专用侧视差0，混合周期侧视独立试验均值52.718→53.910px，保留实验。\n'
            '- run_fast_structure.py增加互斥--visibility-period/--pooled-axis-period，默认仍旧。新开关RAW1.2s完成36时刻/32当前整机观测/28有效预测；本机16.258FPS，不冒充30或与历史异步输入严格速度对照。\n'
            '- DDHF原文https://arxiv.org/abs/2603.08386：每像素NDFT+谱平坦度+窗外中位噪声谐波梳，候选活动区不等于物理桨盘外缘；零仰角部分失败与框合并有关。作者完整代码链接本次未找到，不断言不存在。EBFM参考作者库已查，未复制代码，README/根许可标识不一致需核对。\n'
            '- harmonic_pixels.py自写CPU不完整参考：K128/基频模式4..31为本机补充选择，论文表2标量参数原样。3真实33.333ms窗保留10261/4332/8506，全部等于事件数≥6候选；固定随机时序256/256也通过。明确不能叫复现成功或用来否定作者。\n'
            '- 单次独立随机反例99%经验分位门槛8.403，复用频谱后7182/2837/2486像素，图片已看；不称真实误检率已校准，不当逐桨精度。CPU1.48–3.70s/窗，未入默认或推力链路。\n'
            '- 报告output/ddhf_review_20261027/index.html、docs/DDHF_REVIEW.md、docs/VISIBILITY_PERIOD.md；7位置核心与已发布预览ZIP字节保持，保护yolov10目录未动。无后台作业。下一步先频谱选择性/软弧支持，未知力反馈优化仍待续。\n')
    print(json.dumps(dict(report=str(OUT/'index.html'),checks=checks),ensure_ascii=False))


if __name__=='__main__':main()
