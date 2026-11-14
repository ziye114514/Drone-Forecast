"""Record the ownership audit and the thrust-channel ablation in WORKLOG/handoff.

Only writes documents and the machine handoff. No prediction, no default change.
"""
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import unquote
import json,re
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
OUT=WORK/'output/peak_ownership_20261029'
MARK='## 2026-10-29 空间方差主峰归属审计'
TEXT=f'''{MARK}（NEXT_ROTOR_INPUT 第1节）

- 用户授权按交接下一步继续。本轮只做归属，不进合力/推力变化率；默认行为、原预测、已发布ZIP均未改。
- 新增`spatial_peak_ownership.py`：固定网格多峰谱＋复用`VisibilityPeriodProxy._slots`的当前槽位匹配。容差3%，谐波2×/3×与半频不冒充身份，两个槽位同时进容差记歧义不挑选。状态机同时比较上一帧主峰与上一帧槽位集合：集合稳定而主峰跳走=`dominance_switch`（换人），同槽位频率变化=`rotation_change_of_same_owner`，主峰动而所归属槽位未动=`peak_moved_without_slot_change`标可疑，上一帧归属超过0.35s则`owner_reinitialized`不硬连。
- 合成机制检查13项全通过：单分量归属、四分量分辨、弱分量被泄漏掩掉（如实保留）、身份无关平方和、单分量不能证实四槽位、支配切换≠转速变化、可比振幅时支配转移会掩盖转速变化（不伪称）、转速变化可识别、谐波/半频/双槽歧义、直接命中优先于谐波、共同计数倍率不造峰。
- 真实RAW三段只重解码同一截止窗（公开rolling reader），不重跑、不改预测器、不用标注：清晰26–28s共60帧、侧视110–118s共240帧、另一段40–46s共180帧。
- 清晰段：24个有槽位声明、21个已归属；主峰与所归属槽位频率中位偏差1.00%、最大2.16%；最长连续归属为槽位1共13帧0.43s。26.100–26.600s 有14个时刻四个槽位同时被独立证实，逐峰偏差中位1.09%/P95 2.08%/最大2.15%；其余10个时刻不完整（3/4、2/4、1/4）。
- 修正上一批表述：“单主峰只对应一个分量”是单峰取最大值的说法；多峰分解在同一截止时刻能分出四个（231.88/305.77/279.07/251.97Hz 对 230.21/303.25/284.43/256.48Hz），属修正不是推翻。单峰读数231.875Hz用滚动读取器重解码逐位相同。
- 侧视240帧、另一段180帧已识别槽位均为0；两段合计129个力输入时刻全部落在“无已识别槽位”，仍是pooled/legacy汇总口径。该观测在这两段不可能成为逐桨输入。
- 未加窗Lomb–Scargle有泄漏：合成振幅比3:1时四个桨只能分辨出三个（`weak_component_masked_by_leakage`）。限制保留，不加窗修饰结果。
- 本轮无精度结论：清晰段只有1.5Hz稀疏框、无法给0.4秒评分；另两段无可归属槽位。因此不进入“合力与推力变化率”，也不设为默认输入。
- 独立RPM数据重试仍为自签名证书失败，未绕过、未下大文件，记录`output/peak_ownership_20261029/independent_RPM_dataset_access_retry.json`；effqpdet README未说明1100–8200RPM是实测还是控制设定，拿到也不能直接当真值。
- 报告`output/peak_ownership_20261029/index.html`；`docs/PEAK_OWNERSHIP.md`、`FRED_TRANSFER.md`、`NEXT_ROTOR_INPUT.md`、README、结果入口与latest_handoff已同步。无后台作业。
'''
MARK2='## 2026-10-29 推力通道消融、标注使用与本批收口'
TEXT2=f'''{MARK2}

- 用户要求继续挖那4个无输入起点。先逐帧查原因，不先改阈值。
- 侧视240帧：122帧有力输入。其余118帧分类为——无平面/无整机72（合法缺席，其中65帧无当前整机）、只有未达门槛的弱周期估计36、完全没有频率5、通过门槛后被连贯或幅值拒绝5。另一段180帧对应91/50/31/1。
- 被打分的4个无输入起点全部在117.03–117.63的0.6秒内：无人机以约680px/s向左出画，池化估计仍给263–290Hz，但频率像素只有2–23（合格帧中位98.5）。属观测不足，不是门槛写错。
- 新增tools/probe_force_coverage.py：把弱估计放在最低一档（合格周期→0.12s持稳→弱估计），要求年龄≤0.3s、与上次已接受频率差≤8%（复用既有连贯常数）、≥2频率像素、不计入增益学习；机制检查10项全通过（不覆盖持稳、不学习、不一致/过期/无历史都拒绝）。
- 侧视26起点：默认34.6049；弱先验兜底34.0265（-0.58）；持稳0.3s 34.0442（-0.56）；两者同时34.0442（-0.56）。5个起点变好、1个变差（113.63s +1.75px）。另一段59.7896→59.7821。两种不同机制收益几乎相同，说明起作用的是“缺口里有点输入”，不是弱估计更准。不推广：单片段、26起点、0.58px。
- 修正上一轮：有输入31.06 vs 无输入54.08的“23px差”是选择偏差，因果上补输入只值0.4–0.6px。
- 误差来源分解（同26起点）：源帧当前中心误差均值6.92/中位6.19/P95 18.13/最大24.18px，无一超30；0.4秒未来误差34.60/30.02/72.54，13/26超30。差的约28px长在0.4秒外推里，不是定位、不是推力精度、不是覆盖。
- 报告新增“侧视线索为什么丢、丢了到底值多少”与coverage_diagnosis.png；docs/PEAK_OWNERSHIP.md、NEXT_ROTOR_INPUT.md、README、结果入口、latest_handoff同步。默认行为/发布ZIP/原预测未改，无后台作业。
- 下一步按证据：回到运动外推（速度、加速度、转向与长尾）；不要再在推力/周期/覆盖上找精度，也不要推广弱先验或长持稳。
'''
TEXT2=f'''{MARK2}

- 用户指示把交接剩余项做完，并指出output里有人工标注。先盘点标注再决定怎么测，没有先看误差再挑样本。
- 标注盘点：`dense_labels_side_110_1195`（75框/20Hz）、`dense_labels_train2_40_46`（36框/20Hz）、`dense_labels_train6_20_35`（23框/20Hz，20.05–21.20s）。清晰26–28s只有历史归档稀疏框`paper_reproduction_v030/evaluation_data/train6.json`（1.5Hz，该段3个present）：检出3/3、当前定位1.65/3.36/6.10px（均值3.71px）；框间隔超过150ms插值门限，不能给0.4秒评分。
- 按同一候选入口新跑有人工框的20–21.2s：42帧，insufficient27/axis_only15/four0，36.98处理FPS；当前定位中位18.86px、0.4秒91.50px（n=3）。该段无人机贴右缘、桨盘被裁，没有槽位，不能用于通道替换。首次运行因缺numba失败（EDF_NUMBA_PATH应指向`D:\drone forecast\.cache\numba_cpu_064`），失败目录已删除重跑，未把失败当结果。
- 新增`tools/probe_force_channel_ablation.py`：只改`force.input_u`，位置/曲线/光流分支不变；三片段回放先逐项复原冻结预测（最大差0.0），差别只来自这一项。
- 侧视26起点：默认34.6049 → 通道关41.3702（+6.7652），P95 72.5433→97.2166。按起点拆：有力输入22个31.0633px，无输入4个54.0838px。
- 另一段15起点：默认59.7896 → 59.9792（+0.1897）；只有1个起点有力输入（10.5137px）。
- 清晰替换实验（无连续标注，只比预测差）：多峰相对f²替换SDK相对f²（只在4/4独立证实时替换，其余保持原值），输入差中位2.78%、最大6.15%；预测位置最大移动0.3778px、P95 0.2627、中位0.0027。
- 结论：通道存在值约6.8px，但输入精度3%只值0.4px以内；34.60→30px还差4.6px，推力变化率Ṫn+Tṅ按证据关闭。真正缺口是覆盖：侧视4个起点没有力输入，54.08 vs 31.06。
- 独立RPM仍被自签名证书挡住；本机有界名字扫描（4根目录、深度3、`propeller|qpdet|rpm|blade|rotor`、只读名字不打开文件）命中0，记录`output/peak_ownership_20261029/local_rpm_dataset_scan.json`。名字扫描不证明整机都没有。
- 报告新增“推力通道到底值多少像素”“人工标注用在了哪里”“独立RPM”三节与`thrust_channel_leverage.png`；docs、README、结果入口、latest_handoff同步。默认行为/发布ZIP/原预测未改，无后台作业。
- 下一步按证据排序：侧视那4个无输入起点的资格/连续性/坐标轴 → 整机定位与运动入口 → 独立RPM换算；不要继续调净力系数或推力变化率。
'''
MARK3='## 2026-10-29 侧视力输入覆盖诊断与误差来源'
TEXT3=f'''{MARK3}

- 用户要求继续挖那4个无输入起点。先逐帧查原因，不先改阈值。
- 侧视240帧：122帧有力输入。其余118帧分类为——无平面/无整机72（合法缺席，其中65帧无当前整机）、只有未达门槛的弱周期估计36、完全没有频率5、通过门槛后被连贯或幅值拒绝5。另一段180帧对应91/50/31/1。
- 被打分的4个无输入起点全部在117.03–117.63的0.6秒内：无人机以约680px/s向左出画，池化估计仍给263–290Hz，但频率像素只有2–23（合格帧中位98.5）。属观测不足，不是门槛写错。
- 新增tools/probe_force_coverage.py：把弱估计放在最低一档（合格周期→0.12s持稳→弱估计），要求年龄≤0.3s、与上次已接受频率差≤8%（复用既有连贯常数）、≥2频率像素、不计入增益学习；机制检查10项全通过（不覆盖持稳、不学习、不一致/过期/无历史都拒绝）。
- 侧视26起点：默认34.6049；弱先验兜底34.0265（-0.58）；持稳0.3s 34.0442（-0.56）；两者同时34.0442（-0.56）。5个起点变好、1个变差（113.63s +1.75px）。另一段59.7896→59.7821。两种不同机制收益几乎相同，说明起作用的是“缺口里有点输入”，不是弱估计更准。不推广：单片段、26起点、0.58px。
- 修正上一轮：有输入31.06 vs 无输入54.08的“23px差”是选择偏差，因果上补输入只值0.4–0.6px。
- 误差来源分解（同26起点）：源帧当前中心误差均值6.92/中位6.19/P95 18.13/最大24.18px，无一超30；0.4秒未来误差34.60/30.02/72.54，13/26超30。差的约28px长在0.4秒外推里，不是定位、不是推力精度、不是覆盖。
- 报告新增“侧视线索为什么丢、丢了到底值多少”与coverage_diagnosis.png；docs/PEAK_OWNERSHIP.md、NEXT_ROTOR_INPUT.md、README、结果入口、latest_handoff同步。默认行为/发布ZIP/原预测未改，无后台作业。
- 下一步按证据：回到运动外推（速度、加速度、转向与长尾）；不要再在推力/周期/覆盖上找精度，也不要推广弱先验或长持稳。
'''


def links(path):
    text=path.read_text('utf-8')
    broken=[]
    for href in re.findall(r'href="([^"]+)"',text):
        if href.startswith(('http','#')):
            continue
        target=(path.parent/unquote(href)).resolve()
        if not target.exists():
            broken.append(href)
    return broken


def main():
    report=OUT/'index.html'
    broken=links(report)
    if broken:
        raise ValueError(f'broken local links: {broken}')
    stats=json.loads((OUT/'attribution.json').read_text('utf-8'))['summary']
    ablation=json.loads((OUT/'force_channel_ablation.json').read_text('utf-8'))
    coverage=json.loads((OUT/'force_coverage.json').read_text('utf-8'))
    log=WORK/'WORKLOG.md'
    text=log.read_text('utf-8')
    for mark,body in ((MARK,TEXT),(MARK2,TEXT2),(MARK3,TEXT3)):
        if mark not in text:
            with log.open('a',encoding='utf-8') as handle:
                handle.write('\n\n'+body)
            text=log.read_text('utf-8')
    path=WORK/'output/checkpoints/latest_handoff.json'
    handoff=json.loads(path.read_text('utf-8'))
    stamp=datetime.now(timezone.utc).isoformat()
    side=ablation['clips']['side'];other=ablation['clips']['other'];clear=ablation['clips']['clear']
    handoff.update(utc=stamp,updated_utc=stamp,updated_local_date='2026-10-29',
        active_batch='peak_ownership_20261029',
        last_completed_named_batch='peak_ownership_20261029',
        execution_status='Ownership audit and thrust-channel ablation completed; default inputs, forecaster and '
            'published ZIP unchanged; no background jobs; no accuracy gain claimed',
        in_progress=False,experiments_running=False,
        previous_report=handoff.get('latest_report'),
        latest_report=str(report),
        peak_ownership=dict(report=str(report),document=str(ROOT/'docs/PEAK_OWNERSHIP.md'),
            next_step_source=str(ROOT/'docs/NEXT_ROTOR_INPUT.md'),
            summary_file=str(OUT/'attribution.json'),checks=str(OUT/'artifact_checks.json'),
            source_snapshot=str(OUT/'source_snapshot/manifest.json'),
            ablation_snapshot=str(OUT/'source_snapshot_ablation/manifest.json'),
            default_enabled=False,forecaster_modified=False,annotations_used_for_decisions=False,
            dominant_peak_is_one_rotor_not_whole_body=True,
            four_rotor_simultaneous_observation_ticks=stats['clear']['fully_resolved_ticks'],
            four_rotor_peak_error_median=stats['clear']['fully_resolved_peak_error']['median'],
            four_rotor_peak_error_p95=stats['clear']['fully_resolved_peak_error']['p95'],
            dominant_to_slot_error_median=stats['clear']['median_peak_to_slot_relative_error'],
            dominant_to_slot_error_max=stats['clear']['max_peak_to_slot_relative_error'],
            side_ticks_with_slot_claims=stats['side']['claimed_ticks'],
            other_ticks_with_slot_claims=stats['other']['claimed_ticks'],
            unattributable_force_inputs=stats['side']['force_active_ticks']+stats['other']['force_active_ticks'],
            mechanical_RPM_measured=False,thrust_measured=False),
        thrust_channel_leverage=dict(
            method='same frozen rows, only force.input_u changed; as-is replay reproduces frozen predictions exactly',
            as_is_reproduces_frozen=side['as_is_reproduces_frozen_predictions'],
            side_mean_error_as_is=side['modes']['as_is']['error_0.4']['mean'],
            side_mean_error_without_channel=side['modes']['no_thrust_input']['error_0.4']['mean'],
            side_p95_as_is=side['modes']['as_is']['error_0.4']['p95'],
            side_p95_without_channel=side['modes']['no_thrust_input']['error_0.4']['p95'],
            side_with_input=side['modes']['as_is']['error_by_input_presence']['with_input'],
            side_without_input=side['modes']['as_is']['error_by_input_presence']['without_input'],
            other_mean_error_as_is=other['modes']['as_is']['error_0.4']['mean'],
            other_mean_error_without_channel=other['modes']['no_thrust_input']['error_0.4']['mean'],
            clear_substitution_max_prediction_shift_px=clear['difference_variance_hybrid']['max'],
            clear_substitution_p95_prediction_shift_px=clear['difference_variance_hybrid']['p95'],
            clear_input_relative_difference_median=clear['input_pairs']['median_relative_difference'],
            clear_input_relative_difference_max=clear['input_pairs']['max_relative_difference'],
            thrust_rate_step_entered=False,
            reason='3percent input difference moves predictions below0.4px while the channel presence is worth6.8px; '
                   'the remaining4.6px gap is not a thrust-precision problem'),
        labeled_near_segment=ablation['labeled_near_segment'],
        sparse_clear_localization=ablation['sparse_clear_localization'],
        force_coverage=dict(
            method='same frozen rows; weak unqualified pooled estimate added only below the existing hold',
            controlled_checks=coverage['controlled'],
            side_census=coverage['clips']['side']['census']['base'],
            other_census=coverage['clips']['other']['census']['base'],
            variants={k:dict(input_ticks=v['input_ticks'],mean_error=v['error_0.4']['mean'])
                for k,v in coverage['clips']['side']['modes'].items()},
            no_input_origins_share='117.03-117.63s exit interval; pooled still reports263-290Hz with2to23 frequency '
                'pixels against a98.5 median on qualified ticks',
            selection_effect_correction='the23px with/without-input split is a selection effect; causally recovering '
                'the input is worth0.4to0.6px',
            source_localization_px=dict(mean=6.92,median=6.19,p95=18.13,maximum=24.18),
            future_0p4_px=dict(mean=34.60,median=30.02,p95=72.54,origins_above30=13),
            promoted=False),
        local_RPM_dataset_scan=str(OUT/'local_rpm_dataset_scan.json'),
        next_action='Leave the thrust, period and coverage branches alone: precision is worth below0.4px, coverage '
            'below0.6px, and source localization is already6.9px. Work on the0.4s motion extrapolation (velocity, '
            'acceleration, turn and long tail) where the remaining28px is created.',
        critical_next_steps=['Do not promote the weak pooled prior or the extended hold: one clip,26 origins,0.58px, '
            'and one origin gets worse',
            'Target the0.4s extrapolation: source localization is6.9px while the future error is34.6px, so the gap '
            'is created after the source frame',
            'Independent RPM scale remains blocked by the self-signed certificate on ptak.felk.cvut.cz; a bounded '
            'local name scan found no substitute and verification was not bypassed',
            'Thrust-rate Tdotn+Tn_dot stays closed: a3percent thrust-input change only moves predictions below0.4px'],
        current_running_jobs=[],running_jobs=[],current_jobs=[],pending_runs=[],
        report_local_links_checked=True,latest_named_report_is_authoritative=True)
    path.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),'utf-8')
    checks=dict(report=str(report),local_links_checked=True,broken_links=[],
        worklog_sections=[MARK,MARK2,MARK3],
        worklog_recorded=all(mark in log.read_text('utf-8') for mark in (MARK,MARK2,MARK3)),
        handoff_keys=['peak_ownership','thrust_channel_leverage','force_coverage','labeled_near_segment',
            'sparse_clear_localization','next_action'],
        figures=['ownership.png','thrust_channel_leverage.png','coverage_diagnosis.png'],
        json_files=['attribution.json','mechanism_checks.json','summary.json','force_channel_ablation.json',
            'force_coverage.json','local_rpm_dataset_scan.json','independent_RPM_dataset_access_retry.json'],
        running_jobs=0,default_enabled=False,forecaster_modified=False,
        annotations_used_for_decisions=False,annotations_used_for_scoring=True)
    (OUT/'artifact_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),'utf-8')
    print(json.dumps(checks,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
