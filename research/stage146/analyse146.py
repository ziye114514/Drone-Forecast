"""Score emitted experimental branches AFTER inference, then make a local report."""
from pathlib import Path
import argparse
from collections import Counter
from types import SimpleNamespace
import html
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
OUT = WORK/'output/stage146_threshold_branches_20261024'
sys.path[:0] = [str(ROOT/'core'),str(ROOT/'src'), str(ROOT/'research/stage146'), str(ROOT/'research/stage144')]
from branches146 import ThresholdFrontend, render_panel
from polarity import counts
from score_dense import nearest, load_labels
from event_drone_forecast.raw_io import open_raw, raw_chunks
from metavision_sdk_core import BaseFrameGenerationAlgorithm

CASES = dict(
    clear4=dict(raw='E:/dataset/train/6/media/event.raw', labels=None,
                reference='future detector box centre; consistency only'),
    side4=dict(raw='E:/dataset/train/6/media/event.raw',
               labels=WORK/'output/dense_labels_side_110_1195/labels.json',
               reference='human boxes; exposed development window',
               previous=WORK/'output/stage144_e2e_default_damped/predictions.jsonl'),
    train2_4=dict(raw='E:/dataset/train/2/aligned/event.raw',
                  labels=WORK/'output/dense_labels_train2_40_46/labels.json',
                  reference='human boxes; exposed second recording',
                  previous=WORK/'output/stage144_e2e_train2_default/predictions.jsonl'),
    fred0_4=dict(raw=str(OUT/'input/fred0/events.raw'), labels='fred0',
                 reference='FRED supplied coordinates; one selected large-target window'))


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def references(case, rows):
    if case['labels'] is None:
        return [dict(t=r['t'], centre=r['body']['center']) for r in rows if r['body'].get('detected')]
    if case['labels'] != 'fred0': return load_labels(case['labels'])[0]
    result = []
    for line in (OUT/'input/fred0/coordinates.txt').read_text().splitlines():
        stamp, coords = line.split(':', 1); stamp = float(stamp)
        if not rows[0]['t'] <= stamp <= rows[-1]['t']: continue
        box = np.array([float(v) for v in coords.split(',')])
        if len(box) != 4: continue
        result.append(dict(t=stamp, centre=((box[:2]+box[2:])/2).tolist()))
    return result


def score(refs, rows, name, require=None):
    indexed = {round(r['t'], 6): r for r in rows}; result = {}
    for horizon in (.1, .2, .4):
        errors = []; pairs = []; missing = 0
        for reference in refs:
            row = nearest(indexed, reference['t']-horizon)
            point = None if row is None else row.get('predictions', {}).get(name, {}).get(str(horizon))
            if require is not None and row is not None:
                if str(horizon) not in row.get('predictions', {}).get(require, {}): point = None
            if point is None:
                missing += 1; continue
            error = float(np.linalg.norm(np.array(point)-reference['centre']))
            errors.append(error); pairs.append(dict(t=reference['t'], origin_t=row['t'],
                          error_px=error, predicted=point, truth=reference['centre']))
        result[str(horizon)] = dict(n=len(errors), missing=missing,
            mean_px=float(np.mean(errors)) if errors else None,
            median_px=float(np.median(errors)) if errors else None,
            p90_px=float(np.quantile(errors, .9)) if errors else None, pairs=pairs)
    return result


def rebuild_panel(case, row, destination):
    reader = open_raw(case['raw']); h, w = reader.get_size()
    _, events = next(iter(raw_chunks(reader, round(row['t']-.02, 6), .02, 20000)))
    prepared = counts(events, w, h)
    front = ThresholdFrontend(w, h)
    feature = front.feature(prepared['on'], prepared['off'])
    feature[front.count_gate(prepared['on'], prepared['off'], front.cfg.min_pixel_count)] = 0
    density = cv2.GaussianBlur(feature, (3, 3), .8)
    measurement = row['rotor_branch']
    mask = np.uint8(density >= measurement['threshold'])*255
    exact = int(np.count_nonzero(mask)) == measurement['retained_pixels']
    image = np.zeros((h, w, 3), np.uint8)
    BaseFrameGenerationAlgorithm.generate_frame(events, image)
    panel = render_panel(image, SimpleNamespace(mask=mask), measurement, row['t'])
    cv2.imwrite(str(destination), panel)
    return dict(t=row['t'], case=case['raw'], exact_spatial_mask_reconstruction=exact,
                new_inference=False, image_source='same current20ms RAW with recorded measurements overlaid')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--suffix', default='')
    suffix = parser.parse_args().suffix
    all_rows = {}; summary = {}; scores = {}; panels = {}
    for name, case in CASES.items():
        run_directory = OUT/(name+suffix)
        rows = load(run_directory/'predictions.jsonl'); all_rows[name] = rows
        completion = json.loads((run_directory/'completion.json').read_text())
        states = Counter(r['rotor_branch']['geometry']['state'] for r in rows)
        proxy = [r for r in rows if r['rotor_branch']['thrust_proxy']['valid']]
        weights = [r['rotor_branch']['expert']['weight'] for r in rows]
        summary[name] = dict(frames=len(rows), states=dict(states),
            observed_count_hypotheses=dict(Counter(str(r['rotor_branch']['geometry']['observed_count']) for r in rows)),
            body_detection_ticks=sum(r['body']['detected'] for r in rows),
            conditional_thrust_ticks=len(proxy), fused_active_ticks=sum(w>0 for w in weights),
            maximum_fusion_weight=max(weights, default=0),
            main_threshold=4., thin_threshold=2.,
            thin_fallback_ticks=sum(r['rotor_branch']['selected_mask']=='thin_fallback' for r in rows),
            split_recovery_ticks=sum(r['rotor_branch']['selected_mask']=='split_recovery' for r in rows),
            actual_run_directory=str(run_directory),
            retained_display_event_fraction=sum(r['rotor_branch']['retained_display_events'] for r in rows)/max(1,sum(r['event_count'] for r in rows)),
            processing_median_ms=completion['processing_median_ms'],
            branch_median_ms=completion['rotor_branch_median_ms'], sources_unchanged=completion['sources_unchanged'],
            reference=case['reference'])
        refs = references(case, rows)
        scores[name] = dict(reference=case['reference'], reference_count=len(refs),
            baseline=score(refs, rows, 'optimized_geometry'),
            fused=score(refs, rows, 'rotor_fused'),
            physical_supported_only=score(refs, rows, 'rotor_physics'),
            baseline_same_physical_origins=score(refs, rows, 'optimized_geometry', require='rotor_physics'))
        if case.get('previous'):
            old = {round(r['t'], 6): r for r in load(case['previous'])}
            differences = []
            for r in rows:
                before = old[round(r['t'], 6)]['predictions'].get('optimized_geometry', {})
                after = r['predictions'].get('optimized_geometry', {})
                assert before.keys() == after.keys()
                for h in before: differences.append(float(np.max(np.abs(np.array(before[h])-after[h]))))
            summary[name]['baseline_max_coordinate_change_vs_144_px'] = max(differences, default=0.)
    choices = {}
    choices['clear_four'] = ('clear4', next(r for r in all_rows['clear4'] if r['rotor_branch']['geometry']['state']=='four' and r['t']>=23.))
    choices['review_C_four'] = ('side4', min(all_rows['side4'], key=lambda r: abs(r['t']-112.02)))
    choices['merged_recovered'] = ('clear4', min(all_rows['clear4'], key=lambda r: abs(r['t']-26.66)))
    for state, label in [('three_completed', 'three_completion'), ('two_side', 'two_side')]:
        found = [(name, r) for name, rows in all_rows.items() for r in rows
                 if r['rotor_branch']['geometry']['state'] == state]
        if found:
            # Display selection does not change any forecast or score.
            name, row = max(found, key=lambda q: min([r['major'] for r in q[1]['rotor_branch']['geometry']['observed_rotors']]))
            choices[label] = name, row
    choices['fred_limitation'] = ('fred0_4', min(all_rows['fred0_4'], key=lambda r: abs(r['t']-109.62)))
    for label, (name, row) in choices.items():
        panels[label] = rebuild_panel(CASES[name], row, OUT/f'{label}.png')
    (OUT/'summary146.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'scores146.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'panels146.json').write_text(json.dumps(panels, ensure_ascii=False, indent=2), encoding='utf-8')
    # Scientific plot: manual box centres and all 0.4s emitted forecasts on the
    # exact same origins. No detector-output reference is mixed into this plot.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, name in zip(axes, ('side4', 'train2_4')):
        for key, color in [('baseline', '#777777'), ('fused', '#d77719')]:
            pairs = scores[name][key]['0.4']['pairs']
            points = np.array([p['predicted'] for p in pairs])
            axis.plot(points[:, 0], points[:, 1], '.', color=color, label=key, markersize=4)
        truth = np.array([p['truth'] for p in scores[name]['baseline']['0.4']['pairs']])
        axis.plot(truth[:, 0], truth[:, 1], 'o', color='#238051', markerfacecolor='none', markersize=4, label='manual reference')
        axis.set_title(name+' | horizon 0.4 s'); axis.set_xlabel('x (native pixels)'); axis.set_ylabel('y (native pixels)')
        axis.invert_yaxis(); axis.legend(fontsize=8); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'forecast_comparison.png', dpi=150); plt.close(fig)
    rows_html = ''
    for name, values in summary.items():
        a, b = scores[name]['baseline']['0.4'], scores[name]['fused']['0.4']
        fmt = lambda v: '—' if v is None else f'{v:.2f}'
        rows_html += ('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in
            [name, values['frames'], values['states'].get('four', 0), values['states'].get('three_completed', 0),
             values['states'].get('two_side', 0), values['conditional_thrust_ticks'],
             f'{fmt(a["mean_px"])} → {fmt(b["mean_px"])} (n={a["n"]}/{b["n"]})',
             f'{values["processing_median_ms"]:.1f} ms'])+'</tr>')
    image_html = ''.join(f'<figure><img src="{label}.png"><figcaption>{html.escape(label)}，t={info["t"]:.2f}s。同期RAW与已发布测量的显示重建；不是新推理。</figcaption></figure>'
                         for label, info in panels.items())
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>阈值抑制与四三二桨预测实验</title>
<style>body{font:17px/1.8 "Microsoft YaHei",sans-serif;max-width:1320px;margin:30px auto;padding:0 20px;color:#172b3d;background:#f5f7fa}section{background:white;padding:22px;margin:18px 0;border-radius:12px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:9px;border:1px solid #cad3df;text-align:left}img{max-width:100%;height:auto}code{background:#edf1f6;padding:3px 6px}a{color:#215b9d}</style>
<h1>阈值抑制与四／三／二桨预测实验</h1><p>2026-10-24 · 实验模块146 · 原生RAW输入 · 不使用G重建 · 物理分支尚未升为默认</p>
<section><h2>这次实际改了什么</h2><p>对20ms内的ON/OFF计数取重复双极性活动，经原有邻域平滑后阈值抑制。主桨盘通道阈值4；三轮廓时尝试阈值8，仅在四个独立高阈值轮廓能对应原三块的1/1/2局部分裂时采用。薄线回退阈值2，且只有两个细长、方向一致的分量才进入侧视分支。计数阈值不是0—255灰度阈值；保留掩模用于显示和轮廓，周期估计继续读取原始事件时间戳。高阈值获得的活动核心不等于完整物理桨盘边界。</p>
<p>四桨：四个实测候选建立共面投影模型，透视下用对角线交点表示投影中心，不强行要求图像是平行四边形。三桨：优先从近期四点模型做局部仿射传递并检查椭圆尺寸一致性；无历史模型时，仅在三个形状相容的椭圆提供的弱透视度量能消除角点歧义时补第四点，形状不相容的粘连块拒绝补点。二桨：检查扁率、轴线与中心连线一致性后给出无向法线。0/1个分量或形状不可信时回退运动学基线。</p>
<p>补点不等于补转速。缺失电机只允许短时保留以前的频率，或明确标注“隐藏桨与可见桨等速”的条件假设。输出Σf²相对推力代理与两种法线符号，不输出牛顿单位的虚构推力。共轴平行桨的转速差会产生力矩，不直接把合推力转向桨心连线。</p></section>
<section><h2>真实RAW运行结果</h2><p>下表的四／三／二是算法合格假设次数，不是人工核验的可见桨盘数，也不是转速或姿态正确率。</p>
<table><tr><th>窗口</th><th>帧数</th><th>四桨</th><th>三桨补点</th><th>二桨侧视</th><th>条件推力代理</th><th>0.4s 基线→融合 px</th><th>处理中位</th></tr>'''+rows_html+'''</table>
<p>side4/train2_4用既有人工框中心作参考；clear4用未来检测器框中心，仅衡量自洽；fred0_4用FRED提供的框，窗口按较大目标选取，不能视为全FRED基准或盲测。缺失预测、逐端点误差和物理分支同起点基线另列在评分JSON中。各行不能混为同一精度指标。</p>
<p>物理专家用已到期的0.2s观测残差更新有效像素增益与综合扰动，0.1/0.2/0.4s各自按本时域已经到期的误差决定小权重融合；最大权重25%再乘几何置信度，隐藏转速假设另减半。质量、推力系数、真实风速并未被独立辨识。当前默认阻尼基线保持不变，新增分支通过开关启用。</p></section>
<section><h2>抑制与几何结果</h2><p>绿色为拟合到实测支持的候选轮廓；紫色空圈是推算位置；双箭头是尚未确定正负的法线。显示裁剪来自当前机体检测框。</p>'''+image_html+'''</section>
<section><h2>预测对照</h2><img src="forecast_comparison.png"><p>橙色为融合预测，灰色为原基线，空心绿色为人工框中心；重合意味着新物理信息没有改变该起点的预测，不能宣称已用推力提升精度。</p></section>
<section><h2>当前卡点</h2><p>单一阈值不能保证只剩桨盘：相邻桨盘可能粘连，高阈值也会将同一桨盘切成多块，机体的重复边缘仍可能保留。FRED即使挑较大目标，当前窗口也未稳定获得可用于建模的独立桨盘。三点无模型的补全仍依赖等半径、矩形机架与弱透视假设；二桨活动组可能混有四个电机的周期，不能唯一得到四电机推力分布。应先解决轮廓分裂／粘连与电机身份，随后评估力矩和姿态变化；不把补出的结构当成新测量。</p></section>
<section><h2>复用来源与复跑</h2><p>实际复用本项目110/124的合并保护与61的先判线几何、79的现有SDK周期引擎。论文参考：<a href="https://arxiv.org/html/2603.01997v1">RPM-KF 的高活动阈值与周期统计</a>、<a href="https://arxiv.org/abs/2604.18289">椭圆／频率驱动相对状态估计</a>、<a href="https://arxiv.org/abs/2610.31384">CoHelix 的四桨共享投影</a>。没有将论文作者的完整系统当作已复现。</p>
<pre>python event_drone_forecast/run_optimized_forecast.py --raw 你的RAW --output 一个新目录 --start 110 --duration 9.5 --rotor-branches --rotor-threshold 4 --preview</pre>
<p><a href="summary146.json">运行汇总</a> · <a href="scores146.json">全部评分与缺失数</a> · <a href="controls.json">几何与因果控制</a> · <a href="threshold_probe/results.json">阈值对照（观测重放）</a> · <a href="panels146.json">图像重建说明</a></p>
<p>各窗口文件夹含 protocol.json、completion.json、predictions.jsonl、predictions.mp4 与逐帧原生显示。处理时间大于20ms，不声称持续50Hz实时。</p></section></html>'''
    report = OUT/'阈值抑制与四三二桨预测实验.html'
    report.write_text(page, encoding='utf-8')
    print(json.dumps(dict(summary=summary, scores={name: {key: {h: dict(n=v['n'], missing=v['missing'], mean_px=v['mean_px']) for h, v in values.items()}
         for key, values in scores[name].items() if isinstance(values, dict)} for name in scores}, report=str(report)), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
