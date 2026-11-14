"""Build the stage10 HTML report from frozen run outputs.

Reads protocol/summary/analysis/labels JSON only; never re-runs inference and
never changes a frozen definition. Figure labels are ASCII on purpose so the
plot does not depend on an installed CJK font; Chinese text lives in the HTML.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def figure(run, destination):
    analysis = load(run / 'analysis.json')
    if analysis is None:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for method, payload in analysis['methods'].items():
        seconds = [row['second'] for row in payload['per_second']]
        axes[0].plot(seconds, [row['detected_fraction'] for row in payload['per_second']],
                     label=method, linewidth=1.0)
        axes[1].plot(seconds, [row['median_extent'] for row in payload['per_second']],
                     label=method, linewidth=1.0)
        axes[2].plot(seconds, [row['median_rotors'] for row in payload['per_second']],
                     label=method, linewidth=1.0)
    axes[0].set_ylabel('detected fraction / s')
    axes[1].set_ylabel('median whole extent (px)')
    axes[2].set_ylabel('median rotor count')
    axes[2].set_xlabel('RAW time (s)')
    for axis in axes:
        axis.grid(alpha=.3)
        axis.legend(fontsize=8)
    axes[1].set_yscale('log')
    fig.suptitle(f"stage10 continuous audit: {run.name}")
    fig.tight_layout()
    fig.savefig(destination, dpi=110)
    plt.close(fig)
    return destination.name


def overview_row(name, method, analysis, summary):
    return (name, method,
            analysis['ticks'], round(analysis['coverage'], 3),
            analysis['four_rotor_ticks'], analysis['layout_inferred_ticks'],
            len(analysis['fragments']), len(analysis['rotor_swaps']),
            analysis['ambiguous_pairs'], round(analysis['rigid_fraction'], 3),
            len(analysis['transfers']), analysis['longest_gap_ticks'],
            None if analysis['median_best_residual'] is None
            else round(analysis['median_best_residual'], 2),
            analysis['rotor_count_histogram'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    runs = [Path(item.strip()).resolve() for item in args.runs.split(',') if item.strip()]
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for run in runs:
        protocol = load(run / 'protocol.json')
        summary = load(run / 'summary.json')
        analysis = load(run / 'analysis.json')
        labels = load(run / 'labels.json')
        if protocol is None or analysis is None:
            blocks.append(f'<section><h2>{run.name}</h2><p>缺少 protocol.json 或 analysis.json，未纳入报告。</p></section>')
            continue
        image = figure(run, out.parent / f'{run.name}_timeline.png')
        rows = ''.join(
            '<tr>' + ''.join(f'<td>{value}</td>' for value in overview_row(run.name, method, payload,
                                                                          summary))
            + '</tr>' for method, payload in analysis['methods'].items())
        label_rows = ''
        if labels:
            for method, payload in labels['methods'].items():
                label_rows += ('<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>'
                               '<td>{}</td><td>{}</td><td>{}</td></tr>').format(
                    method, payload['judged_present_instants'], payload['hit_instants'],
                    payload['miss_instants'], round(payload['instant_hit_rate'], 3),
                    payload['absent_instants_with_publication'], payload['labelled_absent_instants'],
                    payload['centre_error_native_px']['median'])
        episodes = []
        for method, payload in analysis['methods'].items():
            for item in payload['fragments'][:25]:
                episodes.append((method, '碎片', item['t'], f"extent={item['extent']:.1f} "
                                 f"baseline={item['baseline']:.1f} ratio={item['ratio']:.2f}"))
            for item in payload['rotor_swaps'][:25]:
                episodes.append((method, '串桨', item['t'],
                                 f"best={item['best_residual']:.1f} tol={item['tolerance']:.1f}"))
            for item in payload['transfers'][:25]:
                episodes.append((method, '重新捕获', item['t'], f"extent={item['extent']:.1f}"))
        false_eps = []
        if labels:
            for method, payload in labels['methods'].items():
                for item in payload['false_publication_episodes'][:25]:
                    false_eps.append((method, item['start'], item['end'], item['instants']))
        blocks.append(f'''<section>
<h2>{run.name}</h2>
<p>RAW：<code>{protocol['raw']}</code>　区间：{protocol['start_s']}–{protocol['start_s']+protocol['duration_s']}s　
步长：{protocol['step_us']/1000:.0f} ms　方法：{', '.join(protocol['methods'])}　冷启动、无隐藏预热</p>
<div class="scroll"><table>
<tr><th>运行</th><th>方法</th><th>tick</th><th>覆盖</th><th>四桨tick</th><th>布局推断</th><th>碎片</th>
<th>串桨</th><th>非刚性对</th><th>刚性比例</th><th>重新捕获</th><th>最长中断(10ms)</th><th>刚性残差中位</th>
<th>桨数直方图</th></tr>
{rows}</table></div>
{'<h3>稀疏标注事后核对</h3><div class="scroll"><table><tr><th>方法</th><th>present时刻</th><th>命中</th><th>漏检</th><th>命中率</th><th>absent时刻有发布</th><th>absent时刻</th><th>中心误差中位(px)</th></tr>' + label_rows + '</table></div>' if label_rows else ''}
<h3>失败事件（最多各25条）</h3>
<div class="scroll"><table><tr><th>方法</th><th>类型</th><th>t (s)</th><th>说明</th></tr>
{''.join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>' for a, b, c, d in episodes) or '<tr><td colspan="4">无</td></tr>'}
</table></div>
<h3>标注absent区间内的发布</h3>
<div class="scroll"><table><tr><th>方法</th><th>开始(s)</th><th>结束(s)</th><th>标注时刻数</th></tr>
{''.join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>' for a, b, c, d in false_eps) or '<tr><td colspan="4">无</td></tr>'}
</table></div>
{f'<p><img src="{image}" alt="timeline"></p>' if image else ''}
</section>''')
    html = ('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>第十阶段：自采连续段整机身份与恢复审计</title><style>
body{background:#eef3f7;color:#21324a;font:15px/1.8 "Microsoft YaHei",sans-serif;margin:0}
main{max-width:1300px;padding:25px;margin:auto}section{background:white;padding:18px;border-radius:10px;margin:18px 0}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:left;border-bottom:1px solid #d8e0e9;white-space:nowrap}
th{background:#dce8f5}img{max-width:100%}.scroll{overflow:auto}code{background:#f2f5f9;padding:2px 4px}
.note{border-left:4px solid #b77c23;background:#fff6e2;padding:15px}</style><main>
<h1>第十阶段：自采连续段整机身份与恢复审计</h1>
<div class="note">本页只做一件事：在自采的连续录像上，检查整机模型是否串桨、缩成碎片、重新捕获到错误目标。
所有数字来自冻结协议下的像素观测；四候选不等于四真桨，标注只在事后核对，时间对齐约±0.7 s。</div>
''' + ''.join(blocks) + '''<section><h2>边界</h2><ul>
<li>没有逐桨、姿态、RPM真值；帧间一致不等于身份正确。</li>
<li>稀疏标注是操作者画的整机粗框（ds4×4像素），不是质心或桨盘真值，刻度约0.67 s，标注之间的时刻不判定。</li>
<li>一次离线运行的耗时含解码与落盘，不是流式实时性证明。</li>
<li>失败目录与旧结果保留；本阶段不修改阶段6–9的源码与协议。</li>
</ul></section></main></html>''')
    out.write_text(html, encoding='utf-8')
    print(json.dumps(dict(written=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    main()
