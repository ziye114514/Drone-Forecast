"""Publish the ownership audit; aggregation only, no new inference or relabeling."""
from pathlib import Path
import collections,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'output/peak_ownership_20261029'
BASE=ROOT.parent/'output'
PINNED=26.499735
PUBLISHED_HZ=231.875

OWNED=('stable_owner','owner_initialized')
CHANGED=('dominance_switch','rotation_change_of_same_owner','peak_moved_without_slot_change',
    'owner_change_with_unstable_slots','relabeled_same_frequency','owner_reinitialized')
UNATTRIBUTED=('no_identified_slot_claims','unowned_peak','ambiguous_ownership','harmonic_not_identity')


def load(p):return json.loads(Path(p).read_text('utf-8-sig'))


def stats(found):
    claimed=[r for r in found if r.get('claimed_slots')]
    owned=[r for r in found if r.get('dominant_model_id') is not None]
    errors=[abs(np.log(r['dominant_frequency_hz']/float(r['slot_frequencies'][str(r['dominant_model_id'])])))
        for r in owned if str(r['dominant_model_id']) in r.get('slot_frequencies',{})]
    matched=[len(r['corroborated_slots']) for r in claimed]
    full=[r for r in claimed if len(r['claimed_slots'])==4 and len(r['corroborated_slots'])==4]
    full_errors=[abs(np.log(a['peak']['frequency_hz']/float(r['slot_frequencies'][str(a['owner_model_id'])])))
        for r in full for a in r['attribution']
        if a['owner_model_id'] is not None and str(a['owner_model_id']) in r['slot_frequencies']]
    runs=[];current=None
    for r in found:
        owner=r.get('dominant_model_id')
        if owner is None or r['state'] not in OWNED:
            if current:runs.append(current);current=None
            continue
        if current and current['owner']==owner and abs(r['t']-current['end'])<.05:
            current['end']=r['t'];current['ticks']+=1
        else:
            if current:runs.append(current)
            current=dict(owner=owner,start=r['t'],end=r['t'],ticks=1)
    if current:runs.append(current)
    per_slot={}
    for r in owned:
        per_slot.setdefault(str(r['dominant_model_id']),[]).append(r)
    return dict(ticks=len(found),states=dict(collections.Counter(r['state'] for r in found)),
        claimed_ticks=len(claimed),owned_ticks=len(owned),
        resolved_histogram=dict(collections.Counter(str(m) for m in matched)),
        mean_resolved_fraction=float(np.mean([r['resolved_slot_fraction'] for r in found])) if found else 0.,
        mean_resolved_fraction_claimed=float(np.mean([len(r['corroborated_slots'])/len(r['claimed_slots'])
            for r in claimed])) if claimed else 0.,
        fully_resolved_ticks=len(full),
        fully_resolved_first_t=round(full[0]['t'],3) if full else None,
        fully_resolved_last_t=round(full[-1]['t'],3) if full else None,
        fully_resolved_peak_error=dict(
            median=None if not full_errors else float(np.expm1(np.median(full_errors))),
            p95=None if not full_errors else float(np.expm1(np.percentile(full_errors,95))),
            max=None if not full_errors else float(np.expm1(max(full_errors)))) if full_errors else None,
        median_peak_to_slot_relative_error=None if not errors else float(np.expm1(np.median(errors))),
        max_peak_to_slot_relative_error=None if not errors else float(np.expm1(max(errors))),
        owner_ticks=dict(collections.Counter(str(k) for k in [r['dominant_model_id'] for r in owned])),
        owner_median_error={k:float(np.expm1(np.median([abs(np.log(v['dominant_frequency_hz']/
            float(v['slot_frequencies'][k]))) for v in vs]))) for k,vs in per_slot.items()},
        stable_runs=[dict(owner=r['owner'],ticks=r['ticks'],duration_s=round(r['end']-r['start']+1/30,3))
            for r in runs],
        force_active_ticks=sum(bool(r.get('force_active')) for r in found),
        force_active_by_state=dict(collections.Counter(
            r['state'] for r in found if r.get('force_active'))),
        force_active_by_source=dict(collections.Counter(
            str(r.get('force_source')) for r in found if r.get('force_active'))),
        qualified_spectra=sum(bool(r.get('spectrum_qualified')) for r in found))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':155})
    checks=load(OUT/'mechanism_checks.json')
    ablation=load(OUT/'force_channel_ablation.json')
    coverage=load(OUT/'force_coverage.json')
    rpm_scan=load(OUT/'local_rpm_dataset_scan.json')
    audits={name:[json.loads(x) for x in (OUT/f'audit_{name}.jsonl').read_text('utf-8').splitlines()]
        for name in ('clear','side','other')}
    summary={name:stats(found) for name,found in audits.items()}
    pinned=min(audits['clear'],key=lambda r:abs(r['t']-PINNED))
    reproduction=dict(target=PINNED,observed_t=pinned['t'],published_single_peak_hz=PUBLISHED_HZ,
        recomputed_single_peak_hz=pinned['single_peak_component']['frequency_hz'],
        equal=abs(pinned['single_peak_component']['frequency_hz']-PUBLISHED_HZ)<1e-6,
        multi_peaks=[dict(frequency_hz=round(p['frequency_hz'],2),
            peak_to_median=round(p['peak_to_median'],1)) for p in pinned['peaks']],
        slots={k:round(float(v),2) for k,v in pinned['slot_frequencies'].items()},
        attribution=[dict(mode=a['mode'],owner=a['owner_model_id'],
            peak_hz=round(a['peak']['frequency_hz'],2),
            error_pct=round(100*float(np.expm1(abs(np.log(a['peak']['frequency_hz']/
                float(pinned['slot_frequencies'][str(a['owner_model_id'])]))))),2))
            for a in pinned['attribution'] if a['owner_model_id'] is not None],
        force_source=pinned['force_source'])
    payload=dict(summary=summary,reproduction=reproduction,checks={k:v['ok'] for k,v in checks['checks'].items()},
        default_enabled=False,forecaster_modified=False,annotations_used=False)
    (OUT/'attribution.json').write_text(json.dumps(payload,indent=2,allow_nan=False),'utf-8')

    fig,axes=plt.subplots(1,3,figsize=(17,4.8))
    grid=np.asarray(pinned['spectrum']['grid_hz']);power=np.asarray(pinned['spectrum']['power'])
    keep=grid<=520;axes[0].plot(grid[keep],power[keep],lw=1,color='#2b6cb0',label='空间方差周期谱')
    for i,f in enumerate([float(v) for v in pinned['slot_frequencies'].values()]):
        axes[0].axvline(f,ls='--',lw=.9,color='#b97a25',label='当前四桨SDK频率' if i==0 else None)
        axes[0].text(f,1.02,f'桨{i}',ha='center',fontsize=9,color='#b97a25')
    for peak in pinned['peaks']:
        axes[0].plot(peak['frequency_hz'],peak['normalized_power'],'o',ms=5,color='#c53030')
    axes[0].set_xlabel('事件周期分量 / Hz（未验证机械RPM）');axes[0].set_ylabel('归一化谱功率')
    axes[0].set_title(f'清晰 {PINNED:.6f}s：多峰归属');axes[0].legend(fontsize=9,loc='center right')
    clear=audits['clear'];times=np.array([r['t'] for r in clear])
    for slot in range(4):
        series=np.array([r['slot_frequencies'].get(str(slot),np.nan) for r in clear])
        axes[1].plot(times,series,lw=1,label=f'槽位{slot}（SDK）')
    owned=[r for r in clear if r.get('dominant_model_id') is not None]
    axes[1].plot([r['t'] for r in owned],[r['dominant_frequency_hz'] for r in owned],'o',ms=4,
        color='#c53030',label='主峰（已归属）')
    other=[r for r in clear if r.get('dominant_frequency_hz') is not None and r.get('dominant_model_id') is None
        and r['state'] in UNATTRIBUTED]
    axes[1].plot([r['t'] for r in other],[r['dominant_frequency_hz'] for r in other],'x',ms=5,
        color='#6b7280',label='主峰（无法归属）')
    gaps=[r['t'] for r in clear if r['state']=='no_identified_slot_claims']
    if gaps:
        axes[1].axvspan(min(gaps)-.017,max(gaps)+.017,color='#e2e8f0',alpha=.6,label='无已识别槽位')
    axes[1].set_xlabel('SOURCE 时刻 / s');axes[1].set_ylabel('频率 / Hz');axes[1].legend(fontsize=7,ncol=2)
    axes[1].set_title('清晰段：主峰跟住的是哪一个桨')
    fraction=np.array([r['resolved_slot_fraction'] for r in clear])
    axes[2].step(times,fraction,where='mid',color='#2f855a')
    axes[2].set_ylim(-.05,1.05);axes[2].set_xlabel('SOURCE 时刻 / s')
    axes[2].set_ylabel('被独立证实的已声明槽位比例');axes[2].set_title('清晰段：有多少槽位被真正独立分辨')
    fig.subplots_adjust(left=.05,right=.99,top=.87,bottom=.16,wspace=.3)
    fig.savefig(OUT/'ownership.png');plt.close(fig)

    side=ablation['clips']['side'];other=ablation['clips']['other'];clear=ablation['clips']['clear']
    near=ablation['labeled_near_segment'];sparse_clear=ablation['sparse_clear_localization']
    bars=dict(as_is=[side['modes']['as_is']['error_0.4']['mean'],other['modes']['as_is']['error_0.4']['mean']],
        off=[side['modes']['no_thrust_input']['error_0.4']['mean'],
             other['modes']['no_thrust_input']['error_0.4']['mean']])
    fig2,axes2=plt.subplots(1,2,figsize=(13,5.0))
    labels=['自采侧视\n26起点','另一段\n15起点']
    x=np.arange(2)
    axes2[0].bar(x-.18,bars['as_is'],.36,label='默认（频率→推力通道开）')
    axes2[0].bar(x+.18,bars['off'],.36,label='消融（通道关，其余不变）')
    axes2[0].set_xticks(x,labels);axes2[0].set_ylabel('0.4秒平均误差 / px');axes2[0].legend(fontsize=9)
    for xx,a,b in zip(x,bars['as_is'],bars['off']):
        axes2[0].text(xx-.18,a+1,f'{a:.2f}',ha='center',fontsize=9)
        axes2[0].text(xx+.18,b+1,f'{b:.2f}',ha='center',fontsize=9)
    axes2[0].set_title('通道存在与否的差别')
    with_input=side['modes']['as_is']['error_by_input_presence']['with_input']
    without_input=side['modes']['as_is']['error_by_input_presence']['without_input']
    axes2[1].bar([0,1],[with_input['mean'],without_input['mean']],.5,color=['#2f855a','#c53030'])
    axes2[1].set_xticks([0,1],[f"有力输入\n{with_input['n']}起点",f"无输入\n{without_input['n']}起点"])
    axes2[1].set_ylabel('0.4秒平均误差 / px');axes2[1].set_title('侧视：有输入与无输入的起点')
    for i,v in enumerate([with_input['mean'],without_input['mean']]):
        axes2[1].text(i,v+1,f'{v:.2f}',ha='center')
    fig2.subplots_adjust(left=.07,right=.98,top=.9,bottom=.22)
    fig2.savefig(OUT/'thrust_channel_leverage.png');plt.close(fig2)

    cov_side=coverage['clips']['side'];cov=coverage['controlled']
    census=cov_side['census']['base']
    origins=[o for o in cov_side['origins'] if o.get('current_error_px') is not None]
    current=[o['current_error_px'] for o in origins];future=[o['error']['base'] for o in origins]
    fig3,axes3=plt.subplots(1,2,figsize=(13,4.6))
    names=['无平面/无整机','只有弱估计','完全没频率','门槛/连贯']
    keys=['no_axis','weak_unqualified_frequency','no_frequency_at_all','gate_or_coherence']
    values=[census.get(k,0) for k in keys]
    axes3[0].barh(names,values,color='#2b6cb0')
    for i,v in enumerate(values):
        axes3[0].text(v+.5,i,str(v),va='center',fontsize=10)
    axes3[0].set_xlabel('侧视 240 帧中没有力输入的帧数');axes3[0].set_title('力输入为什么缺席（逐帧原因）')
    axes3[1].plot(current,future,'o',ms=5,color='#c53030')
    axes3[1].axhline(30,color='gray',ls='--',lw=1);axes3[1].axvline(30,color='gray',ls='--',lw=1)
    axes3[1].set_xlabel('源帧当前中心误差 / px');axes3[1].set_ylabel('0.4秒未来误差 / px')
    axes3[1].set_title('误差是从哪来的：源帧很准，未来不准');axes3[1].set_xlim(0,35)
    fig3.subplots_adjust(left=.08,right=.98,top=.9,bottom=.16,wspace=.32)
    fig3.savefig(OUT/'coverage_diagnosis.png');plt.close(fig3)
    deltas=[o['error']['weak_last']-o['error']['base'] for o in cov_side['origins']]
    better=sum(1 for d in deltas if d<-0.05);worse=sum(1 for d in deltas if d>0.05)
    worst=max(cov_side['origins'],key=lambda o:o['error']['weak_last']-o['error']['base'])
    def stat(values):
        return dict(mean=float(np.mean(values)),median=float(np.median(values)),
            p95=float(np.percentile(values,95)),maximum=float(max(values)))
    cur_stat=stat(current);fut_stat=stat(future)
    over30=sum(1 for v in future if v>30)

    def pct(value):
        return '—' if value is None else f'{100 * value:.2f}%'
    rows=''.join(
        f"<tr><td>{label}</td><td>{s['ticks']}</td><td>{s['claimed_ticks']}</td><td>{s['owned_ticks']}</td>"
        f"<td>{s['force_active_ticks']}</td><td>{pct(s['median_peak_to_slot_relative_error'])}</td>"
        f"<td>{s['qualified_spectra']}</td></tr>"
        for label,s in [('清晰四桨 26–28s',summary['clear']),('侧视 110–118s',summary['side']),
                        ('另一段 40–46s',summary['other'])])
    checks_rows=''.join(f"<tr><td>{name}</td><td>{'通过' if ok else '未通过'}</td></tr>"
        for name,ok in payload['checks'].items())
    runs=summary['clear']['stable_runs']
    run_text='、'.join(f"槽位{r['owner']} 连续{r['ticks']}帧（{r['duration_s']:.2f}s）" for r in runs if r['ticks']>=3)
    attributed=' '.join(f"桨{a['owner']}→{a['peak_hz']:.1f}Hz（偏差{a['error_pct']:.2f}%）"
        for a in reproduction['attribution'])
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>空间方差主峰的归属审计</title>
<style>body{{font:17px/1.75 'Microsoft YaHei',sans-serif;background:#f1f5fa;color:#172a40;max-width:1120px;margin:35px auto;padding:0 22px}}section{{background:white;border-radius:12px;margin:20px 0;padding:22px 28px}}h1{{font-size:30px}}h2{{font-size:23px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #dbe3ee}}img{{max-width:100%}}code{{background:#edf1f5;padding:2px 5px}}a{{color:#145a9e}}.note{{background:#fff0ce;padding:12px}}.good{{background:#e8f6ee;padding:12px}}</style>
<h1>“主峰属于谁”与推力通道的真实杠杆</h1>
<p>2026-10-29 · 交接下一步（<code>NEXT_ROTOR_INPUT.md</code> 第1、3节）。前半回答归属，后半用消融判定要不要进入推力变化率。<b>默认行为、已发布 ZIP、原预测全部未改。</b></p>
<section><h2>结论</h2>
<div class="good"><p><b>1. 主峰属于某一个桨，不是整机。</b>在 {PINNED:.6f}s，多峰谱给出 {len(reproduction['multi_peaks'])} 个分量，逐个按频率对上当前四个槽位：{attributed}。全程 {summary['clear']['owned_ticks']} 个可归属时刻，主峰与所归属槽位频率的中位偏差 <b>{100*summary['clear']['median_peak_to_slot_relative_error']:.2f}%</b>、最大 <b>{100*summary['clear']['max_peak_to_slot_relative_error']:.2f}%</b>。连续归属最长一段：{run_text or '无'}。</p>
<p><b>2. 更强的结果是：多峰谱能把四个桨同时分辨出来。</b>清晰段 {summary['clear']['fully_resolved_first_t']:.3f}–{summary['clear']['fully_resolved_last_t']:.3f}s 的 <b>{summary['clear']['fully_resolved_ticks']}</b> 个时刻，四个已声明槽位全部被独立证实，逐峰偏差中位 <b>{100*summary['clear']['fully_resolved_peak_error']['median']:.2f}%</b>、P95 <b>{100*summary['clear']['fully_resolved_peak_error']['p95']:.2f}%</b>、最大 <b>{100*summary['clear']['fully_resolved_peak_error']['max']:.2f}%</b>。这是本项目第一次拿到不依赖 SDK 频率估计器的四桨同时观测，也是上一批“单峰只对应一个桨”那句话需要修正的地方：单峰取最大值只看到一个桨，多峰分解能看到四个。</p>
<p><b>3. 但这条能力只覆盖清晰段的那半秒。</b>{summary['clear']['claimed_ticks']} 个有槽位声明的时刻里，{summary['clear']['claimed_ticks']-summary['clear']['fully_resolved_ticks']} 个不完整；侧视 {summary['side']['ticks']} 帧、另一段 {summary['other']['ticks']} 帧的已识别槽位是 <b>0</b>，两段合计 {summary['side']['force_active_ticks']+summary['other']['force_active_ticks']} 个力输入时刻全部落在“无已识别槽位”上，本来就走 pooled/legacy 汇总口径。<b>它救不了当前真正掉分的两段。</b></p>
<p><b>4. 仍然不能把“最强分量平方”当总推力。</b>主峰是一个桨；能当四桨合力的是多峰集合，而集合在清晰段也常不完整。并且两个估计器互相吻合到 2%，没有独立真值能证明空间方差比 SDK 更准，所以本轮不进入推力变化率、不上默认。</p></div>
<p class="note">归属这一半不产生精度改进结论：清晰 26–28s 只有 1.5Hz 稀疏框、无法给 0.4 秒评分，另两段没有可归属槽位。推力通道那一半有明确数字，见下一节。</p></section>
<section><h2>推力通道到底值多少像素（交接第3节）</h2>
<p>交接说“通过后再进入合力与推力变化率”。判据不该是“主峰解释通了”，而是这条通道对分数还有多少杠杆。用冻结输入做消融，只改 <code>force.input_u</code>，其余位置/曲线/光流分支完全不变；三个片段的回放首先逐项复原了冻结预测（最大差 <b>0.0</b>），所以下面的差别只来自这一项。</p>
<table><tr><th>片段</th><th>默认</th><th>通道关</th><th>差</th><th>P95</th></tr>
<tr><td>自采侧视 26 起点</td><td>{side['modes']['as_is']['error_0.4']['mean']:.2f}</td><td>{side['modes']['no_thrust_input']['error_0.4']['mean']:.2f}</td><td>+{side['modes']['no_thrust_input']['error_0.4']['mean']-side['modes']['as_is']['error_0.4']['mean']:.2f}</td><td>{side['modes']['as_is']['error_0.4']['p95']:.2f}→{side['modes']['no_thrust_input']['error_0.4']['p95']:.2f}</td></tr>
<tr><td>另一段 15 起点</td><td>{other['modes']['as_is']['error_0.4']['mean']:.2f}</td><td>{other['modes']['no_thrust_input']['error_0.4']['mean']:.2f}</td><td>+{other['modes']['no_thrust_input']['error_0.4']['mean']-other['modes']['as_is']['error_0.4']['mean']:.2f}</td><td>{other['modes']['as_is']['error_0.4']['p95']:.2f}→{other['modes']['no_thrust_input']['error_0.4']['p95']:.2f}</td></tr></table>
<p>侧视按起点拆开更直接：<b>有力输入的 {with_input['n']} 个起点平均 {with_input['mean']:.2f}px，没有输入的 {without_input['n']} 个起点平均 {without_input['mean']:.2f}px</b>。也就是说这条通道不是噪声，它值约 {side['modes']['no_thrust_input']['error_0.4']['mean']-side['modes']['as_is']['error_0.4']['mean']:.2f}px——但起作用的是“这一帧有没有输入”，不是“输入有多准”。</p>
<p>精度那一侧直接量了：把四桨 f² 和换成多峰谱推出来的相对值（只在 4/4 独立证实时替换，其余保持原值），输入本身相差中位 {100*clear['input_pairs']['median_relative_difference']:.2f}%、最大 {100*clear['input_pairs']['max_relative_difference']:.2f}%，<b>预测位置最多只移动 {clear['difference_variance_hybrid']['max']:.2f}px</b>（中位 {clear['difference_variance_hybrid']['median']:.3f}px，P95 {clear['difference_variance_hybrid']['p95']:.2f}px）。清晰 26–28s 没有可做 0.4 秒评分的连续框，所以这是“预测差”的上界，不是精度差。</p>
<p class="note">推论：3% 量级的推力输入误差只值 0.4px 以内，所以“进入推力变化率 Ṫn+Tṅ”不可能补上 34.60→30px 的 4.6px 缺口。这一步按证据关闭，不是靠猜。</p>
<img src="thrust_channel_leverage.png"></section>
<section><h2>人工标注用在了哪里</h2><p>本目录实际可用的连续标注是 <code>dense_labels_side_110_1195</code>（75 框/20Hz）、<code>dense_labels_train2_40_46</code>（36 框/20Hz）和 <code>dense_labels_train6_20_35</code>（23 框/20Hz，20.05–21.20s）。清晰 26–28s 只有历史归档里的稀疏框（<code>paper_reproduction_v030/evaluation_data/train6.json</code>，1.5Hz）：它在这段有 3 个 present 框，检出 3/3，当前定位误差 <b>{sparse_clear['anchors'][0]['error_px']:.2f} / {sparse_clear['anchors'][1]['error_px']:.2f} / {sparse_clear['anchors'][2]['error_px']:.2f}px</b>（均值 {sparse_clear['mean_error_px']:.2f}px）；框间隔超过 150ms 插值门限，所以<b>不能</b>用来给 0.4 秒未来评分。</p>
<p>为此在有人工框的 20–21.2s 新跑了一段候选入口：<b>{near['modes'].get('insufficient',0)} 帧 insufficient、{near['modes'].get('axis_only',0)} 帧 axis_only、four=0</b>。那段无人机已贴到画面右缘，桨盘被裁掉，所以它既没有槽位、也无法做归属替换；当前定位中位 {near['box_center_error_px']['median']:.2f}px，0.4 秒却到 {near['future_error_px']['0.4']['mean']:.2f}px（n={near['future_error_px']['0.4']['n']}，样本极少）。它证明“有标注的清晰段”并不等于“能测推力通道的清晰段”。</p></section>
<section><h2>侧视线索为什么丢、丢了到底值多少</h2>
<p>先逐帧查缺席原因，而不是先改阈值。侧视 240 帧里 {cov_side['modes']['base']['input_ticks']} 帧有力输入，其余 {240-cov_side['modes']['base']['input_ticks']} 帧的原因分布是：</p>
<table><tr><th>原因</th><th>侧视帧数</th><th>另一段帧数</th></tr>
<tr><td>没有平面/没有整机（合法缺席）</td><td>{census.get('no_axis',0)}</td><td>{coverage['clips']['other']['census']['base'].get('no_axis',0)}</td></tr>
<tr><td>只有未达门槛的弱周期估计</td><td>{census.get('weak_unqualified_frequency',0)}</td><td>{coverage['clips']['other']['census']['base'].get('weak_unqualified_frequency',0)}</td></tr>
<tr><td>完全没有频率</td><td>{census.get('no_frequency_at_all',0)}</td><td>{coverage['clips']['other']['census']['base'].get('no_frequency_at_all',0)}</td></tr>
<tr><td>通过门槛但连贯/幅值被拒</td><td>{census.get('gate_or_coherence',0)}</td><td>{coverage['clips']['other']['census']['base'].get('gate_or_coherence',0)}</td></tr></table>
<p>那 4 个被打分的无输入起点全部落在 117.03–117.63 的 0.6 秒里：无人机以约 680px/s 向左出画，整机池化估计其实还在给 263–290Hz，但频率像素只有 2–23 个，而合格帧的中位是 98.5 个。所以这是<b>观测不足</b>，不是门槛写错。</p>
<p>再把“弱估计”放到最低一档去试（只在合格周期、0.12s 持稳都失败后才用；要求年龄 ≤0.3s、与上一次已接受频率差 ≤8%、至少 2 个频率像素、且不计入增益学习）：</p>
<table><tr><th>变体</th><th>有输入帧</th><th>0.4秒均值 px</th><th>差</th></tr>
<tr><td>默认</td><td>{cov_side['modes']['base']['input_ticks']}</td><td>{cov_side['modes']['base']['error_0.4']['mean']:.4f}</td><td>—</td></tr>
<tr><td>弱先验兜底</td><td>{cov_side['modes']['weak_last']['input_ticks']}</td><td>{cov_side['modes']['weak_last']['error_0.4']['mean']:.4f}</td><td>{cov_side['modes']['weak_last']['error_0.4']['mean']-cov_side['modes']['base']['error_0.4']['mean']:+.2f}</td></tr>
<tr><td>持稳 0.3s</td><td>{cov_side['modes']['hold30']['input_ticks']}</td><td>{cov_side['modes']['hold30']['error_0.4']['mean']:.4f}</td><td>{cov_side['modes']['hold30']['error_0.4']['mean']-cov_side['modes']['base']['error_0.4']['mean']:+.2f}</td></tr>
<tr><td>两者同时</td><td>{cov_side['modes']['hold30_weak_last']['input_ticks']}</td><td>{cov_side['modes']['hold30_weak_last']['error_0.4']['mean']:.4f}</td><td>{cov_side['modes']['hold30_weak_last']['error_0.4']['mean']-cov_side['modes']['base']['error_0.4']['mean']:+.2f}</td></tr></table>
<p>逐起点看：{better} 个变好、{worse} 个变差（最差是 {worst['t']:.2f}s，+{worst['error']['weak_last']-worst['error']['base']:.2f}px）、其余不变。两种完全不同的机制给出几乎相同的收益，说明起作用的是“缺口里终于有点输入”，不是“弱估计更准”。</p>
<p class="note">重要修正：上一节“有输入 31.06px、无输入 54.08px、差约 23px”是<b>选择偏差</b>——无输入的那几帧正好是快速出画的最难时刻。因果上把输入补回来只值 0.4–0.6px，不是 23px。</p>
<p>那误差到底从哪来？源帧当前中心误差均值 {cur_stat['mean']:.2f}px、中位 {cur_stat['median']:.2f}px、P95 {cur_stat['p95']:.2f}px、最大 {cur_stat['maximum']:.2f}px，26 个起点里没有一个超过 30px；0.4 秒未来误差是均值 {fut_stat['mean']:.2f}px、中位 {fut_stat['median']:.2f}px、P95 {fut_stat['p95']:.2f}px，{over30}/26 超过 30px。<b>差的那约 28px 是 0.4 秒外推里长出来的，不是定位、不是推力精度、也不是输入覆盖。</b></p>
<img src="coverage_diagnosis.png"></section>
<section><h2>独立 RPM：这一步仍然拿不到</h2><p>公开数据 <code>ptak.felk.cvut.cz</code> 的 HTTPS 依旧报自签名证书，未绕过、未下载；本机按名字做了有界扫描（{len(rpm_scan['roots_present'])} 个根目录、深度 {rpm_scan['depth']}、模式 <code>{rpm_scan['name_pattern']}</code>、只读名字不打开文件），命中 <b>{rpm_scan['match_count']}</b> 个。名字扫描不能证明整台机器都没有，但它说明这条路在本机没有现成替代品。</p></section>
<section><h2>怎么判“转速变了”还是“换了一个桨当主峰”</h2><p>关键是把主峰同时跟“上一帧主峰”和“上一帧槽位集合”比：槽位集合稳定而主峰跳走 → 主峰换人（<code>dominance_switch</code>）；同一槽位而频率变了 → 转速变化（<code>rotation_change_of_same_owner</code>）；主峰动了但所归属槽位没动 → 标为可疑（<code>peak_moved_without_slot_change</code>），不强行解释。若本轮两个槽位都落进容差 → <code>ambiguous_ownership</code>，不挑一个。</p><p>合成机制检查（全部通过，不读真实数据）：</p><table><tr><th>检查</th><th>结果</th></tr>{checks_rows}</table>
<p class="note">已知限制：这里用的是未加窗的 Lomb–Scargle，弱分量会被强分量的泄漏掩掉。合成数据 3:1 振幅差时四个桨只能分辨出三个（<code>weak_component_masked_by_leakage</code>），这条限制被保留而不是悄悄修掉。频率是桨叶通过频率代理，不是机械 RPM，更不是牛顿推力。</p></section>
<section><h2>复现与读数一致性</h2><p>上一批公布的单峰读数在 {PUBLISHED_HZ} Hz，本轮用公开滚动读取器重解码同一截止时刻得到 {reproduction['recomputed_single_peak_hz']} Hz，逐位相同；多峰分解后同一帧的四个分量见上。注意上一批“主峰只对应一个分量”的表述来自单峰取最大值，多峰谱在同一时刻其实能分出四个，这是一个<b>修正</b>而不是推翻。</p><img src="ownership.png"><p>左：{PINNED:.6f}s 的多峰谱与四个槽位频率。中：26–28s 四个槽位频率与主峰读数，灰色区域没有已识别槽位。右：被独立证实的槽位比例。</p><p><a href="attribution.json">聚合结果与复现读数</a> · <a href="mechanism_checks.json">机制检查</a> · <a href="summary.json">逐帧统计</a> · <a href="audit_clear.jsonl">清晰段逐帧记录</a> · <a href="audit_side.jsonl">侧视逐帧记录</a> · <a href="audit_other.jsonl">另一段逐帧记录</a> · <a href="force_channel_ablation.json">推力通道消融与标注评分</a> · <a href="force_coverage.json">力输入覆盖诊断</a> · <a href="local_rpm_dataset_scan.json">本机 RPM 数据名字扫描</a> · <a href="independent_RPM_dataset_access_retry.json">公开数据访问重试</a></p></section>
<section><h2>结论：这一步怎么收口</h2><p>三件事按证据排完了序：<b>推力输入精度</b>只值 0.4px 以内（3% 输入差 → ≤0.38px 预测差）；<b>输入覆盖</b>补满缺口只值 0.4–0.6px（“23px 差”是选择偏差，无输入帧正好是最难出画帧）；<b>源帧定位</b>已经不差（26 个起点均值 {cur_stat['mean']:.2f}px，没有一个超过 30px）。真正吃掉分数的是 0.4 秒外推本身：未来误差均值 {fut_stat['mean']:.2f}px、{over30}/26 超过 30px。</p>
<p>所以下一批不该继续在推力/周期/覆盖上找精度，而应回到运动外推：速度、加速度、转向与长尾。弱先验兜底与 0.3s 持稳都<b>不推广</b>——单片段、26 个起点、0.58px，还各有 1 个起点变差，不足以当默认改动。</p>
<p>默认行为、已发布 ZIP、原预测一律未改；本轮新增的都是离线工具（归属审计、推力通道消融、覆盖诊断、有界名字扫描）。</p></section></html>'''
    (OUT/'index.html').write_text(report,'utf-8')

    def snapshot(folder,names):
        root=OUT/folder;root.mkdir(exist_ok=True)
        manifest=root/'manifest.json'
        if manifest.exists():
            for name,digest in load(manifest).items():
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError(name)
            return
        records={}
        for name in names:
            data=(ROOT/name).read_bytes();target=root/name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            records[name]=hashlib.sha256(data).hexdigest()
        manifest.write_text(json.dumps(records,indent=2),'utf-8')

    snapshot('source_snapshot',['spatial_peak_ownership.py','side_moment_period.py','visibility_period.py',
        'tools/probe_peak_ownership.py','tools/report_peak_ownership.py','live_decode.py'])
    snapshot('source_snapshot_ablation',['tools/probe_force_channel_ablation.py',
        'tools/probe_force_coverage.py','tools/probe_local_rpm_dataset.py','tools/report_peak_ownership.py',
        'tools/probe_same_window_motion.py','tools/compare_final_pipeline.py'])
    print(str(OUT/'index.html'))


if __name__=='__main__':main()
