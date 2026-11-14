"""Publish the rotor-only front-end pilot; aggregation only, no new inference."""
from pathlib import Path
import collections,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'output/rotor_support_20261101'


def load(path):return json.loads(Path(path).read_text('utf-8'))


def pct(value,digits=1):return '—' if value is None else f'{100*value:.{digits}f}%'


def main():
    summary=load(OUT/'summary.json')
    sequence=load(OUT/'sequence.json')
    tracks=load(OUT/'tracks.json')
    merge=load(OUT/'merge.json')
    disc=load(OUT/'disc_geometry.json')
    envelope=load(OUT/'sweep_envelope.json')
    frequency=load(OUT/'frequency_tracks.json')
    gate=load(OUT/'rotor_gate_forecast.json')
    coverage=load(OUT/'coverage_variants.json')
    validation=load(OUT/'period_validation.json')
    clips=summary['clips']
    names=[('clear','清晰四桨 26.5s'),('partial','部分遮挡 26.7s'),('side','侧视 111.0s'),
        ('negative','空场 0.5s')]
    rows=[]
    for key,label in names:
        clip=clips[key];counts=clip['counts'];ref=clip.get('frozen_reference') or {}
        distance=ref.get('support_distance') or {}
        residual=ref.get('residual') or {}
        median='—' if not distance else f"{distance['median']:.2f}"
        within='—' if not distance else pct(distance['within_1_semimajor'],0)
        far='—' if residual.get('density_only_far_fraction') is None else pct(residual['density_only_far_fraction'],0)
        rows.append(f"<tr><td>{label}</td><td>{counts['events']:,}</td><td>{counts['density_pixels']}</td>"
            f"<td>{counts['period_pixels']}</td><td>{counts['components']}</td>"
            f"<td>{median}</td><td>{within}</td><td>{far}</td></tr>")
    frequency_rows=''
    for key,label in names:
        clip=clips[key]
        for item in clip.get('frequency_check',[]):
            align=item.get('alignment') or {}
            frequency_rows+=f"<tr><td>{label}</td><td>{item['component_pixels']}</td>" \
                f"<td>{item['component_frequency_hz']:.0f}</td><td>{align.get('nearest_rate_hz',float('nan')):.1f}</td>" \
                f"<td>{align.get('ratio',float('nan')):.2f}</td><td>{align.get('harmonic_tag','—')}</td></tr>"
    parameters=summary['parameters']
    arms=[('A','逐帧（单窗）'),('B','累加，固定像素'),('C','累加＋给定速度'),('D','累加＋自估运动')]
    sequence_rows=''
    for name,entry in sequence['sequences'].items():
        for key,label in arms:
            arm=entry['arms'][key]
            sequence_rows+=(f"<tr><td>{entry['spec']['name']}</td><td>{label}</td>"
                f"<td>{arm['iou_raw_median']:.3f}</td><td>{arm['iou_object_median']:.3f}</td>"
                f"<td>{min(arm['support_pixels'])}–{max(arm['support_pixels'])}</td></tr>")
    estimate_error=sequence['sequences']['clear_26.4_26.8']['self_estimate_error_px']
    side_error=sequence['sequences']['side_110.9_111.3']['self_estimate_error_px']
    def spread_text(value):
        return '—' if value is None else f'{100*value:.0f}%'
    track_rows=''
    for name,entry in tracks['sequences'].items():
        confirmed=entry['confirmed_counts']
        track_rows+=(f"<tr><td>{entry['spec']['name']}</td><td>{entry['cutoffs']}</td>"
            f"<td>{entry['tracks_total']}</td><td>{entry['tracks_hits_ge5']}</td>"
            f"<td>{entry['median_age_s']:.3f}</td><td>{min(confirmed)}–{max(confirmed)}</td>"
            f"<td>{spread_text(entry['frequency_relative_std_median'])}</td></tr>")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':150})
    fig3,axes3=plt.subplots(1,3,figsize=(16,4.4))
    clear_entry=tracks['sequences']['clear_26.2_27.0']
    long_ids=[track['id'] for track in clear_entry['frames'][-1]['tracks'] if track['hits']>=5][:4]
    full_hits=[track for track in clear_entry['frames'][-1]['tracks'] if track['hits']>=clear_entry['cutoffs']]
    for track_id in long_ids:
        series=[]
        for frame in clear_entry['frames']:
            item=next((t for t in frame['tracks'] if t['id']==track_id),None)
            if item:
                x0,y0,x1,y1=item['bbox'];series.append((frame['t'],(y0+y1)/2))
        if series:
            axes3[0].plot([s[0] for s in series],[s[1] for s in series],'.-',ms=3,label=f'区域{track_id}')
    axes3[0].set_xlabel('截止时刻 / s');axes3[0].set_ylabel('区域中心 y / px')
    axes3[0].set_title('清晰段：四个区域跟住 0.8 秒');axes3[0].legend(fontsize=8)
    for axis,key,title in ((axes3[1],'clear_26.2_27.0','清晰段：确认区域数'),
                           (axes3[2],'side_110.9_111.7','侧视段：确认区域数')):
        entry=tracks['sequences'][key]
        times=[frame['t'] for frame in entry['frames']]
        axis.plot(times,entry['confirmed_counts'],'.-',label='已确认')
        axis.plot(times,entry['total_counts'],'.:',label='全部区域')
        axis.set_xlabel('截止时刻 / s');axis.set_ylabel('区域数');axis.set_title(title);axis.legend(fontsize=8)
    fig3.subplots_adjust(left=.06,right=.99,top=.9,bottom=.16,wspace=.28)
    fig3.savefig(OUT/'track_regions.png');plt.close(fig3)
    diagnostic=None
    reference=(ROOT.parent/'output'/clear_entry['spec']['predictions']).resolve()
    if reference.exists():
        frozen_rows=[json.loads(line) for line in reference.read_text('utf-8').splitlines()]
        frame=min(clear_entry['frames'],key=lambda item:abs(item['t']-26.5))
        row=min(frozen_rows,key=lambda item:abs(item['t']-26.5))
        rotors=[r for r in row['geometry'].get('rotors',[]) if r.get('current_observation',True)]
        qualified=[item for item in frame['tracks'] if item['qualified']]
        distances=[]
        for item in qualified:
            x0,y0,x1,y1=item['bbox'];cx=(x0+x1)/2;cy=(y0+y1)/2
            distances.append(min(float(np.hypot(cx-r['cx'],cy-r['cy'])/max(float(r.get('major',1))/2,1e-6))
                for r in rotors))
        diagnostic=dict(target=26.5,frozen_mode=row['structure'].get('mode'),frozen_rotors=len(rotors),
            qualified_tracks=len(qualified),
            max_distance_in_semi_major=None if not distances else float(max(distances)),
            all_within_one_semi_major=True if not distances else bool(max(distances)<=1.),
            long_track_ids=long_ids,
            note='diagnostic against the frozen current rotor conics; several tracks may sit on one rotor, '
                 'so this does not establish one-to-one rotor identity')
    merge_rows=''
    for name,curve in merge['gap_curve'].items():
        cells=''.join(f"<td>{curve[str(gap)]['median']:.0f}<br><span style='font-size:12px;color:#718096'>"
            f"{curve[str(gap)]['minimum']}–{curve[str(gap)]['maximum']}</span></td>" for gap in (4,8,12,16,24,32,48))
        merge_rows+=f"<tr><td>{name}</td>{cells}</tr>"
    env_frames=envelope['clips']['clear_26.2_27.0']['frames']
    env_frozen=[json.loads(line) for line in (ROOT.parent/'output'/
        envelope['clips']['clear_26.2_27.0']['spec']['predictions']).read_text('utf-8').splitlines()]
    env_check={}
    for target in (26.517,26.817):
        frame=min(env_frames,key=lambda item:abs(item['t']-target))
        row=min(env_frozen,key=lambda item:abs(item['t']-target))
        rotors=[r for r in row['geometry'].get('rotors',[]) if r.get('current_observation',True)]
        expected=sum(np.pi*(float(r['major'])/2)*(float(r['minor'])/2) for r in rotors)
        distances=[]
        for box in frame['boxes']:
            cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
            if rotors:
                distances.append(min(float(np.hypot(cx-r['cx'],cy-r['cy'])/max(float(r['major'])/2,1e-6))
                    for r in rotors))
        env_check[str(target)]=dict(t=frame['t'],components=frame['components'],
            support_pixels=frame['support_pixels'],expected_area=expected,
            area_ratio=None if not expected else frame['support_pixels']/expected,
            widths=[b[2]-b[0] for b in frame['boxes']],heights=[b[3]-b[1] for b in frame['boxes']],
            median_distance=None if not distances else float(np.median(distances)),
            max_distance=None if not distances else float(max(distances)),
            frozen_rotors=len(rotors))
    merge_detail=merge.get('at_midpoint',{})
    clear_merge=merge_detail.get('clear_26.2_27.0',{})
    single=[cand for clip in disc['clips'].values() for frame in clip['frames']
        for cand in frame['candidates'] if len(cand['members'])==1]
    accepted=[cand for clip in disc['clips'].values() for frame in clip['frames']
        for cand in frame['candidates'] if cand['accepted']]
    accepted_radii=[cand['r'] for cand in accepted]
    merged_accepted=[cand for cand in accepted if len(cand['members'])>1]
    def band(values):
        return '—' if not values else f'{min(values):.2f}–{max(values):.2f}'
    disc_stats=dict(single=len(single),single_rms_ratio=band([c['rms_over_r'] for c in single]),
        single_radius=band([c['r'] for c in single]),
        accepted=len(accepted),merged=len(merged_accepted),
        merged_rms_ratio=band([c['rms_over_r'] for c in merged_accepted]),
        merged_coverage=band([c['coverage_deg'] for c in merged_accepted]),
        median_radius=None if not accepted_radii else float(np.median(accepted_radii)))
    disc_detail={}
    frozen_rows=[json.loads(line) for line in (ROOT.parent/'output'/disc['clips']['clear_26.4_26.95']['spec']['predictions']).read_text('utf-8').splitlines()]
    reference_row=min(frozen_rows,key=lambda item:abs(item['t']-26.517))
    frozen_rotors=[r for r in reference_row['geometry'].get('rotors',[]) if r.get('current_observation',True)]
    if frozen_rotors:
        semi=[float(r['major'])/2 for r in frozen_rotors]
        frame=min(disc['clips']['clear_26.4_26.95']['frames'],key=lambda item:abs(item['t']-26.517))
        best=max([cand for cand in frame['candidates'] if cand['accepted']] or [None],
            key=lambda cand:0 if cand is None else cand['r'])
        if best is not None:
            distance=min(float(np.hypot(best['cx']-r['cx'],best['cy']-r['cy'])/max(float(r['major'])/2,1e-6))
                for r in frozen_rotors)
            disc_detail=dict(t=frame['t'],frozen_semi_major=band(semi),fitted_radius=best['r'],
                fitted_center=[best['cx'],best['cy']],distance_semi_major=distance,
                frozen_centers=[[float(r['cx']),float(r['cy'])] for r in frozen_rotors])
    side_merge=merge_detail.get('side_110.9_111.7',{})
    images=''.join(f'<figure><img src="{key}_triplet.png"><figcaption>{label}：原始 ON/OFF / 密度筛选 / 周期确认</figcaption></figure>'
        for key,label in names)
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>桨叶支持图前端：短片段三联对照</title>
<style>body{{font:17px/1.75 'Microsoft YaHei',sans-serif;background:#f1f5fa;color:#172a40;max-width:1150px;margin:35px auto;padding:0 22px}}section{{background:white;border-radius:12px;margin:20px 0;padding:22px 28px}}h1{{font-size:30px}}h2{{font-size:23px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #dbe3ee}}img{{max-width:100%}}code{{background:#edf1f5;padding:2px 5px}}figure{{margin:14px 0}}figcaption{{font-size:14px;color:#4a5568}}.note{{background:#fff0ce;padding:12px}}.good{{background:#e8f6ee;padding:12px}}</style>
<h1>不依赖整机框的桨叶支持图：首版前端与四段短对照</h1>
<p>2026-11-01 · 依据 <code>docs/ROTOR_ONLY_HANDOFF_20261101.md</code> 第 5A 节做的第一件交付物。只跑离线短片段，没有改默认入口、权重、已发布 ZIP，也没有用标注或未来事件。</p>
<section><h2>做了什么</h2><p>新增 <code>rotor_support_front.py</code>：从原生 (x,y,t,p) 事件出发，<b>不读整机框</b>、不做灰度重建，分两级给出证据——</p><ol><li><b>密度筛选</b>：33.3ms 窗内逐像素事件数 ≥ {parameters['min_events_per_pixel']}，且 ≥ {parameters['local_multiplier']:.0f}× 局部（{parameters['local_size']}×{parameters['local_size']}）均值。绝对下限复用 <code>harmonic_pixels</code> 的 6 事件；局部对比是为了压掉整场均匀闪烁。</li><li><b>周期确认</b>：复用 <code>predictive_harmonics</code>，在窗的<b>前半段</b>发现 1–4 次谐波模板、在<b>后半段</b>用冻结模板检验极性一致性；只有通过检验的像素才叫“桨叶支持”。</li></ol><p>绿色框 = 周期确认后连通域（≥{parameters['min_component_pixels']}px，闭运算 {parameters['close_radius']} 次），描述子只给像素数、PCA 轴、长宽和中位频率。<b>不声明转子数量、不声明机械 RPM、不声明桨盘物理边缘。</b></p>
<p class="note">我在这个会话里<b>看不到图</b>（当前模型不接受图像输入），所以三联图是给你目视的；下面的数字是我能核的代理指标，不是“我看过了”。</p></section>
<section><h2>四段结果</h2><table><tr><th>片段</th><th>窗内事件</th><th>密度像素</th><th>周期像素</th><th>确认块</th><th>到最近转子中心中位（半长轴）</th><th>≤1 个半长轴</th><th>密度残留在 3a 外的比例</th></tr>{''.join(rows)}</table>
<p>“转子中心”来自同一时刻<b>冻结</b>的整机结构输出，只作诊断参照，既不是真值也不是本前端的输入。</p>
<div class="good"><p>清晰段 {clips['clear']['counts']['period_pixels']} 个周期像素里 <b>{pct(clips['clear']['frozen_reference']['overlap']['period_inside_fraction'],1)}</b> 落在冻结转子锥线内，归一化距离中位 {clips['clear']['frozen_reference']['support_distance']['median']:.2f}、P95 {clips['clear']['frozen_reference']['support_distance']['p95']:.2f}，<b>100% 在 1 个半长轴以内</b>。也就是说：不靠整机框，这个前端把支持像素几乎全部落在了真实桨盘上。</p>
<p>空场片段 {clips['negative']['counts']['period_pixels']} 个周期像素、{clips['negative']['counts']['components']} 个确认块——这个背景没有被伪造成桨叶。</p>
<p>密度过了、周期没过的那部分（清晰段 {clips['clear']['frozen_reference']['residual']['density_only_pixels']} px）里只有 {pct(clips['clear']['frozen_reference']['residual']['density_only_far_fraction'],0)} 落在所有转子 3 个半长轴之外；其余都在桨盘附近，更像没通过周期检验的桨叶/边缘像素，而不是远处背景。</p></div></section>
<section><h2>连续时刻：单窗会闪断，自带运动估计能撑住</h2><p>各取 12 个连续截止时刻（约 0.4 秒），用两种口径算相邻帧重合：<b>固定像素</b>和<b>物体坐标</b>（把上一帧掩膜按一个独立的过去速度移动后再比）。后者才公平——被跟踪的物体本来就应该在图像里移动。</p>
<table><tr><th>片段</th><th>臂</th><th>固定像素 IoU 中位</th><th>物体坐标 IoU 中位</th><th>支持像素范围</th></tr>{sequence_rows}</table>
<p>三段读数：</p><ol><li><b>单窗支持是瞬态的</b>：物体坐标 IoU 只有 0.15–0.18，也就是每 33ms 换一批像素。</li><li><b>固定像素累加会制造假稳定</b>：固定像素 IoU 0.66 看着很好，换到物体坐标立刻掉到 0.48——那是拖影，不是稳定。</li><li><b>自带运动估计后撑住了</b>：自估位移＋累加在物体坐标里到 <b>{sequence['sequences']['clear_26.4_26.8']['arms']['D']['iou_object_median']:.2f}</b>（清晰）/ <b>{sequence['sequences']['side_110.9_111.3']['arms']['D']['iou_object_median']:.2f}</b>（侧视），比给定冻结速度还高（{sequence['sequences']['clear_26.4_26.8']['arms']['C']['iou_object_median']:.2f} / {sequence['sequences']['side_110.9_111.3']['arms']['C']['iou_object_median']:.2f}）。</li></ol>
<p>为什么自估反而更好：它量的是桨叶支持区域自己的位移，而冻结速度来自整机卡尔曼状态、会滞后。自估位移与冻结速度折算位移的差中位 <b>{estimate_error['median']:.2f}px</b>（清晰，最大 {estimate_error['max']:.2f}）/ <b>{side_error['median']:.2f}px</b>（侧视，最大 {side_error['max']:.2f}）。</p>
<p class="note">方法上也留了一条负结果：整帧相位相关在真实数据上直接返回 (0,0)——静止背景的零延迟峰完全压过目标。改成“只在上一帧支持区域做匹配滤波、在 ±40px 内搜索”之后，合成位移测试 [7,−4] 精确复原。这个负结果不删。</p></section>
<section><h2>频率对得上吗（谐波歧义要留着）</h2><p>把每个确认块的中位频率和同一时刻冻结的逐桨频率比：</p><table><tr><th>片段</th><th>块像素</th><th>块频率 Hz</th><th>最近逐桨 Hz</th><th>比值</th><th>判定</th></tr>{frequency_rows}</table>
<p>清晰段三个块直接对上（0.98 / 0.99 / 1.00），另外两个（135Hz、120Hz）是 230Hz 那一桨的<b>亚谐波/半频</b>。这正是交接里点名要保留的歧义：模板梳允许 1–4 次谐波，所以一个 230Hz 的桨可能被报成 120Hz。本轮<b>不消歧</b>，也不据此改阈值——先把它标出来，下一步再用相位一致性或跨窗跟踪处理。</p></section>
<section><h2>跨帧区域跟踪：四个区域能连住 0.8 秒</h2>
<p>每个截止时刻把累加支持的连通区域做贪心关联：上一帧区域中心按<b>自估位移</b>预测，门限取 <code>max(12px, 0.75×(r旧+r新))</code> 并要求面积比在 0.4–2.5；连续 3 帧且最近只漏 1 帧才算“已确认”。id 只在本轮有效，<b>不命名桨、不声明数量、不是 RPM</b>。</p>
<table><tr><th>片段</th><th>截止数</th><th>区域总数</th><th>≥5 帧</th><th>轨迹中位年龄 s</th><th>已确认数范围</th><th>频率相对标准差中位</th></tr>{track_rows}</table>
<div class="good"><p>清晰段 26.2–27.0s：24 个截止时刻里只有 <b>{tracks['sequences']['clear_26.2_27.0']['tracks_total']}</b> 个区域，其中 <b>{tracks['sequences']['clear_26.2_27.0']['tracks_hits_ge5']}</b> 个≥5帧；<b>{len(full_hits)}</b> 条轨迹命中全部 {clear_entry['cutoffs']}/{clear_entry['cutoffs']} 帧、零漏检、年龄 {tracks['sequences']['clear_26.2_27.0']['median_age_s']:.3f}s——也就是说这些区域在整段 0.8 秒里没有换 id、没有闪断。侧视段同样 6 个区域，但只有 2 个≥5帧，中位年龄仅 {tracks['sequences']['side_110.9_111.7']['median_age_s']:.3f}s，属碎片化。</p>
<p>交叉核对（仅诊断）：在 26.5s，{diagnostic['qualified_tracks']} 个已确认区域全部落在冻结的 {diagnostic['frozen_rotors']} 个转子锥线 1 个半长轴以内（最远 {diagnostic['max_distance_in_semi_major']:.2f}a）。但 7 &gt; 4，说明<b>一个桨盘会被拆成多块</b>，所以这不是“四个桨”的证明，只能说明支持没有跑到远处背景。</p></div>
<p class="note">频率还不稳：清晰段长轨迹的相对标准差中位 {100*tracks['sequences']['clear_26.2_27.0']['frequency_relative_std_median']:.0f}%、侧视 {100*tracks['sequences']['side_110.9_111.7']['frequency_relative_std_median']:.0f}%。逐窗模板会在 120/225/300 之间跳，和前面单帧看到的亚谐波一致。这说明<b>空间跟踪已经能用，频率还没到能进推力估计的程度</b>——这是下一步要拆的。</p>
<img src="track_regions.png"><p>左：清晰段四条长轨迹的中心 y 随时间（无人机在下降，轨迹跟着走）。中／右：两段的区域数变化。</p></section>
<section><h2>把碎片并成桨盘：阈值一挪数量就变，这条路不通</h2>
<p>一个桨盘常被拆成 2–3 块，所以试了“按框间距并簇”（<code>merge_region_clusters</code>）。阈值不给死：把整条曲线都摊开——每格是“该阈值下每帧区域数的中位”，小字是范围。</p>
<table><tr><th>片段</th><th>4px</th><th>8px</th><th>12px</th><th>16px</th><th>24px</th><th>32px</th><th>48px</th></tr>{merge_rows}</table>
<div class="note"><p><b>负结果</b>：清晰段在 4–8px 下是 5 块，12–16px 下变 4 块，24px 及以上直接塌成 2 块。也就是说<b>没有与阈值无关的“桨盘数量”</b>。</p>
<p>按物理直觉选的 24px（≈冻结锥线的短轴，一个桨扫过的范围）反而把四个桨并成两团：{clear_merge.get('regions_before','—')} 块 → {clear_merge.get('groups','—')} 组，两组内部频率是 {'/'.join(str(int(v)) for c in clear_merge.get('clusters',[]) for v in c['frequencies'])} Hz 这种互相矛盾的值。并簇把不同桨的弧拼在一起，所以它不能拿来数桨。</p>
<p>顺便否掉一个容易自欺的指标：两组框各自都很大，用“笼统覆盖几个冻结转子”去算会得到 4/4，但一个 85px 宽的框同时靠近两个桨盘也被算两次——这个数字没有证明力，不写进结论。</p>
<p>结论：<b>桨盘大小必须来自弧段/椭圆的几何，而不是固定间距</b>。并簇只保留为“可能桨盘区域”的描述子，不参与 4/3/2 计数。侧视段 3→2 组同样不能当可见桨数。</p></div></section>
<section><h2>用圆弧定桨盘：单块退化，成对拟合出 r≈90px——正好是桨盘的 2 倍</h2>
<p>上一轮否掉了固定间距，这一轮改用几何：对每帧的碎片做 Kasa 圆拟合，只有当“并起来还是一个圆”时才合并（<code>rms/r ≤ 0.12</code>、角度覆盖 ≥ 120°、半径 12–140px）。覆盖度是必须的：一小段弧几乎约束不了半径，任何两个小斑都能落在某个圆上。</p>
<div class="note"><p><b>单块碎片是退化的</b>：{disc_stats['single']} 个单块拟合出的半径 {disc_stats['single_radius']}px、rms/r {disc_stats['single_rms_ratio']}——像素带厚度（1–6px）主导了拟合，单块弧给不出圆。</p>
<p><b>并起来就能拟合，而且残差极低</b>：{disc_stats['merged']} 次合并被接受，rms/r {disc_stats['merged_rms_ratio']}、覆盖 {disc_stats['merged_coverage']}°，半径中位 <b>{disc_stats['median_radius']:.0f}px</b>。</p>
<p><b>但这个半径是错的量级</b>：26.517s 冻结的四个转子半长轴只有 {disc_detail.get('frozen_semi_major','—')}px，也就是桨盘半径应在 37–44px；而拟合出来是 <b>{disc_detail.get('fitted_radius','—')}px</b>，圆心在 {disc_detail.get('fitted_center','—')}，离最近转子 {disc_detail.get('distance_semi_major',0):.2f} 个半长轴——落在四个桨盘上方，不在任何一个桨盘上。</p>
<p>所以这个圆是把<b>不同桨的碎片</b>串起来的大弧：r≈90 时，覆盖 120–140° 很容易满足，残差当然小。这是欠约束，不是检测成功。</p>
<p>结论：<b>无监督圆拟合给不出桨盘尺度</b>。要么先用碎片的线宽做“去厚度”的中心线拟合，要么加尺度先验，要么用跨时刻一致性（同一个桨盘在 0.8 秒里半径和相对位置应当稳定）来判。</p>
<p class="note">顺带修了个实现 bug：第一版合并循环在“最优对不合格”时直接 break，没去试次优对，导致几乎所有帧都 0 个候选；改成“按残差排序取第一个合格对”后才有上面这些数。这个修复本身也说明候选数对实现细节敏感，不能当结论。</p></div></section>
<section><h2>长时间扫掠包络：能连成带，填不满桨盘</h2>
<p>单块弧定不了半径，原因可以直接算：半径 40px 的圆上取 30px 弦，矢高只有 <b>2.9px</b>，而支持像素带本身厚 1–6px——曲率被厚度吃掉。所以去厚度的中心线拟合也救不了，问题不在拟合方法。</p>
<p>换个办法：桨叶几毫秒就扫一圈，于是把周期支持在<b>物体坐标</b>里做 0.8 秒的“或”累加（自估运动对齐、不衰减），理论上应该把整个桨盘扫成实心。</p>
<table><tr><th>时刻</th><th>包络块数</th><th>支持像素</th><th>冻结桨盘面积</th><th>比值</th><th>块宽</th><th>块高</th><th>到最近转子中位</th></tr>
<tr><td>{env_check['26.517']['t']:.3f}</td><td>{env_check['26.517']['components']}</td><td>{env_check['26.517']['support_pixels']}</td><td>{env_check['26.517']['expected_area']:.0f}</td><td>{env_check['26.517']['area_ratio']:.2f}</td><td>{min(env_check['26.517']['widths'])}–{max(env_check['26.517']['widths'])}</td><td>{min(env_check['26.517']['heights'])}–{max(env_check['26.517']['heights'])}</td><td>{env_check['26.517']['median_distance']:.2f}a</td></tr>
<tr><td>{env_check['26.817']['t']:.3f}</td><td>{env_check['26.817']['components']}</td><td>{env_check['26.817']['support_pixels']}</td><td>{env_check['26.817']['expected_area']:.0f}</td><td>{env_check['26.817']['area_ratio']:.2f}</td><td>{min(env_check['26.817']['widths'])}–{max(env_check['26.817']['widths'])}</td><td>{min(env_check['26.817']['heights'])}–{max(env_check['26.817']['heights'])}</td><td>{env_check['26.817']['median_distance']:.2f}a</td></tr></table>
<div class="note"><p>结果：包络确实连成了条带，而且几乎全在转子附近（中位 {env_check['26.517']['median_distance']:.2f} 个半长轴，最远 {env_check['26.517']['max_distance']:.2f}a），但<b>只填了桨盘面积的 {100*env_check['26.517']['area_ratio']:.0f}%</b>；宽 40–76px、高只有 14–24px——是沿桨盘长轴的窄带，不是实心圆盘。</p>
<p>原因不在运动补偿，而在<b>周期检验本身是选择性的</b>：只有局部极性图案在前半窗/后半窗都对上模板的像素才通过，桨叶扫过的大部分像素拿不到这个证据。0.8 秒的“或”累加也只是把通过过的像素攒起来，攒不满一个盘。</p>
<p>结论：<b>桨盘尺度这条路，在“只用周期确认像素”的前提下走不通</b>。要么放宽证据（例如纳入未过周期但密度合格的像素，代价是背景残留），要么由外部尺度先验给桨盘大小——而这必须先声明是假设。</p></div></section>
<section><h2>频率跨窗选择：跳变里有一半是“窗口换了峰”</h2>
<p>逐窗模板搜 1–4 次谐波，所以同一块区域会在 270Hz 与 135Hz 之间来回跳（正好差 2 倍）。每个合格像素其实带<b>两个</b>候选频率，之前只用了被选中的那个。这一轮把两个都留下来，做因果选择：只看<b>过去</b>已确认值的中位，从“本窗报告值 ∪ 本区域候选”里挑最接近的那个（门限 1.6 倍）。</p>
<table><tr><th>片段</th><th>报告值散布</th><th>只做中位平滑（对照）</th><th>候选＋中位</th><th>每截止换值次数</th></tr>
<tr><td>清晰 26.2–27.0s</td><td>{frequency['clips']['clear_26.2_27.0']['reported']['median']:.3f}</td><td>{frequency['clips']['clear_26.2_27.0']['smoothed_reported']['median']:.3f}</td><td><b>{frequency['clips']['clear_26.2_27.0']['assigned']['median']:.3f}</b></td><td>{frequency['clips']['clear_26.2_27.0']['switches']}</td></tr>
<tr><td>侧视 110.9–111.7s</td><td>{frequency['clips']['side_110.9_111.7']['reported']['median']:.3f}</td><td>{frequency['clips']['side_110.9_111.7']['smoothed_reported']['median']:.3f}</td><td><b>{frequency['clips']['side_110.9_111.7']['assigned']['median']:.3f}</b></td><td>{frequency['clips']['side_110.9_111.7']['switches']}</td></tr></table>
<div class="good"><p><b>对照是关键</b>：只做中位平滑只能把 0.297 压到 0.178（清晰），而允许从候选里挑能压到 <b>0.079</b>；侧视 0.358→0.239→<b>0.021</b>。也就是说，将近一半的跳变不是“真转速变了”，而是<b>窗口在谐波里换了峰</b>，而正确的那个值当时就在候选里。</p>
<p>典型序列（清晰 track 1）：报告 270 270 270 270 <b>135</b> <b>135</b> <b>135</b> <b>135</b>，只平滑会拖成 270…270→135，而候选选择保持在 270——因为 135 是 270 的一半，270 仍在候选里。</p></div>
<p class="note">必须说清的边界：① 被选出来的序列是<b>平滑值</b>，不是新测量，散布变小有一部分是构造使然；② 该规则在阶跃测试里被证明会锁死（见“阶跃响应”一节），散布变小有锁死成分；替代规则已给出但未在真实片段重跑；③ 没有独立 RPM 真值，所以只能说“序列更稳、且差异可归因于换峰”，不能说更准。因此这只是候选，不进默认。</p></section>
<section><h2>把前端当门控：反而更差</h2>
<p>桨盘尺度拿不到，前端至少能回答“这一刻有没有桨叶证据”。那就用它当<b>门控</b>：门关掉时把既有的推力输入丢掉（只减不加）。门控 10Hz、因果、只用过去事件；基线回放逐位复原 34.6049，所以差别只来自门控。</p>
<table><tr><th>片段</th><th>门开样本</th><th>被关掉的输入帧</th><th>默认 0.4s 均值</th><th>门控后</th><th>差</th></tr>
<tr><td>自采侧视 110–118s</td><td>{gate['clips']['side_110_118']['gate_true']}/{gate['clips']['side_110_118']['samples']}</td><td>{gate['clips']['side_110_118']['ticks_closed']}/122</td><td>{gate['clips']['side_110_118']['base']['0.4']['mean']:.4f}</td><td>{gate['clips']['side_110_118']['gated']['0.4']['mean']:.4f}</td><td><b>+{gate['clips']['side_110_118']['gated']['0.4']['mean']-gate['clips']['side_110_118']['base']['0.4']['mean']:.2f}</b></td></tr>
<tr><td>另一段 40–46s</td><td>{gate['clips']['train2_40_46']['gate_true']}/{gate['clips']['train2_40_46']['samples']}</td><td>{gate['clips']['train2_40_46']['ticks_closed']}/7</td><td>{gate['clips']['train2_40_46']['base']['0.4']['mean']:.4f}</td><td>{gate['clips']['train2_40_46']['gated']['0.4']['mean']:.4f}</td><td>{gate['clips']['train2_40_46']['gated']['0.4']['mean']-gate['clips']['train2_40_46']['base']['0.4']['mean']:+.2f}</td></tr></table>
<div class="note"><p>侧视段门控只开 {gate['clips']['side_110_118']['gate_true']}/{gate['clips']['side_110_118']['samples']}（56% 关闭），与被关掉的 18 帧输入重叠，结果 <b>34.6049 → 35.7918，差 1.19px</b>。另一段 7 帧输入恰好都落在门开的 3 个样本上，所以没变化（59.7896 不变）。</p>
<p>这说明：<b>“没有周期确认的桨叶”不等于“这帧的推力输入是错的”</b>。侧视段本来就只有池化代理可用，门控把有用的输入一起砍掉了。</p>
<p>结论：<b>前端不作为门控使用</b>。它现在的信息还不如既有池化口径的“可用性判断”。要让桨叶检测真正参与预测，只能走“替代输入”那条路——而那需要尺度与 4/3/2 可见性，两者都还卡着。</p></div></section>
<section><h2>想提高侧视覆盖：卡在证据，不在阈值</h2>
<p>门控在侧视关了 56%，所以先试四个<b>声明过理由</b>的改法（不看轨迹误差，只看覆盖与空场）：</p>
<table><tr><th>变体</th><th>侧视周期像素</th><th>清晰周期像素</th><th>空场周期像素</th></tr>
<tr><td>A 基线（33.3ms，半窗≥6）</td><td>{coverage['variants']['A']['clips']['side']['counts']['period_pixels']}</td><td>{coverage['variants']['A']['clips']['clear']['counts']['period_pixels']}</td><td>{coverage['variants']['A']['clips']['negative']['counts']['period_pixels']}</td></tr>
<tr><td>B 窗长加倍 66.7ms</td><td>{coverage['variants']['B']['clips']['side']['counts']['period_pixels']}</td><td>{coverage['variants']['B']['clips']['clear']['counts']['period_pixels']}</td><td>{coverage['variants']['B']['clips']['negative']['counts']['period_pixels']}</td></tr>
<tr><td>C 窗长加倍＋半窗门槛 4</td><td>{coverage['variants']['C']['clips']['side']['counts']['period_pixels']}</td><td>{coverage['variants']['C']['clips']['clear']['counts']['period_pixels']}</td><td>{coverage['variants']['C']['clips']['negative']['counts']['period_pixels']}</td></tr>
<tr><td>D 密度放宽（4 事件、1.5×）</td><td>{coverage['variants']['D']['clips']['side']['counts']['period_pixels']}</td><td>{coverage['variants']['D']['clips']['clear']['counts']['period_pixels']}</td><td>{coverage['variants']['D']['clips']['negative']['counts']['period_pixels']}</td></tr></table>
<div class="note"><p>B、C 反而更差（窗拉长后 66ms 内无人机移动约 20px，模板对不上）；C 与 B 完全相同，说明<b>半窗事件门槛根本没起作用</b>（合格像素本来就 ≥6）。D 只给侧视 +9%，清晰却涨 3.4 倍——侧视的瓶颈不在密度门。</p>
<p>真正的差别在<b>检验统计量</b>：侧视合格像素 901、清晰 881，数量相当；但侧视只有 20% 通过（清晰 42%），验证统计量中位 <b>{validation['clips']['side']['bounds']['median']:.3f}</b>（清晰 {validation['clips']['clear']['bounds']['median']:.3f}）。把门槛从 0.01 放到 0.05 能让侧视 177→{validation['clips']['side']['accepted_by_threshold']['0.05']['pixels']}、清晰 374→{validation['clips']['clear']['accepted_by_threshold']['0.05']['pixels']}——但这<b>不能采用</b>：空场片段的“守卫”是空的（合格像素为 0），任何放宽它都照样通过，等于没有对照。</p>
<p><b>机制检查（这一轮最有用的新证据）</b>：把合格像素的极性随机打乱后再跑同一检验，通过数从 176/374 掉到 <b>{validation['clips']['side']['polarity_shuffled_null']['accepted']}/{validation['clips']['clear']['polarity_shuffled_null']['accepted']}</b>。说明这个检验真的在验证“时间上的极性结构”，不是只看事件密度。</p>
<p>结论：<b>侧视覆盖是证据缺口，不是阈值缺口</b>。要放宽门槛，必须先有一个“事件够密但没有桨”的背景对照；在拿到之前，保持 0.01 不动。</p></div></section>
<section><h2>想放宽验证门槛：找不到像样的反面守卫</h2>
<p>上一轮说“要放宽门槛，先要有事件够密但没有桨的背景对照”。这一轮去找，结果很不理想：</p>
<table><tr><th>候选片段</th><th>每 33ms 事件</th><th>密度像素</th><th>合格像素</th><th>0.05 门槛下通过</th></tr>
<tr><td>自采侧视 111.0s（有桨）</td><td>86,870</td><td>1,531</td><td>901</td><td>{validation['clips']['side']['accepted_by_threshold']['0.05']['pixels']}</td></tr>
<tr><td>train5 @0.5s</td><td>13,376</td><td>{summary['clips']['negative']['counts']['density_pixels']}</td><td>{validation['clips']['negative']['eligible']}</td><td>{validation['clips']['negative']['accepted_by_threshold']['0.05']['pixels']}</td></tr>
<tr><td>train6 @0.5s（起飞前）</td><td>25,473</td><td>89</td><td>0</td><td>0</td></tr>
<tr><td>train3 @1.0s（最密）</td><td>80,162</td><td>378</td><td>{validation['clips']['dense_background']['eligible']}</td><td>{validation['clips']['dense_background']['accepted_by_threshold']['0.05']['pixels']}</td></tr></table>
<div class="note"><p>最密的 train3@1.0s 有 8 万事件（和侧视相当），但只有 <b>{validation['clips']['dense_background']['eligible']}</b> 个合格像素——它的 8 万事件是铺满画面的弥散噪声，不是集中在少数像素上的“桨叶式”高计数。侧视是 1,531 个密度像素里 901 个合格（59%），train3 是 378 里 {validation['clips']['dense_background']['eligible']} 个（0.5%）。</p>
<p>也就是说：<b>“事件多”和“每像素事件多”是两件事</b>，而这个前端卡的是后者。现有数据里找不到既密又集中在局部、同时没有周期结构的东西，所以 0.05 门槛在真片上既证明不了安全、也证明不了不安全——<b>守卫仍然不合格</b>，门槛维持 0.01。</p>
<p>要往下走，需要的是一条真实录像：画面里有<b>周期性但不是桨</b>的东西（转动的风扇、闪烁灯、摆动的线），事件密度与桨叶同量级。没有这条数据，任何放宽都是赌。</p></div></section>
<section><h2>阶跃响应：上一轮那个规则其实在“锁死”</h2><p>上一轮用“挑最靠近中位的候选”把频率散布从 0.297 压到 0.079。这一轮做阶跃测试才发现那个规则有不动点：只要旧值还在候选里，它就会一直把旧值再选一遍。30 帧、阶跃发生在第 10 帧：</p><table><tr><th>用例</th><th>规则A 跟随帧</th><th>规则A 换值次数</th><th>规则B 跟随帧</th><th>规则B 换值次数</th></tr><tr><td>阶跃 ×1.11（270→300）</td><td>永不</td><td>20</td><td>12</td><td>1</td></tr><tr><td>阶跃 ×1.48（270→400）</td><td>永不</td><td>20</td><td>12</td><td>1</td></tr><tr><td>阶跃 ×0.50（270→135）</td><td>永不</td><td>20</td><td>12</td><td>1</td></tr><tr><td>斜坡 270→330</td><td>永不</td><td>24</td><td>永不</td><td>1</td></tr><tr><td>纯噪声无真实变化</td><td>永不</td><td>29</td><td>永不</td><td>0</td></tr><tr><td>阶跃且旧值消失</td><td>10</td><td>0</td><td>12</td><td>1</td></tr></table><div class="note"><p><b>规则A（上一轮）</b>：三个阶跃全部<b>永不跟随</b>，却在 30 帧里换了 20 次值；纯噪声下换了 29 次。它只在旧值从候选里消失时（最后一行）才跟上。所以上一轮的“散布变小”有相当一部分是<b>锁死</b>，不是跟踪——这个结论必须修正。</p><p><b>规则B（共识切换）</b>：偏离超过 10% 且连续 3 帧同向才改值。阶跃在第 12 帧跟上（阶跃发生在第 10 帧，滞后 2 帧 ≈ 67ms）；纯噪声下 0 次误换；斜坡只能以 10% 为台阶跟，所以“2% 内跟随”永远达不到——这是滞后，不是失败，但必须写明。</p><p>代价与边界：10% 死区意味着小于 10% 的真实转速变化看不见；阶跃滞后约 2 帧。要用于推力变化率，这个滞后必须计入。规则B 已作为 <code>ConsensusFrequencySelector</code> 放进模块，但<b>尚未在真实片段上重跑</b>——上一轮 0.297→0.079 的数字不能直接归给规则B。</p></div></section><section><h2>总账：这一批做成了什么、证伪了什么、卡在哪</h2><p><b>做成了</b>：① 接管前备份（带逐文件 SHA256 清单，已写进 WORKLOG）；② 不依赖整机框的桨叶支持图前端 <code>rotor_support_front.py</code>，清晰/部分遮挡/侧视/空场四段三联图；③ 连续时刻连续性：单窗物体坐标 IoU 0.18，自估运动对齐后 0.83（清晰）/0.65（侧视）；④ 跨帧区域跟踪：清晰段 4 条轨迹命中 24/24 帧、零漏检、中位年龄 0.767s；⑤ 频率候选保留：证明近一半跳变是窗口换峰、真值一直在候选里。</p><p><b>证伪</b>：① 固定间距并簇求桨盘数量——阈值一挪数量就变（5→4→2）；② 圆拟合定桨盘——单块退化（r 4–19px），成对拟合出 r≈90px 正好是桨盘半长轴的 2 倍，圆心离最近转子 1.75a；③ 把前端当输入门控——侧视 34.6049→35.7918（+1.19px），另一段无变化；④ 上一轮“频率散布 0.297→0.079”的解读降级——阶跃测试显示那个规则是锁死（三个阶跃永不跟随、纯噪声下换值 29 次）。</p><p><b>卡住</b>：① 桨盘尺度不可辨识——矢高 2.9px 对带厚 1–6px，0.8 秒扫掠也只填到桨盘面积 47%；② 验证门槛无法放宽——现有“密但没有桨”的候选合格像素只有 0–2 个，守卫是空集；③ 4/3/2 可见性分支与“替代输入”都依赖①。</p><p><b>需要拍板</b>：① 桨盘尺度用外部先验（必须写成假设），还是放宽证据（把密度合格但未过周期的像素纳入，承担背景残留）？② 要不要补拍一条“周期性但不是桨”的短录像（风扇/闪烁灯/摆动线，密度与桨叶同量级）——这是放宽验证门槛的唯一前提。</p><p class="note">本批<b>没有</b>改默认入口、权重、已发布 ZIP；没碰 <code>D:\yolov10-main</code>，没改 RAW，没用标注或未来事件引导任何判断。所有读数都可从本目录的 JSON 复算；三联图需要人眼核验，我这一侧看不到图。</p><p>文件索引：<a href="index.html">本报告</a> · <a href="summary.json">四段单帧统计</a> · <a href="sequence.json">连续时刻三臂</a> · <a href="tracks.json">跨帧轨迹</a> · <a href="merge.json">并簇曲线</a> · <a href="disc_geometry.json">圆拟合</a> · <a href="sweep_envelope.json">扫掠包络</a> · <a href="frequency_tracks.json">频率跨窗</a> · <a href="frequency_step.json">频率阶跃</a> · <a href="rotor_gate_forecast.json">门控对照</a> · <a href="coverage_variants.json">覆盖变体</a> · <a href="period_validation.json">检验分布</a> · <a href="clear_triplet.png">清晰三联</a> · <a href="partial_triplet.png">部分遮挡三联</a> · <a href="side_triplet.png">侧视三联</a> · <a href="negative_triplet.png">空场三联</a></p></section><section><h2>还没做的</h2><ul><li><b>侧视覆盖是证据缺口</b>：四个改法都没实质提升（最长只 +9%），验证统计量本身弱（0.118 vs 清晰的 0.019）</li><li><b>放宽门槛仍缺合格守卫</b>：现有候选（train5/train6/train3）在 0.05 门槛下都是 0，且合格像素仅 0–2，既不能证明安全也不能证明不安全</li><li><b>门控已试并否掉</b>：侧视段 +1.19px，另一段无变化；前端不能当“输入可用性”判断器</li><li><b>桨盘尺度仍然拿不到</b>：曲率被像素带厚度吃掉（矢高 2.9px vs 带厚 1–6px），周期支持 0.8 秒也只填到桨盘面积的 47%</li><li><b>频率选择器有锁死缺陷</b>：阶跃测试里规则A 三个阶跃全部永不跟随、纯噪声下换值 29 次；上一轮“散布变小”有相当一部分是锁死。替代规则B 已给出（阶跃滞后约 2 帧、10% 死区），但未在真实片段重跑</li><li>4/3/2 可见性分支<b>被尺度问题挡住</b>：没有桨盘大小就不能数桨，现在只能输出“桨叶候选区域”</li><li>周期支持还<b>没有</b>接到原生弧段；替代输入（而不是门控）尚未评估</li><li>没有做任何视觉核验结论；三联图请你看</li></ul></section>
<section><h2>下一步顺序</h2><p>① <b>补一条“周期性但不是桨”的真实短片</b>（转动的风扇、闪烁灯、摆动的线，事件密度与桨叶同量级）——这是放宽验证门槛的唯一前提，现有数据里没有；② <b>尺度二选一并写清是假设</b>（外部先验 or 放宽证据），否则 4/3/2 与替代输入都无从谈起；③ 尺度有结论后再做“替代输入”的同起点对照；④ 补测频率选择器的阶跃响应，否则不能用于推力变化率。</p><p><a href="summary.json">逐段 JSON</a> · <a href="sequence.json">连续时刻三臂读数</a> · <a href="tracks.json">跨帧区域轨迹</a> · <a href="merge.json">并簇曲线与负结果</a> · <a href="disc_geometry.json">圆拟合逐帧候选</a> · <a href="sweep_envelope.json">扫掠包络</a> · <a href="frequency_tracks.json">频率跨窗选择</a> · <a href="rotor_gate_forecast.json">门控对照</a> · <a href="coverage_variants.json">覆盖变体</a> · <a href="period_validation.json">周期检验分布与打乱对照</a> · <a href="frequency_step.json">频率阶跃响应</a> · <a href="../../event_drone_forecast/rotor_support_front.py">前端模块</a> · <a href="../../event_drone_forecast/tools/probe_frequency_tracks.py">频率探针</a> · <a href="../../event_drone_forecast/docs/ROTOR_ONLY_HANDOFF_20261101.md">交接依据</a></p></section>
{images}</html>'''
    (OUT/'index.html').write_text(report,'utf-8')
    manifest=dict(report=str(OUT/'index.html'),summary=str(OUT/'summary.json'),
        figures=[f'{key}_triplet.png' for key,_ in names],
        modules=['rotor_support_front.py','tools/probe_rotor_support.py'],
        no_default_change=True,no_labels_used=True,no_future_events=True,body_box_used=False,
        rotor_count_claimed=False,mechanical_RPM_measured=False)
    (OUT/'report_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),'utf-8')
    print(str(OUT/'index.html'))


if __name__=='__main__':main()
