"""Read-only scoring diagnostics and visual report for stage18."""
import html,json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study18 import ROOT,OUT,BASE,CASES,SNAPS,load,save,verify,labels

NAMES={'clear6':'train6清晰22–27s','side6':'train6侧视108–120s','review4':'train4跨录像74–84s','negative5':'train5无机0–8s'}
METHODS=['incremental_strict','incremental_track']
COLORS=['#158e7b','#e27d21']


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,body):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in body)+'</tbody></table></div>'


def images(allrows):
    links=[]
    for case,rr in allrows.items():
        for t in SNAPS[case]:
            row=min(rr,key=lambda r:abs(r['t']-t));assert abs(row['t']-t)<1e-8
            count=np.load(OUT/case/f'snapshot_{t:.2f}.npz')['total'];h,w=count.shape
            image=cv2.cvtColor(np.uint8(255-230*np.clip(np.log1p(count)/np.log(8),0,1)),cv2.COLOR_GRAY2RGB)
            q=row['methods']['incremental_strict'];points=[[r['cx'],r['cy']] for r in q.get('raw_candidates',[])]
            p=q.get('proposal')
            if p and p['detected']:points.extend(np.array(p['bbox']).reshape(2,2).tolist())
            if points:
                pp=np.array(points);lo=np.maximum(0,np.floor(pp.min(0)-[75,60])).astype(int);hi=np.minimum([w,h],np.ceil(pp.max(0)+[75,60])).astype(int)
                x0,y0=lo;x1,y1=hi
            else:x0,y0,x1,y1=0,0,w,h
            fig,axes=plt.subplots(1,4,figsize=(18,4.5),layout='constrained')
            axes[0].imshow(image);axes[0].set_title(f'{case} {t:.2f}s fullFOV')
            axes[0].add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='#236ea1'))
            axes[1].imshow(image);axes[1].set_title('20ms ON+OFF count (zoom)')
            for ax,m,col in zip(axes[2:],METHODS,COLORS):
                ax.imshow(image);o=row['methods'][m]['observation']
                for r in q.get('candidates',[]):
                    ellipse=matplotlib.patches.Ellipse((r['cx'],r['cy']),r['major'],r['minor'],angle=r['angle'],fill=False,edgecolor='#49a5a0',linewidth=1)
                    ax.add_patch(ellipse)
                if o['detected']:
                    b=o['bbox'];ax.add_patch(plt.Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,color=col,lw=2))
                    ax.scatter(*o['center'],s=65,marker='x',color=col)
                ax.set_title(f"{'strict3' if m==METHODS[0] else 'track latch'}: published={o['detected']}")
            for ax in axes[1:]:ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
            for ax in axes:ax.set_xlabel('native x [px]');ax.set_ylabel('native y [px]')
            name=f'{case}_{t:.2f}.png';fig.savefig(OUT/name,dpi=125);plt.close(fig);links.append((case,t,name))
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
        for m,col in zip(METHODS,COLORS):
            for dim,ax in enumerate(axes[:2]):
                ax.plot([r['t'] for r in rr],[r['methods'][m]['observation']['center'][dim] if r['methods'][m]['observation']['detected'] else np.nan for r in rr],color=col,label=m,lw=1.3,alpha=.8)
        for l in labels(case):
            if l['state']=='present':
                box=np.array(l['bbox']).reshape(2,2).mean(0)
                for i,ax in enumerate(axes[:2]):ax.scatter(l['t'],box[i],color='black',marker='x',s=35)
        axes[0].set_title(f'{case}: current centers; black x = sparse box center')
        axes[0].set_ylabel('x [px]');axes[1].set_ylabel('y [px]')
        tt=[r['t'] for r in rr];timings=[r['methods'][METHODS[0]]['timing_ms'] for r in rr]
        for name in ['histogram_ingest','spectrum','total']:
            axes[2].plot(tt,[d.get(name,np.nan) for d in timings],label=name,lw=.8)
        axes[2].axhline(20,color='red',ls='--',label='20ms output period');axes[2].set_ylabel('processing [ms]');axes[2].set_xlabel('RAW time [s]')
        for ax in axes:ax.legend(loc='best',fontsize=8);ax.grid(alpha=.2)
        fig.savefig(OUT/f'{case}_timeline.png',dpi=135);plt.close(fig)
    return links


def diagnostics(allrows):
    added=[];counts=[];missing=[]
    for case,rr in allrows.items():
        tt=np.array([r['t'] for r in rr]);both=extra=lost=0
        last=None;velocity=np.zeros(2);before=[]
        for r in rr:
            a,b=[r['methods'][m]['observation'] for m in METHODS]
            both+=a['detected'] and b['detected'];extra+=not a['detected'] and b['detected'];lost+=a['detected'] and not b['detected']
            p=r['methods'][METHODS[0]].get('proposal');d={}
            if last is not None and p and p['detected']:
                gap=r['t']-last['t'];box=np.array(p['bbox']);extent=float(max(box[2:]-box[:2]))
                d=dict(predicted_distance_px=float(np.linalg.norm(np.array(p['center'])-(last['center']+velocity*gap))),
                    distance_gate_px=max(50.,max(last['extent'],extent)*.4),
                    extent_ratio=max(last['extent'],extent)/max(1.,min(last['extent'],extent)))
            before.append(d)
            if b['detected']:
                box=np.array(b['bbox']);center=np.array(b['center']);gap=float('inf') if last is None else r['t']-last['t']
                if last is not None and 0<gap<=.16:velocity=.7*velocity+.3*np.clip((center-last['center'])/gap,-2500,2500)
                else:velocity[:]=0
                last=dict(t=r['t'],center=center,extent=float(max(box[2:]-box[:2])))
        counts.append(dict(case=case,both_published=int(both),track_only=int(extra),strict_only=int(lost)))
        for l in labels(case):
            if l['state']!='present':continue
            i=np.searchsorted(tt,l['t'],side='right')-1
            if i<0 or l['t']-tt[i]>.025:continue
            a,b=[rr[i]['methods'][m]['observation'] for m in METHODS]
            if not b['detected']:
                prior=[r['t'] for r in rr[:i] if r['methods']['incremental_track']['observation']['detected']]
                proposal=rr[i]['methods']['incremental_strict'].get('proposal')
                missing.append(dict(case=case,t=l['t'],observation_t=float(tt[i]),proposal_detected=bool(proposal and proposal['detected']),
                    track_status=rr[i]['methods']['incremental_track']['status'],last_track_measurement_age_s=float(tt[i]-prior[-1]) if prior else None,**before[i]))
            if a['detected']==b['detected']:continue
            o=b if b['detected'] else a;box=np.array(l['bbox']);diag=np.linalg.norm(box[2:]-box[:2]);center=(box[:2]+box[2:])/2
            added.append(dict(case=case,t=l['t'],track_added=bool(b['detected']),error_px=float(np.linalg.norm(np.array(o['center'])-center)),
                error_over_box_diag=float(np.linalg.norm(np.array(o['center'])-center)/max(diag,1.)),status=rr[i]['methods']['incremental_track']['status']))
    result=dict(publications=counts,changed_present_labels=added,missing_present_labels=missing,note='Post-score diagnostics only; no parameter changes based on these outputs.')
    save(OUT/'publication_diagnostics.json',result);return result


def main():
    verify();a=load(OUT/'analysis.json');allrows={case:list(map(json.loads,(OUT/case/'observations.jsonl').read_text().splitlines())) for case in CASES}
    diag=diagnostics(allrows);links=images(allrows);completions={case:load(OUT/case/'completion.json') for case in CASES}
    body=[]
    for r in a['table']:
        body.append([NAMES[r['case']],r['method'],f"{r['detected']}/{r['frames']}",f"{r['present_detected']}/{r['present_labels']}",fmt(r['conditional_error_px']),f"{r['joint_success']}/{r['present_labels']}",r['wrong_position'],f"{r['absent_detections']}/{r['absent_labels']}"])
    bench=[dict(case=case,**b) for case in CASES for b in load(OUT/case/'paired_benchmark.json')]
    speed=[]
    for case,c in completions.items():
        rr=allrows[case];valid=[r for r in rr if r['methods'][METHODS[1]]['observation']['detected']]
        pp=[r['methods'][METHODS[0]]['timing_ms']['total']+r['publication_ms'] for r in valid]
        speed.append([NAMES[case],fmt(c['frontend_plus_track_median_ms']),fmt(c['frontend_plus_track_p95_ms']),fmt(float(np.median(pp)) if pp else None),f"{sum(x<=20 for x in pp)}/{len(pp)}",fmt(c['research_wall_including_open_seek_decode_json_bench_s'])])
    snippets=[]
    for case in CASES:
        snippets.append(f'<details><summary>{NAMES[case]}：逐帧画面与当前位置</summary><img src="{case}_timeline.png" loading="lazy">')
        snippets.extend(f'<figure><a href="{name}"><img src="{name}" loading="lazy"></a><figcaption>{t:.2f}s。左：全图与原始计数；右：同一份桨盘提案、两种发布策略。仅当前位置，无未来轨迹。</figcaption></figure>' for c,t,name in links if c==case)
        snippets.append('</details>')
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第十八阶段：增量事件统计与轨迹确认</title><style>
body{font:17px/1.75 "Microsoft YaHei",sans-serif;background:#f3f6f9;color:#243342;margin:0}main{max-width:1250px;margin:auto;background:white;padding:30px 36px}h1{font-size:31px;line-height:1.3}h2{margin-top:40px;padding-bottom:6px;border-bottom:2px solid #dce5ec}a{color:#176b99}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;white-space:nowrap;padding:9px;border-bottom:1px solid #d8e1e8}th{background:#eaf0f6}.scroll{overflow:auto}.note{padding:18px 22px;background:#edf6fa;border-left:5px solid #3387a6}.warn{background:#fff4df;border-color:#c89028}img{max-width:100%;height:auto;border:1px solid #dde4ea}details{padding:15px;border:1px solid #dce5ec;margin:18px 0}summary{cursor:pointer;font-weight:600}figure{margin:22px 0}figcaption{font-size:14px;color:#607383}code{background:#eef2f6;padding:2px 5px}li{margin:6px 0}.small{font-size:14px;color:#5c6f80}</style><main>
<h1>第十八阶段：增量事件统计与有期限的轨迹确认</h1><p>2026-10-18 · 四段35秒自采RAW · 1750个截止时刻 · 保留旧默认与原始报告</p>
<div class="note"><b>两个问题分别验证：</b>增量统计是否改变任何检测数值；轨迹确认是否改善覆盖并控制误报。先保证全局几何行为等价，再单独替换外层发布规则。没有把ROI偏移与提速混在一起。</div>
<h2>1. 检测覆盖与定位</h2><p>strict为阶段17全局规则加增量统计；track与它共享同一份当前整机提案，只改变发布。采用同一稀疏标签，取标签前≤25ms的最新观测；联合通过=有发布且误差≤框对角线25%，明显错位=&gt;50%。条件误差只统计有发布的点，不能脱离覆盖率解释。</p>'''
    text+=table(['片段','方法','发布/时刻','有机覆盖','条件误差px','联合通过','明显错位','无机标签发布'],body)
    text+='<div class="note warn"><b>本轮取舍：</b>保留增量统计作为可复用的等价计算实现；不默认启用新track发布。侧视标签覆盖仍为9/14，train4发布数还从128降为121；当前完整清晰前端中位约67ms，尚不能满足20ms实时周期。旧默认预测入口继续保留。</div>'
    old=load(BASE/'archived_baseline_comparison.json')['table']
    text+='<p>原阶段9邻域前端按相同标签重评的参照：</p>'+table(['片段','旧有机覆盖','旧条件误差px'],[[NAMES[r['case']],f"{r['present_detected']}/{r['present_labels']}",fmt(r['conditional_error_px'])] for r in old])
    text+='<p>旧阶段9是10ms步长且预热不同，仅作原方案参照；本轮strict/track之间才是同输入、同提案的发布策略消融。没有修改几何中心或补出不可见的真实位置。</p>'
    text+='<h2>2. 发布变化是否只是多发了一些框</h2>'
    text+=table(['片段','共同发布','仅track发布','仅strict发布'],[[NAMES[r['case']],r['both_published'],r['track_only'],r['strict_only']] for r in diag['publications']])
    text+='<p>下表逐项列出有机标签中两方法发布状态不同的时刻；同时保留像素误差和相对框对角线误差，防止大框的宽松门槛掩盖偏移。</p>'
    text+=table(['片段','标签时刻s','track新增','当前中心误差px','误差/框对角线','状态'],[[NAMES[r['case']],f"{r['t']:.3f}",r['track_added'],fmt(r['error_px']),fmt(r['error_over_box_diag']),r['status']] for r in diag['changed_present_labels']])
    text+='<p>仍然漏发的有机标签：</p>'+table(['片段','标签时刻s','当前整机提案','发布状态','距上次实测s','关联距离/门限px','跨度倍率'],[[NAMES[r['case']],f"{r['t']:.3f}",r['proposal_detected'],r['track_status'],fmt(r['last_track_measurement_age_s']),f"{fmt(r.get('predicted_distance_px'))}/{fmt(r.get('distance_gate_px'))}",fmt(r.get('extent_ratio'))] for r in diag['missing_present_labels']])
    text+='<p>116.60s与117.28s并非保留时间太短：距上次实测均60ms、位置关联距离仍在门限内，但跨度倍率约2.14/3.59超过2倍。另一类漏检没有当前整机提案。下一步应处理局部桨盘组合导致的整机尺度变化，不能单靠延长寿命或继续放宽关联门槛。</p>'
    text+='<h2>3. 增量统计有没有偷偷改变结果</h2><p>过去100ms切成400个250µs时间格。每次只加入最新20ms、移除过期格，按时间顺序读取同一100ms直方图，再执行原FFT及原几何。频带、阈值、ON/OFF输入、分组、尺度保护、三帧严格规则均保持。</p>'
    text+=table(['片段','严格输出与阶段17全局逐项一致','真实RAW前缀复核'],[[NAMES[case],f"{c['optimized_exact_vs_stage17']}/{c['frames']}",c['causal_prefix_exact']] for case,c in completions.items()])
    text+='<p>这是完整观测字典一致，不只是数量相同。当前实验未使用ROI；几何仍有原生全图分配，尚未做裁剪坐标变换。已做合成随机流30时刻频谱数组逐项一致、完整几何16时刻一致、窗口过期、未来事件隔离与无当前量测不发布检查。见<a href="module_checks.json">module_checks.json</a>。</p>'
    text+='<h2>4. 实际省了多少计算</h2><p>在固定23/110/75s，使用同一事件输入及相同历史状态，旧整窗重算与新“20ms入环+频谱”交替次序各运行7次；新入环成本计入计时，状态副本构造在两者计时外。下表是频谱模块，不能当端到端倍速。</p>'
    text+=table(['片段/时刻','当前事件数','100ms事件数','旧中位ms','新中位ms','模块倍速','频谱逐项一致'],[[f"{NAMES[b['case']]} / {b['t']:.2f}",b['current_events'],b['history_events'],fmt(b['old_median_ms']),fmt(b['incremental_median_ms']),fmt(b['old_median_ms']/b['incremental_median_ms']),b['results_exact']] for b in bench])
    text+='<p>新前端完整调用计时包含入环、频谱、原几何与track发布；不含解码/写JSON。研究wall含读取/seek、两种发布记录、截图和配对基准；仍须与独立单入口耗时区分。目标时刻“只看发布帧”的耗时也列出，以免空档拉低中位数。</p>'
    text+=table(['片段','全片中位ms','全片P95 ms','发布帧中位ms','发布帧≤20ms','研究wall s'],speed)
    text+='<h2>5. 轨迹确认的边界与复用来源</h2><p>新轨迹仍需连续3次当前提案；确认后最多保留160ms关联资格，用过去速度、位置距离与尺度门限检查新提案。没有当前几何提案时不发布；位置/尺度不符时重新确认，至少离开原测量80ms后才转移假设。该规则只避免把短时漏检当作全新目标，不会从空白画面恢复真实姿态。</p>'
    text+='<p><a href="https://github.com/tryolabs/norfair/blob/master/norfair/tracker.py">Norfair作者实现</a>已有初始化延迟与命中惯性，<a href="https://docs.prophesee.ai/stable/api/python/core/bindings.html">Metavision官方SDK</a>已有滚动事件缓冲。本轮实际执行原SDK RawReader与本项目既有几何/恢复模块；额外实现精确250µs分块直方图缓存和外层发布适配，没有把它们包装为新跟踪算法，也未执行Norfair或原effqpdet C++。</p>'
    text+='<div class="note warn">这些数据已用于开发，train4也曾看过抽样，不能声称盲测。稀疏框不足以验证每个新增发布；没有独立逐桨、机械RPM或真实姿态真值。本轮仅当前检测/定位，没有验证0.1/0.2/0.4秒未来轨迹改善。</div>'
    text+='<h2>6. 查看具体画面</h2>'+''.join(snippets)
    text+='<h2>7. 文件与后续取舍</h2><p>最终判断见<a href="阶段结论.md">阶段结论.md</a>；参数/源码与数据指纹见<a href="protocol.json">protocol.json</a>；完整评分<a href="analysis.json">analysis.json</a>；发布变化<a href="publication_diagnostics.json">publication_diagnostics.json</a>。各片段文件夹保留逐20ms JSONL、completion和配对耗时原始记录。</p>'
    if (OUT/'cli_smoke/summary.json').exists():
        cli=load(OUT/'cli_smoke/summary.json')
        text+=f'<p>独立RAW入口2秒烟测：{cli["frames"]}时刻，{cli["detected"]}次发布，含open/seek/解码/写视频实际耗时{cli["wall_including_open_seek_decode_write_preview_s"]:.2f}s。<a href="cli_smoke/observations.mp4">检测视频</a>是源50fps时间轴，不代表实时吞吐；逐观测一致性见<a href="cli_smoke/comparison.json">comparison.json</a>。</p>'
    text+='<p>未用G、未访问或执行yolov10-main；原src/阶段17推理源码及旧结果保留。源码使用说明见项目research/stage18/README.md；交接已更新WORKLOG。</p></main></html>'
    (OUT/'第十八阶段增量统计与轨迹确认报告.html').write_text(text,encoding='utf-8');print('Stage18 report rendered',flush=True)


if __name__=='__main__':main()
