"""Render frozen stage17 results; annotations are used here only for scoring/display."""
import html,json,hashlib
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from study import OUT,ROOT,CASES,SNAPS,load,save,verify,labels

NAMES={'clear6':'train6 清晰 22–27s','side6':'train6 侧视/离场/重现 108–120s',
       'review4':'train4 跨录像检查 74–84s','negative5':'train5 无无人机 0–8s'}
COLORS={'legacy16':'#b244ba','fixed_spectrum':'#8b6c30','global':'#168d79','roi':'#e47b19'}


def rows(case):return list(map(json.loads,(OUT/case/'observations.jsonl').read_text().splitlines()))
def fmt(x):return '—' if x is None else f'{x:.2f}'
def esc(x):return html.escape(str(x))
def table(headers,body):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in body)+'</tbody></table></div>'


def extended(allrows):
    result={'cases':{},'rotor_reference':[]}
    for case,rr in allrows.items():
        per={}
        ll=labels(case)
        times=np.array([r['t'] for r in rr])
        for m in ['global','roi']:
            qq=[r['methods'][m] for r in rr]
            timing=[q['timing_ms'] for q in qq if 'timing_ms' in q]
            diag=[]
            for label in ll:
                if label['state']!='present':continue
                i=np.searchsorted(times,label['t'],side='right')-1
                if i<0 or label['t']-times[i]>.025:continue
                q=qq[i];box=np.array(label['bbox']);center=(box[:2]+box[2:])/2;diagbox=np.linalg.norm(box[2:]-box[:2])
                o=q['observation'];det=o['detected']
                reason=('published_wrong' if np.linalg.norm(np.array(o['center'])-center)>diagbox*.5 else 'published') if det else ('confirmation_gate' if q.get('proposal_detected') else 'no_geometry_proposal')
                diag.append(dict(t=label['t'],reason=reason,accepted=len(q['candidates']),raw=len(q['raw_candidates']),
                    near_accepted=sum(np.linalg.norm([c['cx']-center[0],c['cy']-center[1]])<diagbox for c in q['candidates'])))
            per[m]=dict(spectrum_median_ms=float(np.median([x['spectrum'] for x in timing])),
                post_spectrum_median_ms=float(np.median([x['total']-x['spectrum'] for x in timing])),
                published_step_median_ms=float(np.median([q['timing_ms']['total'] for q in qq if q['observation']['detected']])) if any(q['observation']['detected'] for q in qq) else None,
                active_blocks_median=float(np.median([q.get('active_blocks',0) for q in qq])),
                total_step_le20ms=sum(x['total']<=20 for x in timing),total_timed_steps=len(timing),
                labels=diag,high_band_candidates=sum(c['block_frequency_hz']>500 for q in qq for c in q['candidates']))
        both=[r for r in rr if all(r['methods'][m]['observation']['detected'] for m in ['global','roi'])]
        diffs=[np.linalg.norm(np.array(r['methods']['global']['observation']['center'])-r['methods']['roi']['observation']['center']) for r in both]
        per['comparison']=dict(both_published=len(both),centers_equal_within1px=sum(d<=1 for d in diffs),
            max_center_difference_px=float(max(diffs)) if diffs else None,
            publication_agrees=sum(r['methods']['global']['observation']['detected']==r['methods']['roi']['observation']['detected'] for r in rr))
        result['cases'][case]=per
    refs=load(ROOT.parent/'output/stage8_onoff_rotors_20261017/blind_reference/references.json')
    for ref in refs['frames']:
        r=min(allrows['clear6'],key=lambda r:abs(r['t']-ref['t']));assert abs(r['t']-ref['t'])<1e-8
        for m in ['legacy16','global','roi']:
            q=r['methods'][m]
            options={'peaks':[x['centre'] for x in q['peaks']]} if m=='legacy16' else {
                'accepted_candidates':[[x['cx'],x['cy']] for x in q['candidates']],
                'published_rotors':[[x['cx'],x['cy']] for x in q['observation']['rotors']]}
            for kind,centers in options.items():
                distances=[]
                if centers:
                    d=np.linalg.norm(np.array(centers)[:,None,:]-np.array(ref['centers'])[None,:,:],axis=2)
                    a,b=linear_sum_assignment(d);distances=d[a,b].tolist()
                result['rotor_reference'].append(dict(t=ref['t'],method=m,kind=kind,n_candidates=len(centers),reference_centers=len(ref['centers']),
                    matched_at_radius_px={str(rad):sum(d<=rad for d in distances) for rad in [8,12,20]},distances_px=distances))
    result['rotor_reference_limits']='Three pre-existing AI approximate clear-disk centers, not independent motor ground truth; current integration20ms vs original10ms.'
    save(OUT/'extended_analysis.json',result)
    return result


def archived_baseline(allrows):
    """Same labels/causal alignment, different saved10ms baseline: not a one-factor ablation."""
    results=[];details=[]
    for case,dirname in [('clear6','stage12_clear_22_27_20261017'),('side6','stage12_side_108_120_20261017')]:
        path=ROOT.parent/'output'/dirname/'observations.jsonl'
        rr=list(map(json.loads,path.read_text().splitlines()));times=np.array([r['t'] for r in rr])
        pp=[];absent=[]
        for label in labels(case):
            i=np.searchsorted(times,label['t'],side='right')-1
            if i<0 or label['t']-times[i]>.025:continue
            o=rr[i]['observation'];row=dict(case=case,t=label['t'],observation_t=float(times[i]),state=label['state'],detected=o['detected'])
            if label['state']=='present':
                box=np.array(label['bbox']);diag=np.linalg.norm(box[2:]-box[:2]);target=(box[:2]+box[2:])/2
                error=float(np.linalg.norm(np.array(o['center'])-target)) if o['detected'] else None
                row.update(error_px=error,success=o['detected'] and error<=diag*.25,wrong=o['detected'] and error>diag*.5);pp.append(row)
            else:absent.append(row)
            details.append(row)
        errors=[p['error_px'] for p in pp if p['detected']]
        results.append(dict(case=case,method='archived_stage9_neighbor10_with_stage12_sensing',present_labels=len(pp),present_detected=len(errors),
            conditional_error_px=float(np.mean(errors)) if errors else None,joint_success=sum(r['success'] for r in pp),wrong_position=sum(r['wrong'] for r in pp),
            absent_labels=len(absent),absent_detections=sum(r['detected'] for r in absent),archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    new_details=load(OUT/'analysis.json')['details'];common=[]
    for case in ['clear6','side6']:
        old={r['t']:r for r in details if r['case']==case and r['state']=='present'}
        for m in ['global','roi']:
            new={r['t']:r for r in new_details if r['case']==case and r['method']==m and r['state']=='present'}
            ticks=[t for t in old if old[t]['detected'] and new[t]['detected']]
            common.append(dict(case=case,new_method=m,n=len(ticks),old_error_px=float(np.mean([old[t]['error_px'] for t in ticks])) if ticks else None,
                new_error_px=float(np.mean([new[t]['error_px'] for t in ticks])) if ticks else None))
    out=dict(table=results,details=details,common=common,limits='Archived unchanged neighbor10 front-end with frequency sensing; 10ms ticks vs new20ms and warmup differ. Same sparse labels/past<=25ms evaluation. A deployment reference, not a one-factor causal ablation.')
    save(OUT/'archived_baseline_comparison.json',out)
    return out


def audit():
    checks=[]
    for relative in ['stage9_state_recovery_20261017/protocol.json','stage16_blade_detector_20261017/run_0_35/protocol.json','stage16_blade_detector_20261017/run_105_120/protocol.json']:
        protocol=load(ROOT.parent/'output'/relative);hh=protocol['source_hashes']
        bad=[p for p,v in hh.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=v]
        assert not bad,(relative,bad)
        checks.append(dict(protocol=relative,n_sources=len(hh),unchanged=True))
    verify();save(OUT/'preservation_check.json',checks);return checks


def display_counts(on,off):
    # The image is raw ON+OFF count darkness, with no intensity reconstruction.
    dark=np.clip(np.log1p(on+off)/np.log(8),0,1)
    grey=np.uint8(255-230*dark)
    return cv2.cvtColor(grey,cv2.COLOR_GRAY2RGB)


def draw(im,q,method):
    a=im.copy()
    if method=='legacy16':
        for i,p in enumerate(q['peaks']):
            c=tuple(np.rint(p['centre']).astype(int));cv2.drawMarker(a,c,(178,68,186),cv2.MARKER_CROSS,16,2)
            cv2.putText(a,str(i+1),c,cv2.FONT_HERSHEY_SIMPLEX,.5,(178,68,186),1)
    else:
        for r in q['raw_candidates']:
            col=(20,150,130) if r['accepted'] else (180,180,180)
            c=tuple(np.rint([r['cx'],r['cy']]).astype(int));axes=tuple(np.maximum(1,np.rint([r['major']/2,r['minor']/2])).astype(int))
            cv2.ellipse(a,c,axes,r['angle'],0,360,col,2)
        roi=q['roi']
        if roi:cv2.rectangle(a,tuple(roi[:2]),tuple(roi[2:]),(220,165,30),1)
        o=q['observation']
        if o['detected']:
            box=np.rint(o['bbox']).astype(int);cv2.rectangle(a,tuple(box[:2]),tuple(box[2:]),(225,95,20),2)
            cv2.drawMarker(a,tuple(np.rint(o['center']).astype(int)),(225,95,20),cv2.MARKER_TILTED_CROSS,16,2)
    return a


def snapshots(allrows):
    links=[]
    for case,stamps in SNAPS.items():
        rr=allrows[case];ll=labels(case)
        for t in stamps:
            r=min(rr,key=lambda r:abs(r['t']-t));assert abs(r['t']-t)<1e-8
            snap=np.load(OUT/case/f'snapshot_{t:.2f}.npz');im=display_counts(snap['on'],snap['off']);h,w=im.shape[:2]
            points=[]
            for m in ['global','roi']:
                q=r['methods'][m]
                points.extend([[c['cx'],c['cy']] for c in q['raw_candidates']])
                if q['observation']['detected']:points.extend(np.array(q['observation']['bbox']).reshape(2,2).tolist())
            points.extend([p['centre'] for p in r['methods']['legacy16']['peaks']])
            nearest=min(ll,key=lambda l:abs(l['t']-t)) if ll else None
            if nearest and abs(nearest['t']-t)<.4 and nearest['state']=='present':points.extend(np.array(nearest['bbox']).reshape(2,2).tolist())
            if points:
                points=np.array(points);low=np.maximum(0,np.floor(points.min(0)-[70,55])).astype(int);high=np.minimum([w,h],np.ceil(points.max(0)+[70,55])).astype(int)
                if high[0]-low[0]<300:
                    c=(low[0]+high[0])/2;low[0]=max(0,int(c-150));high[0]=min(w,low[0]+300)
                if high[1]-low[1]<140:
                    c=(low[1]+high[1])/2;low[1]=max(0,int(c-70));high[1]=min(h,low[1]+140)
                x0,y0=low;x1,y1=high
            else:x0,y0,x1,y1=0,0,w,h
            fig,axes=plt.subplots(2,3,figsize=(16,7),layout='constrained')
            axes[0,0].imshow(im);axes[0,0].set_title(f'{case} {t:.2f}s / full native FOV')
            axes[0,0].add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='#2373a8'))
            for ax in axes[0,1:]:ax.axis('off')
            axes[0,1].text(0,.95,'20ms ON+OFF event count image\n\nPurple: old spectral top4 peaks\nTeal: accepted ellipse/line candidates\nGrey: geometry rejected by spectrum\nOrange: confirmed current observation\nGold: ROI bounds\n\nCrop includes all candidate/peak centers.\nA fitted ellipse is not motor truth.',va='top',fontsize=12)
            axes[0,2].text(0,.95,'Causal 100ms frequency history\n250us bins / Nyquist2000Hz\nBand60–1600Hz\n\nNo G reconstruction\nNo future events\nNo mass/size/thrust input\n\nThis figure shows detection,\nnot a future trajectory.',va='top',fontsize=12)
            for ax,m in zip(axes[1],['legacy16','global','roi']):
                q=r['methods'][m];img=draw(im,q,m)
                ax.imshow(img);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0)
                suffix=f"{len(q['peaks'])} peaks" if m=='legacy16' else f"candidates={len(q['candidates'])}, published={q['observation']['detected']}"
                ax.set_title(f'{m}: {suffix}',fontsize=11)
            for ax in axes.flat:
                if ax.axison:ax.set_xlabel('native x [px]');ax.set_ylabel('native y [px]')
            file=f'{case}_{t:.2f}.png';fig.savefig(OUT/file,dpi=130);plt.close(fig)
            links.append((case,t,file))
    return links


def plots(allrows):
    for case,rr in allrows.items():
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
        for m in ['global','roi']:
            tt=[r['t'] for r in rr];qq=[r['methods'][m] for r in rr];oo=[q['observation'] for q in qq]
            for ax,dim in zip(axes[:2],[0,1]):
                ax.plot(tt,[o['center'][dim] if o['detected'] else np.nan for o in oo],label=m,color=COLORS[m],lw=1.5,alpha=.8)
            axes[2].plot(tt,[q['timing_ms']['total'] if 'timing_ms' in q else np.nan for q in qq],label=m,color=COLORS[m],lw=.8,alpha=.7)
        for label in labels(case):
            if label['state']!='present':continue
            b=label['bbox']
            for dim,ax in enumerate(axes[:2]):ax.scatter(label['t'],(b[dim]+b[dim+2])/2,marker='x',color='black',s=35)
        axes[0].set_title(f'{case}: confirmed center (black x = sparse archived box center)')
        axes[0].set_ylabel('x [native px]');axes[1].set_ylabel('y [native px]')
        axes[2].set_ylabel('frontend wall [ms]');axes[2].set_xlabel('RAW time [s]');axes[2].axhline(20,color='red',ls='--',label='20ms output period')
        for ax in axes:ax.legend(loc='best');ax.grid(alpha=.2)
        fig.savefig(OUT/f'{case}_timeline.png',dpi=140);plt.close(fig)


def main():
    verify();a=load(OUT/'analysis.json');allrows={case:rows(case) for case in CASES};extra=extended(allrows);preserved=audit();baseline=archived_baseline(allrows)
    links=snapshots(allrows);plots(allrows)
    perf={case:load(OUT/case/'completion.json') for case in CASES}
    main_table=[]
    for r in a['table']:
        main_table.append([esc(NAMES[r['case']]),r['method'],f"{r['detected']}/{r['frames']}",
            f"{r['present_detected']}/{r['present_labels']}",fmt(r['conditional_error_px']),f"{r['joint_success']}/{r['present_labels']}",
            r['wrong_position'],f"{r['absent_detections']}/{r['absent_labels']}"])
    speed=[]
    for case,p in perf.items():
        for m,d in p['timing_ms'].items():
            speed.append([esc(NAMES[case]),m,d['n'],fmt(d['median']),fmt(d['p95']),
                ('100' if m in ['legacy16','fixed_spectrum'] else '20')])
    refs=[]
    for m,kind in [('legacy16','peaks'),('global','accepted_candidates'),('global','published_rotors'),('roi','published_rotors')]:
        dd=[r for r in extra['rotor_reference'] if r['method']==m and r['kind']==kind]
        refs.append([m,kind,sum(r['n_candidates'] for r in dd),12,*[f"{sum(r['matched_at_radius_px'][str(rad)] for r in dd)}/12" for rad in [8,12,20]]])
    diagrows=[]
    for case in CASES:
        for m in ['global','roi']:
            ds=extra['cases'][case][m];dd=ds['labels']
            diagrows.append([esc(NAMES[case]),m,*[sum(d['reason']==k for d in dd) for k in ['published','published_wrong','confirmation_gate','no_geometry_proposal']],
                fmt(ds['spectrum_median_ms']),fmt(ds['post_spectrum_median_ms'])])
    sections=[]
    for case in CASES:
        rr=next(r for r in a['table'] if r['case']==case and r['method']=='roi');p=perf[case]
        sections.append(f'<h3>{esc(NAMES[case])}</h3><p>ROI发布{rr["detected"]}/{rr["frames"]}；进入ROI计算{rr["roi_frames"]}次。四分支共同运行耗时{p["wall_s"]:.1f}s（不可当单分支实时耗时）。</p><a href="{case}_timeline.png"><img src="{case}_timeline.png" loading="lazy"></a>')
        sections.extend(f'<figure><a href="{file}"><img src="{file}" loading="lazy"></a><figcaption>{t:.2f}s：原生事件图及相同截止时刻的检测。点击放大。</figcaption></figure>' for c,t,file in links if c==case)
    summary=[]
    for case in CASES:
        r=next(r for r in a['table'] if r['case']==case and r['method']=='roi')
        summary.append(f'{NAMES[case]}：有机稀疏标签覆盖 {r["present_detected"]}/{r["present_labels"]}，条件中心误差 {fmt(r["conditional_error_px"])}px，发布 {r["detected"]}/{r["frames"]}。')
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第十七阶段：周期证据、几何验证与ROI</title><style>
body{font:17px/1.7 "Microsoft YaHei",sans-serif;color:#233343;background:#f4f6fa;margin:0}main{max-width:1250px;margin:auto;background:white;padding:28px 36px}h1{line-height:1.3;font-size:31px}h2{margin-top:45px;border-bottom:2px solid #dce6ed;padding-bottom:6px}a{color:#176a94}p{max-width:1100px}.note{padding:16px 22px;background:#edf6fb;border-left:5px solid #3382a5}.warn{background:#fff3df;border-color:#c58a26}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #d6dfe7;padding:9px;white-space:nowrap}th{background:#eaf0f5}img{max-width:100%;height:auto;border:1px solid #dbe1e8}figure{margin:25px 0}figcaption{font-size:14px;color:#607080}code{background:#eef1f4;padding:2px 5px}li{margin:6px 0}.small{font-size:14px;color:#5d6b78}details{border:1px solid #dce6ed;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:600}</style><main>
<h1>第十七阶段：把周期区域变成可核查的桨盘与整机观测</h1><p>2026-10-18 · 纯RAW ON/OFF · 35秒、四个固定片段 · 推理不读标签 · 配置先冻结</p>
<div class="note"><b>这轮完成了三项：</b>修复实际频带上限；复用现有椭圆/线段和状态恢复，以周期性验证几何候选；加入局部ROI跟踪及定期全局重搜。当前只验证检测/定位，尚未连接逐桨转速或未来位置预测。</div>
<h2>1. 结果与判断边界</h2>'''
    text+='<ul>'+''.join('<li>'+esc(s)+'</li>' for s in summary)+'</ul>'
    text+='<div class="note warn"><b>取舍：保留原默认前端。</b>新方案能把分散频谱峰转为较合理的桨盘几何，但侧视有机覆盖9/14，原存档前端同标签规则为13/14；ROI在train4共同3点的中心误差由全局13.99px变为37.38px，且清晰段并未提速。它还不构成可靠升级。</div>'
    text+='<p>表中的“有机覆盖”只看已存档的稀疏整机框；“联合通过”要求同一标签时刻既发布、又使中心误差≤框对角线的25%；“明显错位”定义为误差&gt;50%对角线。条件误差仅对成功发布时刻计算，必须结合覆盖看。不能把该数值称为未来位置预测精度。</p>'
    text+=table(['片段','方法','发布/处理时刻','有机覆盖','条件中心误差px','联合通过','明显错位','无机标签发布'],main_table)
    text+='<p>再把原阶段9邻域前端（阶段12存档，未改几何）的输出按同一批标签、同一“只取过去≤25ms”规则重新评分。旧为10ms步长、新为20ms，预热亦不同，因此是原有可用方案参照，不能归因到单项改动。新方案是否值得替换，必须面对这个对照。</p>'
    text+=table(['片段','旧前端有机覆盖','条件中心误差px','联合通过','明显错位','无机标签发布'],[
        [NAMES[r['case']],f"{r['present_detected']}/{r['present_labels']}",fmt(r['conditional_error_px']),f"{r['joint_success']}/{r['present_labels']}",r['wrong_position'],f"{r['absent_detections']}/{r['absent_labels']}"] for r in baseline['table']])
    text+=table(['片段','新方法','旧新共同发布标签','旧中心误差px','新中心误差px'],[
        [NAMES[r['case']],r['new_method'],r['n'],fmt(r['old_error_px']),fmt(r['new_error_px'])] for r in baseline['common']])
    text+='<p class="small">global=全画面频谱+几何+状态保护；roi=相同模型局部计算+定期全局。模型间精度比较另见<a href="analysis.json">analysis.json</a>的common；缺失输出没有用预测值填补。负例没有逐时人工框，但train5整段已知无无人机，发布次数直接作负对照。</p>'
    text+='<h2>2. 改动、复用与已有研究</h2><p><a href="https://github.com/radimspetlik/effqpdet">effqpdet作者代码</a>以事件率的周期突发找桨叶，并结合累积空间形状。它的原始C++此次没有编译或运行；本轮仍是自有FFT适配，不能写作完整复现。<a href="https://github.com/radimspetlik/helixtrack">HelixTrack作者代码</a>把相位和几何联合跟踪，但需要录像对应的位姿/转速初始化，本轮未直接套入。</p>'
    text+=table(['环节','本轮实际做法','局限'],[
        ['频带','1ms→250µs分箱；采样4kHz，Nyquist2kHz；实际60–1600Hz','频率分辨率仍约10Hz；峰不等于机械RPM'],
        ['计算','只为≥60事件的8px活跃块构建FFT','事件筛选/复制仍随事件数增长'],
        ['几何','直接调用既有PaperPerception椭圆/线段，允许纯ON或纯OFF；周期支持与频率离散门控','重叠轮廓仍可能合并；频率一致不代表语义正确'],
        ['状态','复用阶段9尺度保护/重新确认，另需连续3个20ms候选才发布','确认会延迟进入；侧飞连续缺测时可能一直不发布'],
        ['ROI','按历史整机框和速度扩边；每100ms至少全局重搜；失配160ms回全局','这是搜索约束，不是从历史恢复不可见的真实姿态']])
    text+='<p>100ms历史只供周期计算；几何使用当前20ms事件计数、按2归一到10ms等效计数。候选框要求至少15%空间块有周期支持，块内频率MAD/中位数≤0.35。完整冻结参数和源码SHA见<a href="protocol.json">protocol.json</a>。train4此前看过抽样画面，只称跨录像检查，不称完全盲测；未在本轮结果出来后调参数。</p>'
    text+='<h2>3. 四个峰是否变成四个桨盘</h2><p>用之前预存的23/24/25s三个清晰画面的12个人工智能粗估中心作补充匹配，采用匈牙利一对一匹配，列出8/12/20px半径敏感性。它们不是人类独立逐桨真值，不用于宣称正式召回率；新20ms与旧参考10ms积分窗也不同。候选列表可能有薄线分支重复，候选总数不可直接算作真实桨数。</p>'
    text+=table(['方法','匹配对象','总输出点','参考点','≤8px','≤12px','≤20px'],refs)
    text+='<h2>4. 算得快了吗</h2><p>以下为相同四分支研究循环内各方法调用耗时，不含读取、历史拼接和JSON写入。新分支每20ms调用（计划50Hz），旧与纯频谱每100ms调用（10Hz）。调用频率不等于实际实时吞吐率；若耗时高于发布周期，会积压。完整循环wall仅作可复跑记录，不冒充单分支实时速度。</p>'
    text+=table(['片段','方法','调用数','中位ms','P95 ms','计划周期ms'],speed)
    text+='<p>剔除空档、只看实际发布时刻的中位前端耗时：'+ '；'.join(f'{esc(NAMES[case])} ROI {fmt(extra["cases"][case]["roi"]["published_step_median_ms"])}ms' for case in ['clear6','side6','review4'])+'。低事件率空档会拉低全片段中位数，不能据此宣布持续实时。</p>'
    if (OUT/'hotspots.json').exists():
        hotspots=load(OUT/'hotspots.json')
        text+='<p>固定23/110/75s的事后cProfile诊断（短前缀、仅一次带插桩调用，不是速度基准）显示：约104万/446万/226万历史事件，频谱路径与ROI事件筛选占主要时间；椭圆候选函数本身累计约1–3ms。ROI每次仍需扫描历史事件，几何还保留原生全图数组分配，因此小框不必然省时。见<a href="hotspots.json">hotspots.json</a>。</p>'
    text+='<h2>5. 当前还卡在哪一层</h2><p>下面按稀疏有机标签分解：没有几何整机提案，还是已有提案但未连续确认。频谱耗时与余下几何/状态耗时分开列出，避免把稀疏FFT误称为全部提速。</p>'
    text+=table(['片段','方法','发布且未明显错位','发布但明显错位','确认拦截','无整机提案','频谱中位ms','余下中位ms'],diagrows)
    text+='<p>周期存在仍不自动证明是无人机或某个电机。侧视下空间轮廓粘连/遮挡与近线段中心不确定性，没有因为加ROI而消失。三帧确认与状态保护提高发布要求，也可能降低覆盖；这轮保留此失败，下一轮应使用固定失败片段验证分离方案，而非直接降低所有阈值。频率层和预测层等候可靠定位；本轮不声称推力或轨迹提高。</p>'
    text+='<p>本轮侧视ROI的5个漏标时刻里，3个已产生几何提案，但被连续确认拦截，另2个没有整机提案；并非全部是“频谱找不到桨叶”。下一步更合适的是：先用原前端作主轨迹，把周期证据用作候选评分/身份验证；测试按轨迹确认并带迟滞的发布规则；并把分块直方图改为增量环形缓冲，几何真正按裁剪坐标工作。分别做消融，避免一口气换掉全部链条。</p>'
    text+='<h2>6. 查看画面、轨迹与失败</h2>'+''.join(sections)
    text+='<h2>7. 复核与交接</h2><ul><li>已做已知频率200/350/600/1200/1600Hz合成检验；1ms非法频带拒绝；纯单极四盘、连续确认、空输入、ROI回全局和未来事件隔离检查。见<a href="module_checks.json">module_checks.json</a>。</li>'
    text+=f'<li>清晰段新前端的真实RAW前缀重放：{perf["clear6"]["causal_prefix_times"]}个截止时刻，global与roi分别逐项一致。旧阶段16重算与存档严格一致的截止时刻：清晰{perf["clear6"]["legacy_cache_exact_times"]}、侧视{perf["side6"]["legacy_cache_exact_times"]}。</li>'
    text+='<li>原阶段9与阶段16两份冻结协议源码全部SHA一致：'+', '.join(str(p['n_sources']) for p in preserved)+'个/协议；见<a href="preservation_check.json">preservation_check.json</a>。旧src、旧默认入口和历史结果保留。</li><li>未使用重建G，未访问或执行yolov10-main；无真实RPM/IMU/独立连续姿态真值。不把AI中心参考、稀疏框或检测器自身输出当作实飞物理真值。</li></ul>'
    if (OUT/'cli_smoke/comparison.json').exists():
        text+='<p>独立RAW入口已烟测0.4s：20个时刻、17次发布，附<a href="cli_smoke/observations.mp4">检测视频</a>（仅入口烟测，视频50fps是原始时间轴，不是实际运行速度）。其20个输出与本轮ROI结果逐项相同，见<a href="cli_smoke/comparison.json">comparison.json</a>。</p>'
    text+='<p class="small">运行记录：四段completion与analysis已正常保存；直接运行study.py，最后控制台打印表格会因NumPy int64不能直接JSON序列化而报错（评分已先落盘）。冻结文件保留原样，rerun.py包装入口处理这个已知打印错误；已通过帧数、JSON完整性与源码指纹复核。见<a href="delivery_checks.json">delivery_checks.json</a>。</p>'
    text+='<p>可复用源码：<code>event_drone_forecast/research/stage17/frontend.py</code>；固定实验入口<code>study.py</code>；独立RAW检测入口<code>observe.py</code>；详细计分<a href="analysis.json">analysis.json</a>、<a href="extended_analysis.json">extended_analysis.json</a>。最终取舍与下一步见<a href="阶段结论.md">阶段结论.md</a>。</p></main></html>'
    (OUT/'第十七阶段桨盘几何与ROI实验报告.html').write_text(text,encoding='utf-8')
    print('Report rendered',flush=True)


if __name__=='__main__':main()
