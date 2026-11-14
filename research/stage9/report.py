"""Evidence report for fixed development experiment, with actual observations."""
from pathlib import Path
import hashlib,json,shutil,html
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import ROOT,OUT,OLD,CASES,METHODS,legacy8,load,save,verify,sha

NAMES={'strict10':'Original 10ms','neighbor10':'Neighbor 10ms',
       'guard_strict10':'Protected original','guard_neighbor10':'Protected neighbor',
       'protect_neighbor10':'Protection only (ablation)'}


def traces(case):return [json.loads(x) for x in (OUT/case/'observations.jsonl').read_text().splitlines()]


def number(x):return '—' if x is None else f'{x:.2f}'


def plots():
    for case in ['clear','side']:
        rr=traces(case);tt=[r['t'] for r in rr];fig,axes=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
        for m in METHODS:
            obs=[r['methods'][m]['observation'] for r in rr]
            xx=[o['center'][0] if o['detected'] else np.nan for o in obs]
            yy=[o['center'][1] if o['detected'] else np.nan for o in obs]
            span=[max(np.array(o['bbox'][2:])-o['bbox'][:2]) if o['detected'] else np.nan for o in obs]
            for ax,values in zip(axes.flat[:3],[xx,yy,span]):ax.plot(tt,values,label=NAMES[m],linewidth=1,alpha=.85)
        for label in legacy8.labels(case):
            if label['state']!='present':continue
            box=np.array(label['bbox']);center=(box[:2]+box[2:])/2
            axes[0,0].scatter(label['t'],center[0],c='k',marker='x',s=30)
            axes[0,1].scatter(label['t'],center[1],c='k',marker='x',s=30)
        for m in METHODS:
            yy=[i if r['methods'][m]['observation']['detected'] else np.nan for i,r in enumerate(rr)]
            axes[1,1].plot(tt,[METHODS.index(m) if np.isfinite(v) else np.nan for v in yy],'.',markersize=2,label=NAMES[m])
        for ax,title in zip(axes.flat,['Current center X (cross: sparse box center)','Current center Y (cross: sparse box center)',
                                     'Output box max extent (px)','Detection output times (not recall truth)']):
            ax.set_title(title);ax.set_xlabel('Sensor time (s)');ax.grid(alpha=.2)
        axes[1,1].set_yticks(range(len(METHODS)),[NAMES[m] for m in METHODS]);axes[0,0].legend(fontsize=8)
        fig.suptitle(case+' | causal observations only; gaps are missing outputs')
        fig.savefig(OUT/case/'timeline.png',dpi=140);plt.close(fig)


def white(on,off):
    a=np.log1p(on);b=np.log1p(off);a/=max(float(a.max()),1e-6);b/=max(float(b.max()),1e-6)
    return np.uint8(np.stack([255*(1-a),255*(1-a),255*(1-b)],-1))


def panel(base,q,name,t,crop=None):
    pic=base.copy();o=q['observation']
    if o['detected']:
        for r in o['rotors']:
            cv2.ellipse(pic,(round(r['cx']),round(r['cy'])),(max(1,round(r['major']/2)),max(1,round(r['minor']/2))),r['angle'],0,360,(20,100,230),2)
        b=np.rint(o['bbox']).astype(int);c=np.rint(o['center']).astype(int)
        cv2.rectangle(pic,tuple(b[:2]),tuple(b[2:]),(50,150,30),2);cv2.drawMarker(pic,tuple(c),(50,150,30),cv2.MARKER_CROSS,15,2)
    if crop is not None:
        x0,y0,x1,y1=crop;pic=pic[y0:y1,x0:x1]
    h,w=pic.shape[:2];scale=min(768/w,432/h);ww,hh=round(w*scale),round(h*scale)
    resized=cv2.resize(pic,(ww,hh),interpolation=cv2.INTER_AREA)
    pic=np.full((432,768,3),255,np.uint8);pic[(432-hh)//2:(432-hh)//2+hh,(768-ww)//2:(768-ww)//2+ww]=resized
    out=np.full((478,768,3),255,np.uint8);out[46:]=pic;out[:46]=25
    info='measured/inferred box' if o['detected'] else 'NO DETECTION'
    if q.get('audit',{}).get('transferred'):info+=' | confirmed recovery'
    cv2.putText(out,f'{name} t={t:.2f} | {info}',(9,29),0,.54,(240,240,240),1)
    return out


def pictures():
    links=[]
    for case in CASES:
        rr=traces(case)
        for t in legacy8.SNAPSHOTS[case]:
            r=min(rr,key=lambda r:abs(r['t']-t));assert abs(r['t']-t)<1e-8
            raw=np.load(OLD/case/f'snapshot_{t:.2f}.npz');base=white(raw['on'],raw['off'])
            tiles=[panel(base,r['methods'][m],NAMES[m],t) for m in METHODS]
            image=np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])])
            p=OUT/case/f'comparison_{t:.2f}.png';p.write_bytes(cv2.imencode('.png',image)[1].tobytes())
            entry=dict(case=case,t=t,image=p.relative_to(OUT).as_posix())
            boxes=[r['methods'][m]['observation']['bbox'] for m in METHODS if r['methods'][m]['observation']['detected']]
            if boxes:
                boxes=np.array(boxes);lo=np.maximum(np.floor(boxes[:,:2].min(0)-35).astype(int),0)
                hi=np.minimum(np.ceil(boxes[:,2:].max(0)+35).astype(int),[base.shape[1],base.shape[0]])
                crop=[*lo,*hi];tiles=[panel(base,r['methods'][m],NAMES[m],t,crop) for m in METHODS]
                zoom=np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])])
                zp=OUT/case/f'zoom_{t:.2f}.png';zp.write_bytes(cv2.imencode('.png',zoom)[1].tobytes())
                entry.update(zoom=zp.relative_to(OUT).as_posix(),display_crop=crop)
            links.append(entry)
    save(OUT/'figure_manifest.json',links);return links


def document(links):
    a=load(OUT/'analysis.json');rows=[];common=[]
    for r in a['table']:
        rows.append(f"<tr><td>{r['case']}</td><td>{r['method']}</td><td>{r['detected']}/{r['published']}</td><td>{r['four_selected']}</td><td>{r['present_detected']}/{r['present']}</td><td>{number(r['conditional_error_px'])}</td><td>{r['success_quarter_diagonal']}/{r['present']}</td><td>{r['wrong_half_diagonal']}</td><td>{r['absent_detections']}/{r['absent']}</td><td>{r['transfers']}</td><td>{r['state_rejections']}</td></tr>")
    for r in a['common']:
        common.append(f"<tr><td>{r['case']}</td><td>{r['a']} → {r['b']}</td><td>{r['common_labels']}</td><td>{number(r['mean_a'])}</td><td>{number(r['mean_b'])}</td></tr>")
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第九阶段：整机状态保护与重新确认</title><style>body{background:#eef3f7;color:#21324a;font:15px/1.8 "Microsoft YaHei",sans-serif;margin:0}main{max-width:1300px;padding:25px;margin:auto}section,details{background:white;padding:18px;border-radius:10px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;text-align:left;border-bottom:1px solid #d8e0e9;white-space:nowrap}th{background:#dce8f5}img{max-width:100%}.scroll{overflow:auto}a{color:#196d9c}.note{border-left:4px solid #b77c23;background:#fff6e2;padding:15px}</style><main>'''
    doc+='<h1>第九阶段：整机状态保护与重新确认</h1><p>2026-10-17 · 原生ON/OFF · 无G · 全图因果推理 · 不访问yolov10-main</p>'
    doc+='<section><a href="阶段结论.md">先看结论与下一步</a><p>针对第八阶段已定位的“局部小组合污染整机状态→历史门控拒绝后续组合”做独立研究修复。原默认预测入口、旧源码和旧结果保留。实验只衡量当前检测观测，不宣称RPM、姿态或未来轨迹精度提升。</p></section>'
    doc+='<section><h2>结果：修复了本段邻域分支的大部分退化</h2><p>侧视邻域10ms输出121→164/200；6个稀疏标签覆盖3→5，输出且误差≤0.25框对角线也为3→5。共同3标签当前中心误差21.47→15.42px，新增两点误差12.30、3.49px。清晰段保持189/200。原严格10ms侧视167/200，保护后166/200，不能称所有方法均提升。</p><p>FRED4/6小目标未改善，严格分支FRED4还少覆盖1个标签；缺席/离场175帧均无发布，负例很有限。侧视仅保护消融为120/200，完整修复164/200，覆盖恢复主要来自重新确认。每10ms审计确认109.93/109.94/109.95连续计数1/2/3，109.95s完成一次轨迹转移，之后110s恢复完整输出。</p></section>'
    doc+='<h2>固定改动</h2><section><p>① 四个实际候选连续三帧、直径一致时建立尺度参考；少量部分候选不刷新可靠布局年龄。② 超出参考范围的候选过滤，异常整机尺度更新回滚中心、速度、布局、转速状态。③ 原跟踪失配80ms后才考察独立新组合，至少三个候选、连续三帧确认后重置运动状态。没有以外推充当检出，也不无限扩大原距离门限。</p><p>复用现有分组、椭圆、关联与原局部布局拟合；外部<a href="https://github.com/nwojke/deep_sort/blob/master/deep_sort/track.py">DeepSORT状态管理</a>和<a href="https://github.com/FoundationVision/ByteTrack/blob/main/yolox/tracker/byte_tracker.py">ByteTrack丢失与激活</a>仅查阅，没有执行作者整套网络。尺度参考仍可能由错误四候选建立，不是无人机语义识别。</p></section>'
    doc+='<div class="note">19.5秒开发片段、975发布时刻，与第八阶段相同。原neighbor10本轮重新计算并逐项匹配缓存；其他旧对照使用冻结缓存。标签仅事后评分。FRED4/6不是新的盲测，也不根据结果反复调参。protect_neighbor10只在侧视运行，用于分离状态保护与重新确认的作用。</div>'
    doc+='<h2>观测对照</h2><p>输出帧数不是召回真值；“四候选”不等于真实四桨。误差是稀疏整机框中心的当前误差。≤0.25框对角线成功率以全部有机标签为分母；&gt;0.5框对角线为错误位置代理，不等于确定的身份误锁。无机真标签只5个，不把未标注时刻当负样本。</p><div class="scroll"><table><tr><th>片段</th><th>方法</th><th>输出帧</th><th>四候选帧</th><th>有机标签输出</th><th>条件误差px</th><th>输出且≤0.25diag</th><th>&gt;0.5diag</th><th>无机误发布</th><th>发布帧中确认转移</th><th>发布时刻状态拒绝</th></tr>'+''.join(rows)+'</table></div>'
    doc+='<h2>共同有输出标签</h2><p>条件均值需连同上表覆盖和逐点结果阅读；没有输出的样本不被当作零误差。共同样本少，单点可主导平均值。</p><div class="scroll"><table><tr><th>片段</th><th>比较</th><th>共同标签</th><th>原误差px</th><th>新误差px</th></tr>'+''.join(common)+'</table></div>'
    doc+='<h2>连续状态与固定时刻</h2><p>曲线断开表示没有发布；黑叉为稀疏整机框中心。下方图为原生ON/OFF可视化上叠加检测，绿色为输出框/中心，橙色为选中候选。放大裁剪采用四方法输出框并集，只用于展示，保持宽高比；全视野原图均提供。推理没有标注或裁剪输入。可能包含推断整机框，不把框的补全叫作实际四桨测量。</p>'
    for case in ['clear','side']:doc+=f'<section><h3>{case}</h3><img src="{case}/timeline.png"></section>'
    for r in links:doc+=f'<details {"open" if r["case"]=="side" and r["t"]==110 else ""}><summary>{r["case"]}，{r["t"]:.2f}s</summary><img src="{r.get("zoom",r["image"])}"><p><a href="{r["image"]}">全视野原图</a></p></details>'
    doc+='<section><h2>10ms状态事件补充审计</h2><p>主结果每20ms发布，109.95s的转移发生在两次发布之间，所以主表的“发布帧中确认转移”为0。另行重放同一冻结前端，200发布时刻逐项相等：403处理帧中尺度参考活跃139帧，尺度过滤作用于23帧，整机尺度回滚触发0次，确认转移1次。没有证据称回滚机制本轮贡献了精度收益。</p><a href="side/event_audit.json">完整事件计数</a> · <a href="side/audit_10ms.jsonl">10ms逐帧状态</a></section>'
    doc+='<h2>审查与限制</h2><section><p>DSH→DeepSeek做了一次文本审查，原始意见与逐项监督复核均保留；冻结前显式补充新轨迹的频率/事件历史清理，未再调用。未发送RAW、图片或密钥。重新确认是单目标几何假设，不保证多无人机身份保持；若错误局部组持续通过旧门控后被尺度回滚，新挑战可能不能累积，这是本轮仍有的恢复边界。</p><a href="dsh_review/监督复核.md">监督复核</a> · <a href="dsh_review/result.json">实际调用</a> · <a href="independent_review.md">独立指标复核</a></section>'
    doc+='<p><a href="protocol.json">冻结协议</a> · <a href="module_checks.json">模块检查</a> · <a href="analysis.json">逐标签与共同样本</a> · <a href="validation.json">验证</a></p></main></html>'
    (OUT/'第九阶段整机状态保护报告.html').write_text(doc,encoding='utf-8')


def validate():
    verify();prior=[]
    for stage in ['stage6_body_axis_v2_20261017','stage7_localization_budget_v3_20261017']:
        hashes=load(ROOT.parent/'output'/stage/'protocol.json')['source_hashes']
        for rel,expected in hashes.items():assert sha(ROOT/rel)==expected,(stage,rel)
        prior.append(dict(stage=stage,sources_unchanged=len(hashes)))
    cases=[]
    for case,cfg in CASES.items():
        c=load(OUT/case/'completion.json');assert c['published']==round(cfg['duration']/.02)
        assert c['neighbor_baseline_exact']==c['published'] and not c['labels_in_inference'] and not c['G_used']
        for r in traces(case):
            for m,q in r['methods'].items():
                o=q['observation'];assert o['t']==r['t']
                if o['detected']:assert np.isfinite(o['center']).all()
        cases.append(c)
    assert next(c for c in cases if c['case']=='side')['prefix_equal_processed']>=50
    save(OUT/'validation.json',dict(frozen_sources_unchanged=True,prior=prior,cases=cases,
        cached_baseline_equal=True,causal_prefix_checked=True,yolov10_main_accessed=False))
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.is_file() and p.suffix in ['.py','.md']:shutil.copyfile(p,snapshot/p.name)


if __name__=='__main__':
    verify();plots();links=pictures();validate();document(links);print('Stage9 report complete',flush=True)
