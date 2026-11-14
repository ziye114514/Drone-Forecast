"""Post-score visual report, with explicitly selected active-model debug frames."""
import collections,html,json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study19 import ROOT,OUT,CASES,METHODS,SNAPS,load,save,verify
from event_drone_forecast.raw_io import open_raw,raw_chunks

EXTRAS={'side6':[112.4,112.56],'review4':[78.38]}
NAMES={'clear6':'train6清晰','side6':'train6侧视','review4':'train4跨录像','negative5':'train5无机'}


def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'


def main():
    verify();a=load(OUT/'analysis.json');allrows={c:list(map(json.loads,(OUT/c/'observations.jsonl').read_text().splitlines())) for c in CASES}
    diagnostic=[]
    for case,rr in allrows.items():
        counts=collections.Counter(r['methods']['layout20']['layout'].get('matched_count') for r in rr if r['methods']['layout20']['layout'].get('matched'))
        changed=sum(r['methods']['soft20']['observation'].get('center')!=r['methods']['layout20']['observation'].get('center') for r in rr)
        diagnostic.append(dict(case=case,matched_counts=dict(counts),changed_current_centers=changed,
            full_refresh_times=[r['t'] for r in rr if r['methods']['layout20']['layout'].get('full_template_refreshed')]))
    save(OUT/'layout_diagnostics.json',dict(cases=diagnostic,posthoc_visual_frames=EXTRAS,scope='Targeted model-active examples chosen after scoring, not extra evaluation samples.'))
    for case,stamps in EXTRAS.items():
        for t in stamps:
            target=OUT/case/f'snapshot_{t:.2f}.npz'
            if target.exists():continue
            reader=open_raw(CASES[case]['raw']);h,w=reader.get_size()
            chunks=list(raw_chunks(reader,t-.02,.02,20000));assert len(chunks)==1 and abs(chunks[0][0]-t)<1e-8
            e=chunks[0][1];pixels=e['y'].astype(np.int32)*w+e['x'];total=np.bincount(pixels,minlength=w*h).reshape(h,w)
            np.savez_compressed(target,total=total)
    links=[]
    for case,rr in allrows.items():
        for t in sorted(SNAPS[case]+EXTRAS.get(case,[])):
            r=min(rr,key=lambda r:abs(r['t']-t));assert abs(r['t']-t)<1e-8
            counts=np.load(OUT/case/f'snapshot_{t:.2f}.npz')['total'];h,w=counts.shape
            image=np.uint8(255-230*np.clip(np.log1p(counts)/np.log(8),0,1));points=[]
            for m in METHODS:
                o=r['methods'][m]['observation']
                if o['detected']:points.extend(np.array(o['bbox']).reshape(2,2).tolist())
            if points:
                p=np.array(points);lo=np.maximum(0,np.floor(p.min(0)-[60,50])).astype(int);hi=np.minimum([w,h],np.ceil(p.max(0)+[60,50])).astype(int)
                x0,y0=lo;x1,y1=hi
            else:x0,y0,x1,y1=0,0,w,h
            fig,axes=plt.subplots(1,4,figsize=(18,4.8),layout='constrained');axes[0].imshow(image,cmap='gray',vmin=0,vmax=255);axes[0].set_title(f'{case} {t:.2f}s / raw count')
            for ax,m,color in zip(axes[1:],METHODS,['#197f9c','#d57a1c','#20976d']):
                q=r['methods'][m];o=q['observation'];ax.imshow(image,cmap='gray',vmin=0,vmax=255)
                if o['detected']:
                    box=o['bbox'];ax.add_patch(plt.Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,color=color,lw=2));ax.scatter(*o['center'],marker='x',s=60,color=color)
                    for rotor in o['rotors']:ax.scatter(rotor['cx'],rotor['cy'],s=16,facecolors='none',edgecolors=color)
                suffix=''
                if m=='layout20':suffix=f"\nmodel={q['layout'].get('template_available_after',False)}, matched={q['layout'].get('matched_count',0)}"
                ax.set_title(f"{m}: published={o['detected']}"+suffix,fontsize=10)
            for ax in axes:ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_xlabel('native x [px]');ax.set_ylabel('native y [px]')
            name=f'{case}_{t:.2f}.png';fig.savefig(OUT/name,dpi=125);plt.close(fig);links.append((case,t,name,t in EXTRAS.get(case,[])))
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第十九阶段：软周期证据与四点布局</title><style>body{font:17px/1.75 "Microsoft YaHei",sans-serif;margin:0;background:#f3f6fa;color:#243342}main{max-width:1250px;margin:auto;background:white;padding:30px 36px}h1{font-size:31px;line-height:1.3}h2{margin-top:38px;border-bottom:2px solid #dce4ec}a{color:#176a96}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce4ec;white-space:nowrap;text-align:left}th{background:#edf2f7}.note{padding:18px;background:#fff4df;border-left:5px solid #c78b26}img{max-width:100%;height:auto;border:1px solid #dce4ec}details{padding:14px;border:1px solid #dce4ec;margin:18px 0}summary{cursor:pointer;font-weight:bold}figure{margin:20px 0}figcaption{font-size:14px;color:#637386}</style><main><h1>第十九阶段：周期增强与有期限的四点布局匹配</h1><p>2026-10-18 · 35秒自采RAW · 同一20ms输入 · 三分支消融 · 无结果后调参</p>
<div class="note"><b>取舍：</b>周期增强在train4有收益、在侧视train6退步；布局匹配只在少量侧视时刻有机会工作，稀疏标签未观察到额外收益。保留原默认，不能写作已解决侧视检测。</div>
<h2>1. 同输入结果</h2><p>neighbor20使用原邻域特征；soft20保留邻域基底，加入0.5×ON+OFF周期支持及有限活动评分加成，不硬筛所有非周期候选；layout20在相同软增强上再加已确认的四点布局。三者均使用当前20ms计数除2；没有再加阶段17/18外层发布门。当前框误差不是未来预测误差。</p>'''
    text+=table(['片段','方法','发布/总时刻','有机覆盖','条件中心误差px','联合通过','无机标签发布'],[[NAMES[r['case']],r['method'],f"{r['detected']}/{r['frames']}",f"{r['present_detected']}/{r['present_labels']}",fmt(r['conditional_error_px']),f"{r['joint_success']}/{r['present_labels']}",f"{r['absent_detections']}/{r['absent_labels']}"] for r in a['table']])
    text+='<p>标签取过去≤25ms观测；联合通过=发布且误差≤框对角线25%。train5整段明确无机，三分支各0/400发布。条件误差必须结合覆盖读。</p>'+table(['三者共同有机样本','数量','neighbor20误差px','soft20误差px','layout20误差px'],[[NAMES[r['case']],r['n'],*[fmt(r['errors_px'][m]) for m in METHODS]] for r in a['common']])
    text+='<h2>2. 四点模型是否建立、是否用上</h2><p>仅由连续3次实测四桨、凸四点与非退化二维分布建立模板；部分匹配采用已实现的stage11相似变换与匈牙利关联，局部尺度/旋转/残差有门槛。两三桨可更新当前匹配，但不刷新完整四桨年龄；最后完整观测300ms后失效，无当前候选不发布。相似模型是短时图像近似，不能表达任意姿态变化。</p>'
    text+=table(['片段','模板可用时刻','成功匹配','匹配2/3/4点次数','改变当前中心时刻'],[[NAMES[d['case']],next(r['template_frames'] for r in a['table'] if r['case']==d['case'] and r['method']=='layout20'),sum(d['matched_counts'].values()),'/'.join(str(d['matched_counts'].get(k,0)) for k in [2,3,4]),d['changed_current_centers']] for d in diagnostic])
    text+='<p>清晰段模板可用222/250时刻；侧视仅17/600，完整模板主要在112.28–112.32s刷新。侧视确有13个当前中心被改变，但这些短暂变化没有被本轮稀疏标签充分覆盖，因此不能据标签零增益就断言布局匹配永远无用。需要该短片段密集独立框或受控遮挡实验进一步检验；也不能用模型自己的中心证明自己正确。</p>'
    text+='<h2>3. 具体画面</h2><p>缩放窗口包含所有方法发布框；无发布时展示全图。空心点为本时刻实际输出的桨中心。112.40/112.56/78.38s是评分后按模型活跃选出的诊断画面，明确不作为新增评测样本。</p>'
    for case in CASES:
        text+=f'<details><summary>{NAMES[case]}</summary>'
        for c,t,name,posthoc in links:
            if c==case:text+=f'<figure><a href="{name}"><img src="{name}" loading="lazy"></a><figcaption>{t:.2f}s，'+('事后模型活跃诊断。' if posthoc else '协议预选画面。')+'</figcaption></figure>'
        text+='</details>'
    text+='<h2>4. 实施范围与后续</h2><p>查阅了<a href="https://github.com/siavashk/pycpd">PyCPD作者代码</a>及<a href="https://arxiv.org/abs/0905.2635">CPD原始论文</a>；本轮四点问题直接执行已有stage11拟合与stage9状态代码，没有执行PyCPD或宣称新配准算法。无真实电机轴/姿态/RPM真值，不把候选数量当真实四桨召回。</p><p>下一步应分别检查：20ms积分是否合并移动桨盘（对照更短原生时间窗）；模板在侧视进入前能否可靠建立（受控遮挡及真正清晰→侧视片段）；以及独立密集框。不要靠无限延长模板年龄凑覆盖。</p>'
    text+='<p>配置、源码及数据指纹见<a href="protocol.json">protocol.json</a>；<a href="module_checks.json">不变量检查</a>；<a href="analysis.json">完整计分</a>；<a href="layout_diagnostics.json">布局诊断</a>；<a href="阶段结论.md">阶段结论</a>。各片段保留逐时刻JSONL与完成记录。train4曾看过抽样，属于跨录像检查；这些数据已用于开发，非盲测。</p><p>未用G，未访问/执行yolov10-main，旧src/默认入口/历史冻结文件保留。没有未来轨迹改善声明。</p></main></html>'
    (OUT/'第十九阶段软周期与四点布局报告.html').write_text(text,encoding='utf-8');print('Stage19 report rendered',flush=True)


if __name__=='__main__':main()
