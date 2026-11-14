"""Diagnostic visuals and conservative scoring for frozen ON/OFF pilot."""
from pathlib import Path
import json,hashlib,sys
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from study import OUT,ROOT,METHODS,CASES,load,save,verify

def match(points,candidates,radius):
    if not candidates:return dict(matched=0,unmatched=len(candidates),distances=[])
    p=np.array(points);q=np.array([[r['cx'],r['cy']] for r in candidates]);d=np.linalg.norm(p[:,None]-q[None,:],axis=2)
    a,b=linear_sum_assignment(np.where(d<=radius,d/(radius+1),10000.))
    good=d[a,b]<=radius
    return dict(matched=int(good.sum()),unmatched=int(len(candidates)-good.sum()),distances=d[a[good],b[good]].tolist())

def reference_score():
    refs=load(OUT/'blind_reference/references.json');rows=[]
    for frame in refs['frames']:
        records=[json.loads(x) for x in (OUT/frame['case']/'observations.jsonl').read_text().splitlines()]
        r=min(records,key=lambda r:abs(r['t']-frame['t']));assert abs(r['t']-frame['t'])<1e-8
        for method in METHODS:
            for kind,rr in [('candidates',r['methods'][method]['candidates']),('selected',r['methods'][method]['observation']['rotors'])]:
                for radius in [8,12,20]:rows.append(dict(case=frame['case'],t=frame['t'],method=method,kind=kind,radius_px=radius,total_visible=4,**match(frame['centers'],rr,radius)))
    summary=[]
    for method in METHODS:
        for kind in ['candidates','selected']:
            for radius in [8,12,20]:
                rr=[r for r in rows if r['method']==method and r['kind']==kind and r['radius_px']==radius]
                summary.append(dict(method=method,kind=kind,radius_px=radius,visible_reference=12,
                    matched=sum(r['matched'] for r in rr),unmatched_candidates=sum(r['unmatched'] for r in rr),
                    matched_mean_distance_px=float(np.mean([d for r in rr for d in r['distances']])) if any(r['distances'] for r in rr) else None))
    save(OUT/'visual_reference_score.json',dict(reference=refs,details=rows,summary=summary,
        limitations='AI approximate centers on3fixed clear frames, not human rotor ground truth. Matching thresholds all reported. Unmatched candidates can include duplicates; not a formal FP rate.'))
    return summary

def white_onoff(on,off):
    a=np.log1p(on);b=np.log1p(off);a/=max(float(a.max()),1e-6);b/=max(float(b.max()),1e-6)
    return np.uint8(np.stack([255*(1-a),255*(1-a),255*(1-b)],axis=-1))

def fit_panel(im,label,box=None):
    if box is not None:
        x0,y0,x1,y1=box;im=im[y0:y1,x0:x1]
    h,w=im.shape[:2];ratio=min(640/w,360/h);ww,hh=round(w*ratio),round(h*ratio)
    pic=cv2.resize(im,(ww,hh),interpolation=cv2.INTER_NEAREST)
    canvas=np.full((400,640,3),245,np.uint8);canvas[40+(360-hh)//2:40+(360-hh)//2+hh,(640-ww)//2:(640-ww)//2+ww]=pic
    canvas[:40]=25;cv2.putText(canvas,label,(9,27),0,.59,(230,230,230),1)
    return canvas

def draw(im,rotors,color):
    for r in rotors:
        center=(round(r['cx']),round(r['cy']));axes=(max(1,round(r['major']/2)),max(1,round(r['minor']/2)))
        cv2.ellipse(im,center,axes,r['angle'],0,360,color,1,cv2.LINE_AA)
        cv2.drawMarker(im,center,color,cv2.MARKER_CROSS,8,1)

def figures():
    refs=load(OUT/'blind_reference/references.json');links=[]
    for case in CASES:
        folder=OUT/case;records=[json.loads(x) for x in (folder/'observations.jsonl').read_text().splitlines()]
        for p in sorted(folder.glob('snapshot_*.npz')):
            data=np.load(p);t=float(data['t']);r=min(records,key=lambda r:abs(r['t']-t));assert abs(r['t']-t)<1e-8
            base=white_onoff(data['on'],data['off']);panels=[('RAW ON/OFF only (no G)',base)]
            for method in METHODS:
                # Common absolute display scale across arms, independent of detection thresholds.
                level=np.uint8(255*(1-np.clip(data[method]/10.,0,1)));image=cv2.cvtColor(level,cv2.COLOR_GRAY2BGR)
                draw(image,r['methods'][method]['candidates'],(230,130,50));draw(image,r['methods'][method]['observation']['rotors'],(30,100,230))
                panels.append((method+' | blue candidates, orange selected',image))
            note=np.full_like(base,255);cv2.putText(note,f'{case} t={t:.2f}s',(20,45),0,1,(0,0,0),2)
            cv2.putText(note,'Monochrome maps are ON/OFF count evidence,',(20,85),0,.62,(0,0,0),1)
            cv2.putText(note,'NOT reconstructed grayscale G.',(20,115),0,.65,(0,0,0),1)
            cv2.putText(note,'Display scale:10 equivalent events -> black.',(20,155),0,.62,(0,0,0),1)
            cv2.putText(note,'All algorithms ran on full native frames.',(20,195),0,.62,(0,0,0),1)
            panels.append(('Scope',note))
            full=np.vstack([np.hstack([fit_panel(im,title) for title,im in panels[:3]]),np.hstack([fit_panel(im,title) for title,im in panels[3:]])])
            out=folder/f'comparison_{t:.2f}.png';out.write_bytes(cv2.imencode('.png',full)[1].tobytes())
            entry=dict(case=case,t=t,full=str(out.relative_to(OUT)))
            if case=='clear':
                ref=next(x for x in refs['frames'] if abs(x['t']-t)<1e-8);pts=np.array(ref['centers'])
                lo=np.maximum(np.floor(pts.min(0)-[75,30]).astype(int),0);hi=np.minimum(np.ceil(pts.max(0)+[75,80]).astype(int),[base.shape[1],base.shape[0]])
                box=[*lo,*hi];zoom=np.vstack([np.hstack([fit_panel(im,title,box) for title,im in panels[:3]]),np.hstack([fit_panel(im,title,None if title=='Scope' else box) for title,im in panels[3:]])])
                zp=folder/f'zoom_{t:.2f}.png';zp.write_bytes(cv2.imencode('.png',zoom)[1].tobytes());entry['zoom']=str(zp.relative_to(OUT))
            links.append(entry)
    save(OUT/'figure_manifest.json',links);return links

def document(summary,links):
    a=load(OUT/'analysis.json');rows=[]
    for r in a['table']:
        num=lambda v:'—' if v is None else f'{v:.2f}'
        rows.append(f"<tr><td>{r['case']}</td><td>{r['method']}</td><td>{r['detected_frames']}/{r['published_frames']}</td><td>{r['four_selected_frames']}</td><td>{r['present_detected']}/{r['present_labels']}</td><td>{num(r['mean_current_error_px'])}</td><td>{num(r['candidate_inside_mean'])}</td><td>{num(r['candidate_outside_mean'])}</td><td>{r['absent_detections']}/{r['absent_labels']}</td></tr>")
    refrows=[]
    for r in summary:
        refrows.append(f"<tr><td>{r['method']}</td><td>{r['kind']}</td><td>{r['radius_px']}</td><td>{r['matched']}/12</td><td>{r['unmatched_candidates']}</td></tr>")
    pictures=[]
    for r in links:
        pictures.append(f"<details {'open' if r['case']=='clear' and r['t']==24 else ''}><summary>{r['case']}，{r['t']:.2f}s</summary><img src='{r.get('zoom',r['full'])}'><p><a href='{r['full']}'>全视野对照</a></p></details>")
    review=OUT/'dsh_review'
    reviewnames=['review.txt','监督复核.md','result.json','scope.json','guard_audit.json']
    reviewlinks=''.join(f'<li><a href="dsh_review/{name}">{name}</a></li>' for name in reviewnames if (review/name).exists())
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第八阶段：纯ON/OFF桨盘提取对照</title><style>body{background:#f2f5f9;color:#233146;font-family:Microsoft YaHei,system-ui;line-height:1.8;margin:0}main{max-width:1200px;margin:auto;padding:28px}section,details{background:white;padding:18px;border-radius:10px;margin:18px 0}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #dce3ed;text-align:left;white-space:nowrap}th{background:#dfe8f4}.scroll{overflow:auto}img{max-width:100%}a{color:#176dab}.note{border-left:4px solid #c38728;padding:16px;background:#fff7e6}</style><main>'''
    doc+='<h1>第八阶段：纯 ON/OFF 桨盘提取小实验</h1><p>2026-10-17 · 仅原生事件计数 · 不使用重建G · 不读取/执行/修改yolov10-main</p>'
    doc+='<section><h2>同日补充：侧视退化直接发生在历史运动门控</h2><p>重放原法/邻域10ms，200个发布时刻逐项复现。原法独有的52帧全部有可行几何组合，但被历史预测中心距离门控拒绝；两法无候选帧同为14。110s两法各6个候选，邻域历史预测与可选四点均值相差323.16px，门限80px；之前选中的局部小组合已将整机尺度变为42px。</p><p>可行组合不保证真实正确，也不能直接关闭门控。当前先修整机状态保护与重新捕获，再做运动补偿；下方保留原试验结果。<a href="../stage8_side_diagnosis_20261017/侧视退化定位.md">查看完整诊断</a> · <a href="../stage8_side_diagnosis_20261017/completion.json">重放核查与分类计数</a></p></section>'
    doc+='<section><a href="阶段结论.md">先读：结论与下一步</a><p>这是候选提取与几何分组的小规模开发对照，没有训练检测器，没有重新测量预测轨迹、真实姿态或RPM精度。颜色变黑或候选增加，不自动等于识别更准。</p></section>'
    doc+='<section><h2>本轮结论：有局部帮助，不能直接全局替换</h2><p>六段RAW共19.5秒、975个发布时刻。清晰段：原10ms检出187/200，邻域10ms为189/200；过去30ms两组均为200/200。侧视段：原10ms为167/200，严格30ms为174/200，邻域10ms降至121/200。FRED两段未改善；长窗不能通用。</p><p>三个清晰帧AI近似参考中，最终选中中心的8px匹配由9/12到11/12；12px下两者均12/12。30ms在8px下只2/12与1/12。增益主要是清晰盘局部形状/中心，长窗有收缩、拖影和时刻偏差的风险。下一步优先验证“保持10ms细节＋用过去运动补偿累积”，辅以门控与候选/分组的分层诊断；本轮尚未执行该改动。</p></section>'
    doc+='<h2>固定方法与复用</h2><p>strict10是原10ms逐像素双极min；neighbor10先对ON/OFF分别做3×3、σ0.8高斯，再取min；strict30、neighbor30使用过去30ms计数并除3，保持10ms等效计数率。所有方法复用原阈值、形态学、椭圆/线段、几何分组与短时关联；原src不改。30ms启动不填未来数据。保留原生像素总计数门控，所以完全空像素即使邻域有信号也可能被门控清除，这是本轮保守定义的限制。</p>'
    doc+='<div class="note">每一处理帧都把strict10与未修改PaperPerception对照，必须完全一致才接受结果。全图原生分辨率推理，标签只用于事后审计；清晰图的放大区域由AI参考点决定，只用于展示。</div>'
    doc+='<h2>全片/稀疏标注对照</h2><p>“四个”仅指算法选中四个候选。框内/框外候选不是逐桨真值；候选函数含薄线补充分支，返回数可能含重复。“当前误差”为整机中心对框中心的误差，不是未来预测误差。</p><div class="scroll"><table><tr><th>片段</th><th>方法</th><th>检出帧</th><th>选4候选帧</th><th>有机标签检出</th><th>当前误差px</th><th>框内候选均值</th><th>框外候选均值</th><th>无机标签误发布</th></tr>'+''.join(rows)+'</table></div>'
    doc+='<h2>三个固定清晰帧的AI参考核对</h2><p>先只查看原始ON/OFF白底图、再记录12个可见盘的近似中心；未显示检测结果来放置参考点。位置主观不确定度约5px，不是人工GT或实际转轴真值。8/12/20px门槛全部报告，不选最好看的半径。匹配一对一；未匹配候选含重复，不能直接称全图误检。</p><div class="scroll"><table><tr><th>方法</th><th>候选/最终选中</th><th>半径px</th><th>匹配近似盘中心</th><th>未匹配候选</th></tr>'+''.join(refrows)+'</table></div>'
    doc+='<section><h2>共同样本与耗时复核</h2><p>同一组稀疏整机框：清晰段共同5点，原10ms→严格30ms为8.14→9.17px；侧视共同5点14.59→10.32px，但主要由一处34.53→6.03px拉低均值，其余4点均值反而9.61→11.39px。不能称稳定侧视提升。</p><p>1962处理帧逐帧基线完全一致。四组加原检测器、解码与落盘合计260.21s，输入19.5s；不是单模型测速。清晰段邻域10ms检测部分中位21.40ms、P95为24.74ms，尚不含共享解码/计数，未证明100Hz实时。</p><a href="independent_metrics.json">共同样本逐点与耗时</a> · <a href="独立指标复核.md">独立复核</a></section>'
    doc+='<h2>实际提取结果</h2><p>蓝色：几何候选；橙色：最终选中。黑白单通道图是ON/OFF计数强度可视化，并非灰度重建G。所有方法用相同显示计数尺度，归一化不影响检测阈值。</p>'+''.join(pictures)
    doc+='<h2>已有工作与本轮边界</h2><p><a href="https://github.com/prgumd/EVPropNet">EVPropNet</a>已研究事件桨叶检测，<a href="https://github.com/radimspetlik/effqpdet">effqpdet</a>已有高分辨率快速桨叶检测；本轮复用现有OpenCV与项目几何模块做特征消融，不宣称这一组合有新颖性，也没有执行上述作者模型。窗口与计数来自<a href="https://docs.prophesee.ai/stable/concepts.html">原生事件概念</a>；ON/OFF不是目标绝对亮度，黑色不能直接换算转速。</p>'
    doc+='<h2>DSH / DeepSeek审查</h2><p>已实际成功调用DSH→deepseek-official/deepseek-flash一次作受限文本审查，输出479个汉字；仅发送3845字符代码/协议片段，工具数为零。没有发送RAW、图片或密钥。模型三项误判已在监督复核中指出，有效建议用于结果解释；实验和方向由本项目监督核实。</p><ul>'+reviewlinks+'</ul>'
    doc+='<p><a href="protocol.json">冻结协议</a> · <a href="analysis.json">逐项诊断</a> · <a href="visual_reference_score.json">AI参考匹配</a> · <a href="validation.json">验证</a></p></main></html>'
    (OUT/'第八阶段纯ONOFF桨盘提取报告.html').write_text(doc,encoding='utf-8')

def validate():
    verify();meta=[]
    for case,cfg in CASES.items():
        m=load(OUT/case/'completion.json');assert not m['G_used'] and not m['yolov10_main_accessed'] and not m['labels_in_inference']
        assert m['published']==round(cfg['duration']/.02)
        for line in (OUT/case/'observations.jsonl').read_text().splitlines():
            r=json.loads(line)
            for method,q in r['methods'].items():
                assert q['observation']['t']==r['t']
                if q['observation']['detected']:assert np.isfinite(q['observation']['center']).all()
        meta.append(dict(case=case,published=m['published'],baseline_exact=m['baseline_exact_frames']))
    save(OUT/'validation.json',dict(source_hashes_unchanged=True,original_baseline_equal_every_processed_frame=True,
        G_used=False,yolov10_main_written=False,cases=meta))

if __name__=='__main__':
    verify();s=reference_score();links=figures();validate();document(s,links);print('ON/OFF report complete',flush=True)
