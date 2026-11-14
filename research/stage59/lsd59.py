"""Supplement: actual existing OpenCV LSD, rather than only thin components."""
from pathlib import Path
import argparse,datetime,html,json,sys
import cv2,numpy as np
import study59 as current
from polarity import counts,PolarityFrontend
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=current.OUT;PROTOCOL=OUT/'lsd_supplement.json'

def freeze():
    current.verify();assert not PROTOCOL.exists()
    current.save(PROTOCOL,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=current.sha(Path(__file__)),
        design='Post-pilot diagnostic: same450 native RAW ticks and frozen threshold multipliers, actual OpenCV LSD_REFINE_STD already used in stage6. Density gray=min(density/(2*base_threshold),1)*255; below each threshold zeroed. No binary-component thinness requirement, no morphology. Existing12px line minimum, max width/2, candidate midpoint screened by original past ROI. Multiple edge segments are not rotor identities or hubs.',
        selection='Same complete intervals, not selected successful frames. Fixed25/50/75 percent images. No RPM, labels, forecast input, new future accuracy or new algorithm claim.'))

def verify():
    current.verify();assert current.load(PROTOCOL)['helper_sha256']==current.sha(Path(__file__))

def run():
    verify();assert not (OUT/'lsd_results.json').exists();result=[];detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    for case,cfg in current.CASES.items():
        reference=current.load(OUT/f'{case}.json')['records'];reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();front=PolarityFrontend(w,h,'neighbor10');records=[]
        chosen={int(cfg['duration']*100)*i//4-1 for i in [1,2,3]}
        for index,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
            old=reference[index];assert round(t,8)==round(old['t'],8)
            c=counts(e,w,h);a=front.feature(c['on'],c['off']);a[c['total']<front.cfg.min_pixel_count]=0
            density=cv2.GaussianBlur(a,(3,3),.8);positive=density[density>0]
            threshold=max(2.,float(np.quantile(positive,front.cfg.density_quantile))) if len(positive) else 2.
            assert threshold==old['threshold'];panels=[];variants={};roi=old['roi']
            for name,mult in [('base',1.),('hard1p5',1.5),('hard2',2.)]:
                gray=np.uint8(np.clip(density/(2*threshold),0,1)*255);gray[density<threshold*mult]=0
                lines=detector.detect(gray)[0];found=[]
                for p in ([] if lines is None else lines.reshape(-1,4)):
                    length=float(np.linalg.norm(p[2:]-p[:2]));center=(p[:2]+p[2:])/2
                    if not 12<=length<=w*.5:continue
                    if roi is not None and not (roi[0]<=center[0]<roi[2] and roi[1]<=center[1]<roi[3]):continue
                    found.append(dict(endpoints=p.reshape(2,2).tolist(),midpoint=center.tolist(),length_px=length,verified_hub=False))
                variants[name]=found
                if index in chosen:
                    panel=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
                    for m in found:
                        p=np.rint(m['endpoints']).astype(int);cv2.line(panel,tuple(p[0]),tuple(p[1]),(0,180,255),1)
                        cv2.circle(panel,tuple(np.rint(m['midpoint']).astype(int)),2,(255,0,255),-1)
                    box=roi if roi else [0,0,w,h];x1,y1,x2,y2=box;panel=cv2.resize(panel[y1:y2,x1:x2],(600,320))
                    cv2.putText(panel,f'LSD {case} t={t:.2f} {name} edges={len(found)}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.47,(100,255,100),1);panels.append(panel)
            records.append(dict(t=t,variants=variants))
            if panels:cv2.imencode('.png',np.hstack(panels))[1].tofile(str(OUT/f'lsd_{case}_{index:04d}.png'))
        assert len(records)==len(reference)
        current.save(OUT/f'lsd_{case}.json',dict(records=records));result.extend(dict(case=case,variant=name,ticks=len(records),ticks_with_edge_lines=sum(bool(r['variants'][name]) for r in records),total_edge_lines=sum(len(r['variants'][name]) for r in records)) for name in ['base','hard1p5','hard2'])
        print(case+' LSD complete: '+str(len(records)),flush=True)
    current.save(OUT/'lsd_results.json',dict(table=result,native10=450,threshold_exact59=True,independent_rotor_hub_truth=False))

def report():
    verify();r=current.load(OUT/'lsd_results.json');cols=list(r['table'][0])
    table='<table><tr>'+''.join('<th>'+c+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(row[c]))+'</td>' for c in cols)+'</tr>' for row in r['table'])+'</table>'
    imgs=''.join('<img src="'+p.name+'">' for p in sorted(OUT.glob('lsd_*_0*.png')))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>59 LSD补充</title><style>body{max-width:1500px;margin:30px auto;font:16px/1.8 Arial;color:#233e4c}img{width:100%}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccd8df}</style><h1>补查：从粘连轮廓内部直接提取线段</h1><p>为避免仅用连通轮廓PCA而遗漏内部直边，按原第6阶段实际调用OpenCV LSD_REFINE_STD。重读同样450个当前RAW时刻，不额外挑好片段；全图密度阈值逐时刻精确原59，比较同样1/1.5/2倍率。灰度截顶2T后，低于各阈值的密度清零。这是已有线段检测器的小诊断，不是RT-EvLDT的复现，也不是新的无人机预测结果。</p>'''+table+'''<p>LSD可在宽连通块内部找到直边，但返回的是边缘线，可能同时包括一个桨盘两条边、机臂和碎片。紫色中点没有单桨身份/完整直径保证，候选变多不能当作真实桨心变准。接下来需要端点完整性、相邻双边的中心线合并、桨槽关联、周期与跨阈值/时间稳定性验证；没有这些证据时只更新无向轴。</p>'''+imgs+'''<p><a href="第五十九阶段侧视线段中点小试验.html">主试验</a> · <a href="lsd_supplement.json">补充冻结协议</a> · <a href="lsd_results.json">完整统计</a></p></html>'''
    (OUT/'侧视LSD线段补充诊断.html').write_text(page,encoding='utf-8');print(json.dumps(r['table'],ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','report']);globals()[p.parse_args().action]()
