"""Frozen small RAW threshold/line-midpoint pilot for the user's side-view idea."""
from pathlib import Path
import argparse,datetime,hashlib,html,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage58'),str(ROOT/'research/stage57'),str(ROOT/'research/stage49')]
import study58 as previous
import rigid49 as rigid
import line59 as line
from gating57 import CandidateFrontend
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=ROOT.parent/'output/stage59_side_lines_20261021'
CASES=dict(clear6=dict(start=22.,duration=1.,raw=r'E:\dataset\train\6\media\event.raw'),
           side6=dict(start=108.,duration=2.,raw=r'E:\dataset\train\6\media\event.raw'),
           train3=dict(start=25.,duration=1.,raw=r'E:\dataset\train\3\aligned\event.raw'),
           negative5=dict(start=0.,duration=.5,raw=r'E:\dataset\train\5\media\event.raw'))
save=previous.save;load=previous.load;sha=previous.sha;rows=previous.rows

def sources():
    r=previous.sources()
    for p in [Path(__file__),Path(__file__).with_name('line59.py')]:r[p.relative_to(ROOT).as_posix()]=sha(p)
    return r

def inputs():
    return {f'{c}/{n}':sha(previous.previous.OUT/c/n) for c in CASES for n in ['rotor_ticks.jsonl','observations_forecasts.jsonl']}

def freeze():
    previous.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),input_hashes=inputs(),cases=CASES,parameters=line.PARAMS,
       design='Actual native10ms RAW, original57 CandidateFrontend exactly replayed as control. Separately compute the identical original8 ON/OFF density and global quantile; compare no-opening elongated contours at thresholds1/1.5/2 and saturation2. Body ROI gates complete candidate centers only, using cached57 prior NN hints known before packet start. No tuning to outcomes, no forced four-rotor identity. Labels and future forecast fields are not used.',
       synthetic='Known horizontal100px line, complete/one-side30px loss/nonuniform density/merged two rotors/single outlier. Also known pinhole projected circles at tilts0/30/60/75/82/87/89/90. Same existing geometry._line estimator; physical projected hub is evaluation only.',
       real_metrics='Line candidates/support retention/threshold sensitivity and old current-geometry exactness only. No real hub/pose truth; no new forecast scoring or production promotion.',
       prior_art=['https://arxiv.org/html/2510.06829v1','https://github.com/event-driven-robotics/RT-EvLDT','https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2018.00004/full','https://rpg.ifi.uzh.ch/docs/EBCCSP16_Braendli.pdf'],
       actual_reuse='Unchanged local geometry._line and57/9/8 frontend, existing OpenCV operations. RT-EvLDT/ELiSeD researched, not compiled/executed. Illustrative axis_weight is not used to claim a forecast result.'))
    print('59 frozen: distinct line observations, no new flight-dynamics claim',flush=True)

def verify():
    previous.verify();p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['input_hashes']==inputs()

def synthetic():
    verify();assert not (OUT/'synthetic.json').exists()
    samples=[]
    for name,left,right,gradient,outlier in [('complete',-50.,50.,False,False),('left30px_missing',-20.,50.,False,False),('unequal_event_density',-50.,50.,True,False),('two_rotors_merged',-50.,110.,False,False),('one_collinear_outlier',-50.,50.,False,True)]:
        image=np.zeros((160,260),np.float32);xs=np.arange(int(100+left),int(100+right)+1)
        image[79:82,xs]=np.linspace(2.,20.,len(xs))[None,:] if gradient else 10.
        if outlier:image[80,180]=10.
        for threshold in [2.,6.,12.]:
            yy,xx=np.nonzero(image>=threshold)
            if not len(xx):m=None
            else:m=line.measure(np.c_[xx,yy],250.)
            samples.append(dict(scenario=name,threshold=threshold,measurement=m,
                midpoint_error_px=float(np.linalg.norm(np.array(m['center'])-[100,80])) if m else None,
                robust_midpoint_error_px=float(np.linalg.norm(np.array(m['robust_center'])-[100,80])) if m else None,
                interpretation='First physical rotor hub at100,80; in merged case second rotor occupies110..210, so a single full component cannot identify either rotor hub.'))
    projection=[]
    phase=np.linspace(0,2*np.pi,400,endpoint=False);circle=np.c_[np.cos(phase),np.sin(phase)]*.16
    for a in [0,30,60,75,82,87,89,90]:
        H=rigid.homography(rigid.rotation(a,0,12),[0,0,3])
        for index,hub in enumerate(rigid.TEMPLATE):
            points=rigid.project(H,circle+hub);truth=rigid.project(H,[hub])[0];m=line.measure(points,640.)
            projection.append(dict(tilt_deg=a,rotor=index,line_eligible=m is not None,
               midpoint_error_px=float(np.linalg.norm(np.asarray(m['center'])-truth)) if m else None))
    save(OUT/'synthetic.json',dict(line_cases=samples,pinhole_projection=projection))
    print('59 known-truth controls complete',flush=True)

def real():
    verify();assert not (OUT/'real.json').exists();all_rows=[]
    for case,cfg in CASES.items():
        parent=rows(previous.previous.OUT/case/'observations_forecasts.jsonl');hints={}
        for r in parent:
            hint=r['body_hint_sent']
            if hint is not None:assert round(hint['t']*1e6)<=round((r['t']-.02)*1e6)
            for t in [r['t']-.01,r['t']]:hints[round(t*1e6)]=hint
        reference=rows(previous.previous.OUT/case/'rotor_ticks.jsonl')
        reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();front=CandidateFrontend(w,h,'neighbor10');result=[]
        chosen={int(cfg['duration']*100)*i//4-1 for i in [1,2,3]}
        for index,(t,events) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
            start=time.perf_counter();hint=hints[round(t*1e6)];d=front.set_hint(events,hint,t,'body')
            obs,_=front.detect(events,t);expected=reference[index]
            assert round(t,8)==round(expected['t'],8) and obs==expected['observation'],(case,index,'old57 observation changed')
            prepared=front.prepared;feature=front.feature(prepared['on'],prepared['off']);feature[prepared['on']+prepared['off']<front.cfg.min_pixel_count]=0
            b=front.cfg.spatial_bin;hs,ws=h//b,w//b
            density=cv2.GaussianBlur(feature[:hs*b,:ws*b].reshape(hs,b,ws,b).sum(axis=(1,3)),(3,3),.8)
            assert b==1,'Pilot records native pixel line coordinates only'
            positive=density[density>0];threshold=max(2.,float(np.quantile(positive,front.cfg.density_quantile))) if len(positive) else 2.
            roi=d['roi'];variants={};masks={}
            for name,multiplier in [('base',1.),('hard1p5',1.5),('hard2',2.),('cap2',1.)]:
                source=np.minimum(density,2*threshold) if name=='cap2' else density
                candidates,mask=line.candidates(source,threshold*multiplier,roi,w*.5)
                variants[name]=dict(candidates=candidates,active_pixels=int(np.count_nonzero(mask)));masks[name]=mask
            assert np.array_equal(masks['base'],masks['cap2'])
            for name in ['hard1p5','hard2']:
                for m in variants[name]['candidates']:
                    old=variants['base']['candidates']
                    # Proximity match is sensitivity only, not certified identity.
                    if old:
                        old=min(old,key=lambda r:np.linalg.norm(np.array(r['center'])-m['center']))
                        distance=float(np.linalg.norm(np.array(old['center'])-m['center']))
                        if distance<=max(12.,old['major']*.5):m.update(nearest_base_midpoint_shift_px=distance,length_ratio_to_nearest_base=m['major']/old['major'])
            record=dict(case=case,t=t,raw_events=len(events),roi=roi,threshold=threshold,variants=variants,
                old57_geometry_exact=True,cap_binary_support_exact=True,elapsed_ms=(time.perf_counter()-start)*1000)
            result.append(record)
            if index in chosen:
                # Fixed source-time samples; crop by prior NN box only for display.
                box=roi if roi else [0,0,w,h];panels=[]
                for name in ['base','hard1p5','hard2']:
                    panel=cv2.cvtColor(masks[name],cv2.COLOR_GRAY2BGR)
                    for m in variants[name]['candidates']:
                        e=np.rint(m['endpoints']).astype(int);cv2.line(panel,tuple(e[0]),tuple(e[1]),(0,180,255),2)
                        cv2.circle(panel,tuple(np.rint(m['center']).astype(int)),4,(255,0,255),-1)
                    x1,y1,x2,y2=box;panel=panel[y1:y2,x1:x2];panel=cv2.resize(panel,(600,320))
                    cv2.putText(panel,f'{case} t={t:.2f} {name} lines={len(variants[name]["candidates"])}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.52,(100,255,100),1)
                    panels.append(panel)
                cv2.imencode('.png',np.hstack(panels))[1].tofile(str(OUT/f'{case}_{index:04d}.png'))
        assert len(result)==round(cfg['duration']*100)
        save(OUT/f'{case}.json',dict(records=result,current_geometry_exact57=True,labels_used=False));all_rows.extend(result)
        print(f'{case}: {len(result)} native RAW ticks exact57; threshold pilot complete',flush=True)
    save(OUT/'real.json',dict(native10=len(all_rows),cases=list(CASES),all_geometry_exact57=True,cap_binary_support_exact=True,production_changed=False))

def report():
    verify();s=load(OUT/'synthetic.json');real=load(OUT/'real.json');summary=[]
    for case in CASES:
        rr=load(OUT/f'{case}.json')['records']
        for name in ['base','hard1p5','hard2','cap2']:
            found=[r for x in rr for r in x['variants'][name]['candidates']]
            shifts=[r['nearest_base_midpoint_shift_px'] for r in found if 'nearest_base_midpoint_shift_px' in r]
            summary.append(dict(case=case,variant=name,ticks=len(rr),ticks_with_lines=sum(bool(x['variants'][name]['candidates']) for x in rr),
                total_line_candidates=len(found),median_active_pixels=float(np.median([x['variants'][name]['active_pixels'] for x in rr])),
                matched_midpoint_shift_median_px=float(np.median(shifts)) if shifts else None,
                matched_midpoint_shift_p95_px=float(np.quantile(shifts,.95)) if shifts else None))
    save(OUT/'analysis.json',dict(real=summary,limits='Candidates are not rotor IDs. Midpoint shifts are parameter sensitivity, not real error. Original geometry replayed exactly, no new future results or realtime claim.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name in ['complete','left30px_missing','unequal_event_density','two_rotors_merged','one_collinear_outlier']:
        rr=[r for r in s['line_cases'] if r['scenario']==name]
        axes[0].plot([r['threshold'] for r in rr],[r['midpoint_error_px'] if r['midpoint_error_px'] is not None else np.nan for r in rr],'o-',label=name)
    axes[0].set(xlabel='Synthetic event-density threshold',ylabel='Known hub error (px)');axes[0].legend(fontsize=7)
    for case in CASES:
        rr=[r for r in summary if r['case']==case and r['variant']!='cap2']
        axes[1].plot([1,1.5,2],[r['ticks_with_lines']/r['ticks'] for r in rr],'o-',label=case)
    axes[1].set(xlabel='Real global-threshold multiplier',ylabel='Fraction with any line candidates',ylim=(-.02,1.02));axes[1].legend(fontsize=8)
    fig.suptitle('Line midpoint pilot: known synthetic hub errors vs unlabeled real line availability',fontsize=11)
    fig.savefig(OUT/'line_midpoint_controls.png',dpi=160);plt.close(fig)
    def table(rr):
        if not rr:return ''
        cols=list(rr[0]);return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[c],3) if isinstance(r[c],float) else r[c]))+'</td>' for c in cols)+'</tr>' for r in rr)+'</table>'
    simple=[{k:r[k] for k in ['scenario','threshold','midpoint_error_px','robust_midpoint_error_px']} for r in s['line_cases']]
    images=''.join('<p>固定源时刻，框仅用于显示裁剪；绿色文字为线段候选数，紫点是候选线段中点。</p><img src="'+p.name+'">' for p in sorted(OUT.glob('*_0*.png')))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>59 侧视高事件线与桨心</title><style>body{max-width:1500px;margin:30px auto;color:#233e4c;font:16px/1.8 Arial}img{width:100%}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd8df;padding:6px}</style><h1>59：侧视高事件线取中点，哪些条件成立</h1><p>可作为侧视观测分支：完整、对称、可分离的单个桨盘接近侧视时，轴线比扁椭圆容易拟合。但可见线段的中点不自动等于真实轴心。单侧缺失、亮度不对称、共线异常点与多个桨盘粘连都会破坏它。透视下投影中点也不严格保持真实三维中点。</p><p>先查RT-EvLDT、2018低延迟线跟踪和ELiSeD。此次实际复用本项目未改动geometry._line及57/9/8原检测器，无新依赖；没有声称已复现RT-EvLDT。现有主路径的line模式仍由fitEllipse给中心，通用_line已有端点中点实现，故无需从零写线段算法。</p><img src="line_midpoint_controls.png"><h2>有独立真值的受控反例</h2>'''+table(simple)+'''<h2>实际自采RAW小对照</h2><p>450个原生10ms当前时刻：clear6 1秒、side6 2秒、train3 1秒、empty5 0.5秒，原57当前几何逐时刻精确。仅候选中心按过去NN框筛选，全图阈值统计不裁剪；阈值倍率1/1.5/2事前固定，不挑最优值。cap2饱和图与base二值支持逐像素相同；压低显示亮度不会凭空增加边界信息。线段候选数量不是桨心检测准确率，最近邻偏移只是阈值敏感性，不是实采真实误差。</p>'''+table(summary)+'''<h2>应如何融合</h2><p>椭圆与线段应更新同一个整机/桨槽运动状态。椭圆分支提供中心、轴比和法向线索；侧视分支提供无向轴与有条件的中点，沿线中心不确定度随端点缺失和阈值敏感性增大。以轴比、拟合质量、跨阈值稳定性、周期性、身份关联和过去运动残差计算观测权重；同一批事件产生的两观测相关，不能当独立证据重复加权。用双角平均无向轴，避免正负法线抵消，再平滑权重和增加切换滞回。没有可信端点或单桨身份时，仅更新轴，不强行更新四个桨心/转速/推力。</p><p>当前只完成前端小试验，未接入新预测、未证明未来误差改善，也不声称单条侧视线能恢复完整3D姿态或推力正负。下一步优先解决完整端点和多个桨盘粘连的身份验证，再做固定同位置的历史/椭圆/线段/融合未来对照。</p>'''+images+'''<p><a href="protocol.json">冻结协议</a> · <a href="synthetic.json">全部受控试验</a> · <a href="analysis.json">实际统计</a> · <a href="https://github.com/event-driven-robotics/RT-EvLDT">RT-EvLDT作者库</a> · <a href="https://arxiv.org/html/2510.06829v1">RT-EvLDT论文</a> · <a href="https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2018.00004/full">低延迟事件线跟踪</a></p></html>'''
    (OUT/'第五十九阶段侧视线段中点小试验.html').write_text(page,encoding='utf-8')
    save(OUT/'completion.json',dict(source_hashes_unchanged=sources()==load(OUT/'protocol.json')['source_hashes'],native10=real['native10'],geometry_exact57=True,new_forecasts=False))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','synthetic','real','report']);globals()[p.parse_args().action]()
