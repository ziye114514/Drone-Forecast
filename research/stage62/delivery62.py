"""Measured outcomes, all sparse targets, actual RAW videos and support audit."""
from collections import Counter
import html
import cv2, numpy as np
import raw62 as current
base=current.current;OUT=base.OUT;load=base.load;rows=base.rows;save=base.save


def table(rr):
    if not rr:return '<p>无记录。</p>'
    cols=list(rr[0])
    return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[k],4) if isinstance(r[k],float) else r[k]))+'</td>' for k in cols)+'</tr>' for r in rr)+'</table></div>'


def run():
    current.verify();raw=load(current.OUT/'raw_prefix_checks.json')
    assert raw['full15_and_lifecycle_exact'] and raw['forecast_scores_reusable_from62_replay']
    scores=load(OUT/'analysis.json');checks=load(OUT/'score_checks.json')
    assert checks['rows']==540 and checks['baseline_rows_exact61']==180
    learning=load(OUT/'replay_checks.json')['table'];evidence=[];usage=[]
    saved={};inputs={}
    for case in base.CASES:
        saved[case]=rows(OUT/case/'forecasts.jsonl')
        inputs[case]=rows(base.REFERENCE/f'off_{case}/observations_forecasts.jsonl')
        source=Counter();activated=Counter();max_angle_step=0.;last=None
        for r,s in zip(saved[case],inputs[case]):
            observed=[v for v in s['sensing']['rotors'] if v.get('observed_now') and v.get('geometry_quality',0)>.15]
            category=('no_current_axis' if not observed else 'line_only' if all(v.get('axis_only') for v in observed) else 'ellipse_or_arc')
            source[category]+=1
            q=r['frontends']['axis_learn'];d=q['feature_diagnostic']['models']['adaptive_axis_only']
            if d['active']:activated[category]+=1
        evidence.append(dict(case=case,parent=len(saved[case]),current_axis_source=dict(source),
            active_axis_source=dict(activated),
            current_line_only_learning_possible=False,
            explanation='Original line geometry cap .4 is below unchanged RLS admission .45. '
                        'Line may steer rollout but does not by itself launch learning. '
                        'Learning counts include delayed previous features, not necessarily current line.'))
        if case=='negative5':continue
        anchor={round(r['t'],8):r for r in saved[case]}
        for method in base.METHODS:
            selected=[r for r in scores['details'] if r['case']==case and r['frontend']==method and r['model']=='adaptive_axis_only' and r['nominal_h']==.4 and r['target_state']=='present' and r['forecast_valid']]
            active=updated=equal_cv=0
            for d in selected:
                q=anchor[round(d['anchor_t'],8)]['frontends'][method]
                m=q['feature_diagnostic']['models']['adaptive_axis_only']
                active+=m['active'];updated+=m['updates']>0
                cv=next(c for c in scores['details'] if c['case']==case and c['frontend']==method and c['model']=='cv' and c['nominal_h']==.4 and c['target_t']==d['target_t'])
                equal_cv+=np.allclose(cv['predicted'],d['predicted'],rtol=0,atol=1e-9)
            usage.append(dict(case=case,method=method,valid_targets=len(selected),active_anchors=active,
                              anchors_after_RLS_updates=updated,equal_CV_targets=equal_cv))
    comparison=[r for r in scores['table'] if r['case']!='negative5' and r['nominal_h']==.4 and r['model'] in ['cv','trusted_acc','adaptive_axis_only','adaptive_axis_freq','adaptive_all_mix']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['Shared gate','Axis gate','Axis gate+learn','Pure history'];colors=['#f4a261','#e9c46a','#7b2cbf','#2a9d8f']
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    for ax,case in zip(axes,base.CASES[:3]):
        selected=[next(r for r in comparison if r['case']==case and r['frontend']==m and r['model']=='adaptive_axis_only') for m in base.METHODS]
        selected.append(next(r for r in comparison if r['case']==case and r['frontend']=='shared' and r['model']=='trusted_acc'))
        values=[r['mean_fde_px'] for r in selected];ax.bar(names,values,color=colors)
        for j,value in enumerate(values):ax.text(j,value+max(values)*.025,f'{value:.2f}',ha='center',fontsize=9)
        ax.set(title=case+f" ({selected[0]['valid_forecasts']}/{selected[0]['present_targets']} targets)",ylabel='0.4s sparse endpoint error (px)',ylim=(0,max(values)*1.2))
        ax.tick_params(axis='x',labelrotation=20)
    fig.suptitle('Same CV fallback, positions, sparse targets and unchanged RPM experts',fontsize=11)
    fig.savefig(OUT/'axis_future_comparison62.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,case in zip(axes,base.CASES[:3]):
        for method,model,label,color in [('shared','adaptive_axis_only','Shared axis','#f4a261'),('axis_learn','adaptive_axis_only','Independent axis+learn','#7b2cbf'),('shared','trusted_acc','Pure history','#2a9d8f')]:
            d=[r for r in scores['details'] if r['case']==case and r['frontend']==method and r['model']==model and r['nominal_h']==.4 and r['target_state']=='present' and r['forecast_valid']]
            ax.plot([r['target_t'] for r in d],[r['error_px'] for r in d],'o-',label=label,color=color,ms=4)
        ax.set(title=case,xlabel='Future target source time (s)',ylabel='Endpoint error (px)');ax.legend(fontsize=7)
    fig.suptitle('Every valid0.4s target shown; missing anchors retained in table denominators',fontsize=11)
    fig.savefig(OUT/'all_target_errors62.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(13,8),layout='constrained')
    for ax,case in zip(axes,base.CASES[:3]):
        rr=saved[case];tt=[r['t'] for r in rr]
        for method,color in [('shared','#f4a261'),('axis_learn','#7b2cbf')]:
            ax.step(tt,[int(r['frontends'][method]['feature_diagnostic']['models']['adaptive_axis_only']['active']) for r in rr],where='post',label=method+' active',color=color,alpha=.8)
        ax.plot(tt,[r['axis_audit']['axis_learn']['axis_quality'] for r in rr],color='#277da1',label='Stored geometry quality',lw=1)
        ax.set(title=case,ylim=(-.08,1.15),xlabel='Source time(s)');ax.legend(fontsize=8,ncol=3)
    fig.suptitle('Availability is internal evidence, not attitude accuracy or confidence calibration')
    fig.savefig(OUT/'axis_admission_timeline62.png',dpi=150);plt.close(fig)
    fixed=[]
    for case,t in [('clear6',23.),('side6',110.5),('train3',28.)]:
        cfg=current.CASES[case];index=round((t-cfg['start'])*50)-1
        cap=cv2.VideoCapture(str(current.OUT/case/'forecasts.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();cap.release();assert ok
        r=saved[case][index];assert abs(r['t']-t)<1e-8
        for method,model,color in [('shared','adaptive_axis_only',(50,140,245)),('axis_learn','adaptive_axis_only',(190,60,150)),('shared','trusted_acc',(70,190,60))]:
            q=r['frontends'][method]
            if not q['valid']:continue
            pp=[q['coefficients'][model]['position']]+[q['predictions'][model][str(h)] for h in [.1,.2,.4]]
            points=np.rint(pp).astype(np.int32)
            for a,b in zip(points,points[1:]):cv2.line(frame,tuple(a),tuple(b),color,2,cv2.LINE_AA)
            cv2.circle(frame,tuple(points[-1]),5,color,2)
        cv2.rectangle(frame,(0,0),(1280,55),(22,28,35),-1)
        cv2.putText(frame,f'{case} t={t:.2f}s | orange=old axis purple=independent axis green=history',(12,22),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'Current observations and 0.1/0.2/0.4s forecasts; no true RPM/attitude labels',(12,45),cv2.FONT_HERSHEY_SIMPLEX,.5,(225,225,225),1,cv2.LINE_AA)
        name=f'fixed_{case}_{t:.2f}.png';cv2.imencode('.png',frame)[1].tofile(str(OUT/name));fixed.append(dict(case=case,t=t,path=name))
    compact=[{k:r[k] for k in ['case','frontend','model','present_targets','valid_forecasts','mean_fde_px','joint_success']} for r in comparison]
    stats=[]
    for r in learning:
        for method,d in r['methods'].items():stats.append(dict(case=r['case'],method=method,**d))
    payload=dict(helper_sha256=base.sha(__file__),protocol_sha256=base.sha(OUT/'protocol.json'),raw_protocol_sha256=base.sha(current.OUT/'protocol.json'),
        raw_checks=raw,score_checks=checks,comparison04=compact,learning=stats,axis_support=evidence,scored_axis_usage04=usage,
        fixed_source_times=fixed,default_promoted=False,novel_algorithm_claim=False,
        conclusion='Axis-independent admission works in actual RAW and delayed learning. '
                   'Side0.4 axis-only improves122.894->113.917px, clear15.736->14.004px, '
                   'train3 worsens136.693->138.262px. Side/train3 pure history remains better. '
                   'No new detection/rotor extraction/RPM accuracy result, no realtime claim.',
        next='Inspect geometry quality against temporal support before using new regressors. '
             'Reuse6 body_measurement for true side-view direction observation only if '
             'existing rotor-derived axis coverage insufficient; never require four centers '
             'or fabricate them. Then test local polarity periodicity without complete ellipse '
             'separately, preserving ambiguity and unknown hidden rotor thrust.')
    save(OUT/'delivery62.json',payload)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>62：方向与转速独立使用</title><style>body{max-width:1400px;margin:28px auto;padding:0 22px;font:16px/1.8 system-ui;color:#243c49;background:#f7fafc}h1,h2{color:#172f3d}p{max-width:1200px}.lead{background:#eaf1fa;border-left:5px solid #7b2cbf;padding:16px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;background:white}th,td{border:1px solid #cad6df;padding:6px;vertical-align:top}img,video{max-width:100%;height:auto}a{color:#2457a3}code{background:#e9edf2;padding:3px}</style><h1>62：方向信息不再等待四桨转速齐全</h1><div class="lead"><b>实际RAW接通，侧视方向分支小幅改善，尚无全面优势。</b><p>只改变轴向分支的使用条件，保留相同位置、检测、桨叶几何、全部转速分支和CV回退。侧视方向可用84→175/250，在线更新10→142；0.4秒轴向误差122.89→113.92px，仍差于纯历史81.91px。3号136.69→138.26px变差。不能把更新次数增多等同于预测准确，实验默认未替换。</p></div><h2>做了什么，哪些还没有做</h2><p>shared：原共同频率准入；axis_gate：只让几何方向独立参与预测；axis_learn：方向预测和历史回归样本都按独立几何准入。频率分支、六历史、纯历史混合保持精确旧值；轴向ratio固定1是有效恒推力假设，不是新测得RPM。保留原0.15预测门槛、0.45学习门槛、0.16秒方向时限、0.6秒预热、0.2秒延迟反馈与参数约束。新回归样本必须有当下几何支持和整机位置，过期方向不会持续启动学习。</p><p>桨心本来就由椭圆/残弧几何计算，本轮没有更换检测器，也没有提高四桨提取率。尚未实现不依赖现有桨盘轨迹的整机线方向观测器，尚未实现四桨重叠时的多频率解混。线观测质量上限0.4低于原0.45学习门槛，纯线可以约束方向，但本轮不会独自启动参数学习；这是后续需要单独验证的环节，不能偷偷抬高置信度。</p><h2>完整性与因果控制</h2><p>缓存700个当前输入，三路共2100步。原shared完整15专家与生命周期逐项精确61；全部方法六历史、三个转速专家、历史混合精确旧值。独立重建的225+60时刻缓存前缀通过。另实际运行RAW完整14秒、共享门槛0.8秒控制、侧视4.5秒与3号1.2秒独立RAW前缀：'''+html.escape(str(raw['totals']))+'''；全部当前输入精确61，完整预测与生命周期精确缓存62，所有保存视频逐帧解码验证。推理入口只读RAW与固定本地权重，不读取旧缓存或标签。</p><p>540行未来评分，原基线180行精确61，有效目标覆盖相同。预测保存后才评分。原评分入口因协议cases为列表而报错，另加score62.py只映射原冻结时间区间；原协议和预测源未改，失败及补充适配说明保留。新推理代码冻结前的缩进检查错误已修正，未产生实验结果。</p><h2>可用性与学习次数</h2>'''+table(stats)+'''<h2>0.4秒同目标未来误差（像素）</h2><p>joint_success分母为present_targets，包含没有可用预测的目标；平均误差只对有效目标统计。框中心不是重心，稀疏端点不是稠密ADE，没有独立RPM/姿态真值。</p><img src="axis_future_comparison62.png"><img src="all_target_errors62.png">'''+table(compact)+'''<h2>当时真的使用了方向吗</h2>'''+table(usage)+'''<img src="axis_admission_timeline62.png"><h2>当前方向证据来自哪里</h2>'''+table(evidence)+'''<h2>固定时刻与完整视频</h2><p>固定取清晰23.00、侧视110.50、3号28.00秒，未根据成败挑选。橙色旧方向，紫色新方向，绿色历史。视频紫色是新轴向分支；全部15模型数值仍保留。</p>'''
    for item in fixed:
        case=item['case'];page+='<h3>'+html.escape(case)+'</h3><img src="'+item['path']+'"><video controls preload="metadata" src="raw/'+case+'/forecasts.mp4"></video>'
    page+='''<h2>接下来优先解决</h2><p>方向可用与可靠要分开。本轮新增信息没有全面超过历史，尤其3号变差，下一步先审查短弧/细线方向的跨时刻一致性和可观测分量，再决定是否允许它参与在线学习；不能以更多更新作为目标。现有轴覆盖不足时，复用原6的整机PCA/LSD多阈值观测，借当前整机检测约束身份，仅更新无向轴；不把线中点当桨心、不要求四桨锁定。转速路线继续保留原局部时间分箱/ON-OFF周期和谐波歧义，再单独研究缺椭圆/重叠条件的周期提取。未知质量/推力系数/像素尺度先作为有效组合系数；重力投影和风等作为扰动，不声称分别恢复真实物理值。</p><p>本轮复用<a href="https://arxiv.org/html/2604.18289v1">桨叶状态估计论文</a>的方向/转速/动力学结构、本地27/28/51/52/29算法及<a href="https://www.mathworks.com/help/ident/ug/online-recursive-least-squares-estimation.html">标准在线RLS</a>；未安装新库、不宣称创新。当前实际单步仍超20ms预算，不用视频播放帧率代表实时吞吐。</p><p><a href="delivery62.json">交付核查</a> · <a href="analysis.json">全部评分</a> · <a href="protocol.json">冻结回放协议</a> · <a href="raw/raw_prefix_checks.json">实际RAW核查</a> · <a href="score_adapter.json">评分适配与失败记录</a></p></html>'''
    (OUT/'第六十二阶段方向与转速独立使用报告.html').write_text(page,encoding='utf-8')
    print('62 delivery and fixed-time visual evidence complete',flush=True)


if __name__=='__main__':run()
