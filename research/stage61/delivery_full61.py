"""Complete three-way same-target prediction comparison and fixed-time evidence."""
from pathlib import Path
from collections import Counter
import datetime,html,json,sys
import cv2,numpy as np
import forecast61 as current
ROOT=current.ROOT;OUT=current.current.OUT
load=current.load;save=current.save;sha=current.sha;rows=current.rows

def table(rr):
    if not rr:return '<p>无变化行。</p>'
    cols=list(rr[0]);return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[k],5) if isinstance(r[k],float) else r[k]))+'</td>' for k in cols)+'</tr>' for r in rr)+'</table></div>'

def old_layout_gate():
    sys.path.insert(0,str(ROOT/'research/stage50'));import layout50 as old
    report=[]
    for case in current.SCORING:
        if current.CASES[case]['reference']=='negative5':continue
        hints={}
        for r in rows(current.OUT/case/'observations_forecasts.jsonl'):
            for t in [r['t']-.01,r['t']]:hints[round(t*1e6)]=r['body_hint_sent']
        tester=old.LayoutSlots(old.PaperConfig());eligible=streak=maximum=0;pending=None;reasons=Counter()
        for r in rows(current.OUT/case/'rotor_ticks.jsonl'):
            rotors=r['observation']['rotors'];initial,reason=old.skeleton.initialization(rotors)
            if initial is not None and not tester.inside(rotors,hints[round(r['t']*1e6)],r['t']):initial=None;reason='body_hint_missing_or_outside'
            reasons[reason]+=1
            if initial is None:streak=0;pending=None;continue
            eligible+=1;center=initial[0].mean(0);streak=streak+1 if pending is not None and np.linalg.norm(center-pending)<50 else 1;pending=center;maximum=max(maximum,streak)
        report.append(dict(case=case,eligible_current_ticks=eligible,longest_streak=maximum,required_streak=old.skeleton.PARAMS['init_frames'],reasons=dict(reasons)))
    return report,{name:sha(ROOT/name) for name in ['research/stage50/layout50.py','research/stage4/tracker.py']}

def run():
    current.verify();checks=load(current.OUT/'score_checks.json');prefix=load(current.OUT/'raw_prefix_checks.json')
    assert checks['rows']==360 and checks['history_rows_exact57']==144 and checks['all_coverage_exact57']
    assert prefix['parent_ticks']==570 and prefix['native10_ticks']==1140 and prefix['all_states_full15_and_named_event_hashes_exact']
    totals=dict(parent=0,native10=0,video=0);availability=[];full15_equal={};sourcepairs={};samecontext=0
    for case in current.CASES:
        r=load(current.OUT/case/'regression61.json');s=load(current.OUT/case/'completion.json')
        totals['parent']+=r['parent'];totals['native10']+=r['native10'];totals['video']+=r['video_decoded_frames'];samecontext+=r['same_context_module_current_ticks_exact']
        if case not in current.SCORING:continue
        ticks=rows(current.OUT/case/'rotor_ticks.jsonl');ref=current.CASES[case]['reference'];old=rows(current.REF/ref/'rotor_ticks.jsonl');oldc=load(current.REF/ref/'completion.json');model='adaptive_axis_freq'
        availability.append(dict(case=case,native10=r['native10'],preline_routed=r['preline_routed'],preline_rejected=r['preline_rejected'],line_measured=r['line_measured'],
          pair_fused=r['fused'],pair_inserted=r['inserted'],four_visible=r['four_visible'],four_frequencies=r['four_frequencies'],
          legacy_four_visible=sum(x['sensing']['four_visible'] for x in old),legacy_four_frequencies=sum(x['sensing']['four_frequencies'] for x in old),
          retired_rotor_ids=sum(len(x['slot_audit']['retired_ids']) for x in ticks),
          RLS_updates=s['methods']['delayed_feedback']['actual_updates'][model],legacy_RLS_updates=oldc['methods']['delayed_feedback']['actual_updates'][model],
          processing_median_ms=s['processing_median_ms']))
    assert totals==dict(parent=2010,native10=4020,video=1440)
    for ref in ['clear6','side6','train3','negative5']:
        a=rows(current.OUT/f'off_{ref}/observations_forecasts.jsonl');b=rows(current.OUT/f'paired_{ref}/observations_forecasts.jsonl')
        full15_equal[ref]=dict(parent=len(a),equal_complete_frontend_and_lifecycle_ticks=sum(x['frontends']==y['frontends'] and x['lifecycle']==y['lifecycle'] for x,y in zip(a,b)))
        if ref in ['clear6','negative5']:assert full15_equal[ref]['equal_complete_frontend_and_lifecycle_ticks']==len(a)
        sourcepairs[ref]=(a,b)
    new=load(current.OUT/'analysis.json');old=load(current.REF/'analysis.json');oldmap={(r['case'],r['model'],r['nominal_h']):r for r in old['table']};bynew={(r['case'],r['model'],r['nominal_h']):r for r in new['table']};changed=[]
    active_at_score=[]
    for case in current.SCORING:
        if current.CASES[case]['reference']=='negative5':continue
        anchors={round(r['t'],8):r['frontends']['delayed_feedback'] for r in rows(current.OUT/case/'observations_forecasts.jsonl')}
        details=[r for r in new['details'] if r['case']==case and r['model']=='adaptive_axis_freq' and r['target_state']=='present' and r['forecast_valid']]
        for h in [.1,.2,.4]:
            selected=[r for r in details if r['nominal_h']==h];active=updated=equal_cv=0
            for r in selected:
                q=anchors[round(r['anchor_t'],8)];model=q['feature_diagnostic']['models']['adaptive_axis_freq']
                active+=bool(model['active']);updated+=model['updates']>0
                cv=next(d for d in new['details'] if d['case']==case and d['model']=='cv' and d['nominal_h']==h and d['target_t']==r['target_t'])
                equal_cv+=np.allclose(r['predicted'],cv['predicted'],atol=1e-9,rtol=0)
            active_at_score.append(dict(case=case,horizon=h,valid_sparse_targets=len(selected),axis_freq_active_anchors=active,anchors_with_prior_RLS_updates=updated,axis_freq_prediction_equal_CV=equal_cv))
    for ref in ['clear6','side6','train3']:
        models=[r['model'] for r in new['table'] if r['case']==f'off_{ref}' and r['nominal_h']==.4]
        for model in models:
            for h in [.1,.2,.4]:
                a=oldmap[ref,model,h];b=bynew[f'off_{ref}',model,h];c=bynew[f'paired_{ref}',model,h]
                if dict(b,case=ref)!=a or dict(c,case=ref)!=a:
                    changed.append(dict(case=ref,model=model,horizon=h,legacy_fde_px=a['mean_fde_px'],stable_fde_px=b['mean_fde_px'],paired_fde_px=c['mean_fde_px'],
                        legacy_success=a['joint_success'],stable_success=b['joint_success'],paired_success=c['joint_success']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained');models=['trusted_acc','adaptive_axis_freq','adaptive_all_mix'];names=['History','Axis+frequency','All mix']
    for ax,ref in zip(axes,['clear6','side6','train3']):
        rr=[next(r for r in checks['comparison'] if r['case']==ref and r['model']==m and r['horizon']==.4) for m in models];x=np.arange(3)
        for offset,key,label,color in [(-.25,'legacy57_fde_px','Legacy57','#277da1'),(0.,'stable_fde_px','Stable61','#43aa8b'),(.25,'paired_fde_px','Stable+paired61','#e76f51')]:
            ax.bar(x+offset,[r[key] for r in rr],.25,label=label,color=color)
        for i,r in enumerate(rr):ax.text(i,max(r[k] for k in ['legacy57_fde_px','stable_fde_px','paired_fde_px'])+2,f"{r['legacy57_fde_px']:.2f} / {r['stable_fde_px']:.2f} / {r['paired_fde_px']:.2f}",ha='center',fontsize=7)
        ax.set(title=ref,xticks=x,xticklabels=names,ylabel='0.4s sparse endpoint error (px)');ax.tick_params(axis='x',labelrotation=14);ax.legend(fontsize=7)
        ax.set_ylim(0,max(max(r[k] for k in ['legacy57_fde_px','stable_fde_px','paired_fde_px']) for r in rr)*1.22+6)
    fig.suptitle('Same body positions/coverage, full15 models. Sparse visible boxes are not COM/dense ADE.',fontsize=11)
    fig.savefig(OUT/'stable_paired_future_controls.png',dpi=160);plt.close(fig)
    # Every valid sparse target is shown; no successful-target selection.
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,ref in zip(axes,['clear6','side6','train3']):
        for case,model,label,color in [(f'off_{ref}','trusted_acc','History','#444444'),(f'off_{ref}','adaptive_axis_freq','Stable axis+frequency','#43aa8b'),(f'paired_{ref}','adaptive_axis_freq','Paired axis+frequency','#e76f51')]:
            rr=[r for r in new['details'] if r['case']==case and r['model']==model and r['nominal_h']==.4 and r['target_state']=='present' and r['forecast_valid']]
            ax.plot([r['target_t'] for r in rr],[r['error_px'] for r in rr],'o-',ms=4,label=label,color=color)
        ax.set(title=ref,xlabel='Sparse future target source time (s)',ylabel='Endpoint error (px)');ax.legend(fontsize=7)
    fig.suptitle('All valid0.4s sparse targets; unavailable anchors retained in score-table denominators',fontsize=11)
    fig.savefig(OUT/'stable_paired_target_errors.png',dpi=160);plt.close(fig)
    images=[]
    for ref in ['clear6','side6','train3']:
        n=round(current.CASES[f'off_{ref}']['duration']*50)
        for numerator in [1,2,3]:
            index=n*numerator//4;panels=[]
            for mode,label in [('off','Stable61 / pairing off'),('paired','Stable61 + pairing')]:
                cap=cv2.VideoCapture(str(current.OUT/f'{mode}_{ref}/forecasts.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();cap.release();assert ok
                cv2.rectangle(frame,(0,0),(620,31),(0,0,0),-1);cv2.putText(frame,label+f' source index={index}',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(frame)
            name=f'full61_{ref}_{numerator}.png';cv2.imencode('.png',np.hstack(panels))[1].tofile(str(OUT/name));images.append(name)
    layout_gate,layout_source_hashes=old_layout_gate()
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),totals=totals,
       unique_main_native10=1400,main_native10_processed=2800,same_context_current_module_checks=samecontext,
       score_rows=360,history_exact_rows=144,actual_RAW_prefix_parent=570,actual_RAW_prefix_native10=1140,
       comparison=checks['comparison'],all_changed_score_rows=changed,availability=availability,complete15_and_lifecycle_pair_checks=full15_equal,
       fixed_source_time_examples=images,default_promoted=False,stage60_attempt_still_failed_unscored=True,
       unchanged_old_layout_initialization_gate=layout_gate,layout_diagnostic_source_hashes=layout_source_hashes,
       layout_scope='Current-only raw observations and original past hints; actual old4 initialization/50 inside gates only, no association intervention/new frequency and no true identity claim.',
       physical_feature_usage_at_scored_anchors=active_at_score)
    save(OUT/'full_future_delivery61.json',result)
    gallery=''.join('<img src="'+name+'">' for name in images)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>61 稳定线段与配对：完整预测对照</title><style>body{max-width:1550px;margin:30px auto;font:16px/1.85 Arial;color:#233e4c}h1{font-size:30px}img,video{width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd8df}.scroll{overflow-x:auto;margin:16px 0}.box{padding:18px;background:#eef5f7;border-left:5px solid #277da1}</style><h1>61：稳定线段基线、配对融合与旧方案的完整预测对照</h1><div class="box">两路均在椭圆拟合前识别薄轮廓；off关闭配对但保留稳定线段修复，paired再加入双边/端点观测融合。原NN整机位置、原始频率历史/关联/RLS/延迟选择及完整15模型保持。将旧57→稳定61的测量修复效果，与稳定61→配对61的融合效果分开比较，不将共同修复当作配对收益。</div><h2>0.4秒同目标未来误差</h2><img src="stable_paired_future_controls.png"><img src="stable_paired_target_errors.png"><p>下表包含0.1/0.2/0.4秒，而非只展示有利时距。360个全部模型/时距/case评分行均同覆盖，144行六历史结果精确旧57。目标是人工稀疏可见框中心，不是真实质心、密集ADE、姿态或RPM真值。这些录像已有多轮开发，不当作未见泛化成绩。</p>'''+table(checks['comparison'])+'''<h2>全部15模型中发生变化的评分行</h2>'''+table(changed)+'''<h2>前端、身份与频率内部可用性</h2>'''+table(availability)+'''<p>四可见、四频率、RLS次数和转子ID退休是内部代理指标，不能当作真实转轴身份/RPM准确率。薄线分支保守拒绝很短/不支持的线，数量下降不自动等于退化或改善；是否有预测收益仍看同目标未来误差。中心线的有界活动端点也不是认证物理桨尖。</p><h2>完整RAW与因果复查</h2>'''+table([totals])+'''<p>主对照是同14秒记录重复两路，共2800原生时刻处理、1400个不同主当前窗口；另有关闭试跑及四个独立实际RAW前缀，累计2010父/4020子时刻，1440视频帧全部解码。两个模式各自side4.5s、train31.2s前缀共570/1140，全部当前状态/ROI/线审计/命名事件字段SHA/15预测与生命周期精确，传输内字节校验保持。无薄线清晰0.8s关闭试跑完整15/生命周期精确旧57；初始上下文一致的当前模块片段也逐项对照，包含26.17已定位轮廓。没有在正式推理固定全局随机种子。</p><p>实际OpenCV5.0.0的cv2.pyd、ffmpeg DLL和构建SHA逐次核对，worker实际单线程。源码、配置与派生代码冻结，未按新分数调门槛。全部预测/回归/前缀完成后才运行原55/原22稀疏评分。60旧完整尝试仍保留为失败且未评分，不把其随机分组差异算作有效增益。</p><h2>固定源时刻画面</h2><p>每段25/50/75%固定取帧，左稳定基线、右加配对。绿历史，紫轴向频率；物理active=False时回退CV。50fps是源时间，处理仍超过20ms预算，跨运行负载未控制，不据此宣称固定加速倍数或实时。</p>'''+gallery+'''<h2>侧视完整视频</h2><p>稳定基线：</p><video controls src="forecast/off_side6/forecasts.mp4"></video><p>稳定基线＋配对：</p><video controls src="forecast/paired_side6/forecasts.mp4"></video><p><a href="forecast/analysis.json">全部评分/每个目标</a> · <a href="forecast/raw_prefix_checks.json">两模式真实RAW前缀</a> · <a href="forecast/protocol.json">冻结协议</a> · <a href="full_future_delivery61.json">完整统计</a> · <a href="第六十一阶段先判线再拟合椭圆报告.html">先前RNG因果控制与当前修复</a> · <a href="https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html">复用LSD API</a> · <a href="https://github.com/event-driven-robotics/RT-EvLDT">相关作者事件线跟踪（本轮未执行）</a></p></html>'''
    page=page.replace('<h2>完整RAW与因果复查</h2>','<h2>被评分的锚点是否真的使用了桨叶物理</h2>'+table(active_at_score)+'<p>active来自当时保存的模型状态，不是用未来误差反推；与CV相等按已保存的同一目标预测坐标比较。无有效物理状态时该分支回退CV，所以不能由当前检测变化推断未来一定会变化。</p><h2>原有机架初始化条件还卡在哪里</h2>'+table(layout_gate)+'<p>此表实际复用未修改的4初始化/50整机框门槛，仅诊断同一当前点集，不重新运行频率，也不靠放宽五帧要求凑初始化。仿射闭合/扁平度失败不自动证明检测点错误，透视、遮挡和退化也会造成拒绝；不能把任意四点单应性零残差当作已识别真实机架。</p><h2>完整RAW与因果复查</h2>')
    (OUT/'稳定线段与配对完整预测报告.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(totals=totals,samecontext=samecontext,changed_score_rows=len(changed),four_point_checks=full15_equal,comparison04=[r for r in checks['comparison'] if r['horizon']==.4]),ensure_ascii=False),flush=True)

if __name__=='__main__':run()
