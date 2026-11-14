"""Deliver same-target future scores, fixed source-time videos and explicit failures."""
from pathlib import Path
import datetime,html,json
import cv2,numpy as np
import forecast60 as current
import checks60
OUT=current.current.OUT

def table(rows):
    if not rows:return ''
    cols=list(rows[0]);return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[c],4) if isinstance(r[c],float) else r[c]))+'</td>' for c in cols)+'</tr>' for r in rows)+'</table>'

def run():
    checks60.verify();checks=current.load(current.OUT/'score_checks.json');prefix=current.load(current.OUT/'raw_prefix_checks.json')
    assert checks['rows']==180 and checks['history_rows_exact57']==72 and checks['coverage_exact57']
    assert prefix['all_states_and_semantic_named_event_fields_exact']
    module=current.load(OUT/'results.json')['table'];raw=current.load(current.isolated60.OUT/'raw_summary.json')['table']
    new_scores=current.load(current.OUT/'analysis.json')['table'];old_scores=current.load(current.REF/'analysis.json')['table']
    lookup={(r['case'],r['model'],r['nominal_h']):r for r in old_scores}
    differences=[];funnel=[]
    for r in new_scores:
        old=lookup[r['case'],r['model'],r['nominal_h']]
        if r!=old:differences.append(dict(case=r['case'],model=r['model'],horizon=r['nominal_h'],baseline_fde_px=old['mean_fde_px'],paired_fde_px=r['mean_fde_px'],baseline_joint_success=old['joint_success'],paired_joint_success=r['joint_success']))
    for case in current.SCORING:
        records=current.load(current.isolated60.OUT/f'raw_{case}.json')['records'];rejections={}
        for r in records:
            for reason,n in r['line_audit'].get('profile_rejections',{}).items():rejections[reason]=rejections.get(reason,0)+n
        funnel.append(dict(case=case,base_pairs=sum(r['line_audit'].get('base_pairs',0) for r in records),stable_pairs=sum(r['line_audit'].get('stable_pairs',0) for r in records),confirmed_pairs=sum(r['line_audit'].get('confirmed_pairs',0) for r in records),profile_usable=sum(r['line_audit'].get('profile_usable',0) for r in records),profile_rejections=rejections))
    availability=[];counts=dict(parent=0,native10=0,video=0)
    for case in current.CASES:
        reg=current.load(current.OUT/case/'regression60.json');completion=current.load(current.OUT/case/'completion.json')
        for key,field in [('parent','parent'),('native10','native10'),('video','video_decoded_frames')]:counts[key]+=reg[field]
        if case not in current.SCORING:continue
        old=current.rows(current.REF/case/'rotor_ticks.jsonl');oldc=current.load(current.REF/case/'completion.json');new=current.rows(current.OUT/case/'rotor_ticks.jsonl')
        f='adaptive_axis_freq';availability.append(dict(case=case,native10=reg['native10'],fused=reg['fused'],inserted=reg['inserted'],
          original_four_visible=sum(r['sensing']['four_visible'] for r in old),paired_four_visible=reg['four_visible'],
          original_four_frequencies=sum(r['sensing']['four_frequencies'] for r in old),paired_four_frequencies=reg['four_frequencies'],
          original_RLS_updates=oldc['methods']['delayed_feedback']['actual_updates'][f],paired_RLS_updates=completion['methods']['delayed_feedback']['actual_updates'][f],
          processing_median_ms=completion['processing_median_ms']))
    assert counts==dict(parent=965,native10=1930,video=740)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained');models=['trusted_acc','adaptive_axis_freq','adaptive_all_mix'];labels=['History','Axis+frequency','All mix']
    for ax,case in zip(axes,['clear6','side6','train3']):
        rr=[next(r for r in checks['comparison'] if r['case']==case and r['model']==m) for m in models];x=np.arange(3)
        ax.bar(x-.18,[r['baseline_fde_px'] for r in rr],.36,label='Original57',color='#277da1')
        ax.bar(x+.18,[r['paired_fde_px'] for r in rr],.36,label='Paired60',color='#e76f51')
        ax.set(title=case,xticks=x,xticklabels=labels,ylabel='0.4s sparse endpoint error (px)');ax.tick_params(axis='x',labelrotation=15);ax.legend(fontsize=8)
        for i,r in enumerate(rr):ax.text(i,max(r['baseline_fde_px'],r['paired_fde_px'])+2,f"{r['baseline_fde_px']:.2f} / {r['paired_fde_px']:.2f}",ha='center',fontsize=8)
        ax.set_ylim(0,max(r['baseline_fde_px'] for r in rr)*1.25+5)
    fig.suptitle('Same body positions/coverage; full15 models. Sparse boxes are not COM or dense ADE.',fontsize=11)
    fig.savefig(OUT/'paired_future_controls.png',dpi=160);plt.close(fig)
    examples=[]
    for case in ['clear6','side6','train3']:
        n=round(current.CASES[case]['duration']*50)
        for numerator in [1,2,3]:
            index=n*numerator//4;panels=[]
            for folder,label in [(current.REF/case,'Original57'),(current.OUT/case,'Paired60')]:
                cap=cv2.VideoCapture(str(folder/'forecasts.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read();cap.release();assert ok
                cv2.rectangle(frame,(0,0),(500,30),(0,0,0),-1);cv2.putText(frame,label+' source index='+str(index),(10,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1);panels.append(frame)
            name=f'forecast_compare_{case}_{numerator}.png';cv2.imencode('.png',np.hstack(panels))[1].tofile(str(OUT/name));examples.append(name)
    current.save(OUT/'future_delivery.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=current.sha(Path(__file__)),counts=counts,
        score_rows=180,history_exact_rows=72,comparison=checks['comparison'],all_changed_score_rows=differences,observation_funnel=funnel,availability=availability,default_promoted=False,
        initial_raw_failure_root_cause_confirmed=False,checker_invalid_frame_fix_only=True,fixed_examples=examples))
    images=''.join('<img src="'+name+'">' for name in examples)
    extras='<h2>全部时距变化，避免只看0.4秒</h2>'+('<p>全部180评分行均精确原57。</p>' if not differences else table(differences))+'<h2>候选到有效中心的损失发生在哪</h2>'+table(funnel)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>60 双边中心线完整预测验证</title><style>body{max-width:1500px;margin:30px auto;font:16px/1.8 Arial;color:#233e4c}img,video{width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd8df}.box{padding:18px;background:#eef5f7;border-left:5px solid #277da1}</style><h1>60：双边中心线、端点密度与完整未来预测对照</h1><div class="box">已将侧视线段建议落到实际RAW：标准LSD双边配对、跨阈值与时间确认、条带内部/端点检查，再与同一薄椭圆观测作有界权重融合。整机位置、原频率/关联/RLS和15模型保持。四桨真实身份与物理桨尖仍无独立真值，最终是否有用由下表同目标未来评分判断，默认仍54。</div><h2>相同整机位置与有效覆盖的0.4秒未来端点</h2><img src="paired_future_controls.png">'''+table(checks['comparison'])+'''<p>180行全部15模型×3时距×4片段评分，72行六历史分支逐项精确57，覆盖/目标一致。不是密集ADE或真实重心误差。全部预测、完整视频解码及真实RAW前缀通过后才读稀疏框标签。源数据已多轮开发，不当作未见泛化测试。</p><h2>当前观测、分组与频率可用性</h2>'''+table(raw)+table(availability)+'''<p>原始450当前时刻的双边诊断见下表；正式RAW扩大到1400当前时刻，两前端独立状态比较，关闭原算法逐时刻精确57。没有凑四点。profile_usable只是活动支持有界/近似对称，不是认证的物理桨尖；没有独立周期身份时也不冒称已获得真实电机编号。插入/融合次数不同于实际参与最终分组次数。四可见/四频率是内部代理，不能当成真实RPM精度。</p>'''+table(module)+'''<h2>完整RAW与失败保留</h2><p>正式预测含关闭试跑及独立前缀，共965父/1930原生子时刻，740视频帧全解码。原生20ms整机像素/NN/六历史与覆盖精确57，实际新10ms几何精确独立RAW当前对照；关闭试跑全部15与生命周期精确57。独立4.5秒225/450前缀的全部状态、命名事件字段SHA、ROI/线审计/15模型精确新全段，传输内字节校验保持。50fps是源时间，不是全链实时处理率。</p><p>首次跨录像单进程当前对照在3号发生严格回归断言失败，当时没有保存首个不同观测，根因未确认；clear/side完整输出和原冻结源保留。独立旧前端、关闭分支、交错双分支三个新进程各600时刻精确57。之后按正式worker相同的逐录像新进程生命周期，1400时刻全部通过，没有放宽断言或改数值算法。完整预测首次试跑正常结束，但新校验器错误地在valid=False的空系数字典里索引hold，导致KeyError；单独checks60补充要求无效双方系数为空、有效双方全部六历史键存在并精确，完整15校验保留，原推理源未改。</p><h2>固定源时刻画面对照</h2><p>每段25/50/75%固定取帧，左57/右60。绿历史、紫物理；物理active=False会回退CV。新旧使用相同当前NN位置，不能用画面像素几乎不变证明预测正确。</p>'''+images+'''<h2>下一步</h2><p>本轮先以实际分组/频率/未来结果决定是否保留该分支；不要继续通过放宽门槛凑四点。现阶段侧视在跨阈值前已不稳定，优先核查局部事件表示/时间窗造成的轮廓变化，以及旋转事件与平移事件的分离。原21已有过去速度warp20且没有统一收益，原6也试过平移密度图，不能重复当新方法。已查SCARF原始事件局部记忆和RSS2026 Motion-aware Event Suppression作者库；尚未下载执行，后者预测动态掩码，不等于桨叶物理轨迹。</p><p><a href="forecast/analysis.json">全部未来评分</a> · <a href="forecast/raw_prefix_checks.json">真实RAW前缀</a> · <a href="第六十阶段双边中心线与尺度确认报告.html">最初450样本配对</a> · <a href="isolated_raw/protocol.json">独立当前RAW协议</a> · <a href="forecast/protocol.json">完整预测冻结协议</a> · <a href="forecast/audit_supplement.json">校验器修正说明</a> · <a href="initial_interleaved_failure.json">原失败记录</a> · <a href="https://arxiv.org/html/2609.19973">SCARF</a> · <a href="https://github.com/uzh-rpg/event_suppression">Motion-aware Event Suppression</a></p></html>'''
    page=page.replace('<h2>完整RAW与失败保留</h2>',extras+'<h2>完整RAW与失败保留</h2>')
    page=page.replace('现阶段侧视在跨阈值前已不稳定','现阶段侧视在跨阈值和时间确认时损失大量候选')
    page=page.replace('下一步按冻结门槛在实际RAW密度中检查','下一步按冻结门槛在实际RAW密度中检查')
    page=page.replace('<h2>下一步</h2>','<h2>下一步</h2><p>中心与姿态轴应分开准入。侧视307对原始双边→74对跨阈值稳定→7对时间确认→1对通过端点支持，不能把端点不可靠直接等同轴方向不可用。下一轮应优先复用原6的整机PCA/LSD无向轴观测，以新NN整机身份与位置作约束，在不强造中心的前提下接同位置轴向预测；完整中心准入保持严格。仍需角度一致性/背景与机臂反例，不把事件线直接当推力速度方向。</p>')
    (OUT/'双边中心线完整预测结果.html').write_text(page,encoding='utf-8');print(json.dumps(dict(counts=counts,comparison=checks['comparison'],changed_score_rows=len(differences),funnel=funnel),ensure_ascii=False),flush=True)

if __name__=='__main__':run()
