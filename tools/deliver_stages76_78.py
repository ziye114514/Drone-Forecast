"""Failure mechanisms, isolated forecasts and fixed additional-recording tests."""
from pathlib import Path
import html,sys
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent/'output'
sys.path[:0]=[str(ROOT/'research/stage78'),str(ROOT/'research/stage77'),str(ROOT/'research/stage76')]
import study78 as actual
import study77 as cached
import diagnose76 as audit
OUT=actual.OUT;load=actual.load;save=actual.save;sha=actual.sha;rows=actual.rows

def table(records):
    if not records:return '<p>无可用样本。</p>'
    cols=list(records[0]);return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[k],3) if isinstance(r[k],float) else r[k]))+'</td>' for k in cols)+'</tr>' for r in records)+'</table></div>'

def run():
    actual.verify();cached.verify();checks=load(OUT/'checks78.json');scorechecks=load(OUT/'score_checks.json')
    assert checks['parent']==2185 and checks['native10']==4370 and checks['video_frames_decoded']==1850
    assert scorechecks==dict(rows=1029,original77_rows_exact=588,additional_rules_unchanged=True)
    analysis=load(OUT/'scoring/analysis.json');failure=load(audit.OUT/'audit76.json');controls=load(cached.OUT/'protocol.json')['controls']
    selected=['cv','trusted_acc','dual_cv_event_increment','history_when_missing','rollout_refresh','imm_state_fresh','imm_forecast_fresh','mmae_state_fresh','mmae_state_fresh_constant','mmae_forecast_fresh']
    main=[{k:r[k] for k in ['case','model','nominal_h','present_targets','valid_forecasts','mean_fde_px','joint_success','absent_targets','absent_forecasts_inside']} for r in analysis['table'] if r['nominal_h']==.4 and r['case']!='negative5' and r['model'] in selected]
    development=[r for r in main if r['case'] in ['clear6','side6','train3']]
    additional=[r for r in main if r['case'].startswith('extra')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    plot_models=['cv','trusted_acc','dual_cv_event_increment','mmae_state_fresh','mmae_state_fresh_constant']
    labels=['Original CV','Strong history','Old IMM','Isolated / event','Matched constant']
    colors=['#4e9bca','#52677e','#b16c9d','#cda72b','#9ca5b4']
    for group,cases in [('development',['clear6','side6','train3']),('additional',['extra1','extra2','extra4'])]:
        fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
        for ax,case in zip(axes,cases):
            rr=[next(r for r in main if r['case']==case and r['model']==m) for m in plot_models]
            values=[r['mean_fde_px'] for r in rr]
            if all(v is not None for v in values):
                bars=ax.bar(np.arange(len(values)),values,color=colors);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8);ax.set_ylim(0,max(values+[1])*1.25)
            else:ax.text(.5,.5,'No valid present-target forecasts',ha='center',transform=ax.transAxes)
            ax.set_xticks(np.arange(len(values)),labels,rotation=36,ha='right',fontsize=8);ax.set_title(f"{case} / n={rr[0]['valid_forecasts']}/{rr[0]['present_targets']}")
            ax.set_ylabel('Nominal0.4s sparse-box endpoint error (px)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle('Structural improvements are separate from event-signal benefit' if group=='development' else 'Fixed additional segments: no parameter changes after development ablation')
        fig.savefig(OUT/(group+'_comparison.png'),dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,case in zip(axes,['clear6','side6','train3']):
        dd=[d for d in failure['details'] if d['case']==case and d['nominal_h']==.4];tt=np.arange(len(dd))
        ax.bar(tt-.18,[d['state_shift_norm'] for d in dd],width=.36,label='State displacement',color='#9970ab')
        ax.bar(tt+.18,[d['future_force_shift_norm'] for d in dd],width=.36,label='Future control displacement',color='#d0a648')
        ax.set_title(case);ax.set_xlabel('All valid future targets, chronological');ax.set_ylabel('Displacement from original CV (px)');ax.legend(fontsize=7)
    fig.suptitle('Audit of old IMM: component displacement is not additive error attribution')
    fig.savefig(OUT/'old_error_components.png',dpi=160);plt.close(fig)
    availability=[];timing=[];fixed=[];causal_target_usage=[]
    for case in actual.SCORING_CASES:
        rr=rows(OUT/case/'observations_forecasts.jsonl');by={round(r['t'],8):r for r in rr}
        qq=[r['frontends']['delayed_feedback'] for r in rr]
        valid=[q for q in qq if q['valid']]
        availability.append(dict(case=case,parent=len(rr),valid_forecasts=len(valid),
            current_physical_support=sum(q['separated_diagnostic']['current'] for q in qq),
            history_fallback_valid_ticks=sum(not q['separated_diagnostic']['current'] for q in valid),
            all_forecasts_preserved=len(valid)))
        c=load(OUT/case/'completion.json')
        timing.append(dict(case=case,full_pipeline_median_ms=c['processing_median_ms'],full_pipeline_p95_ms=c['processing_p95_ms'],fraction_over20ms=c['fraction_over20ms']))
        candidates=[d for d in analysis['details'] if d['case']==case and d['model']=='mmae_state_fresh' and d['nominal_h']==.4 and d['forecast_valid'] and d['target_state']=='present']
        causal_target_usage.append(dict(case=case,valid04_targets=len(candidates),
            targets_with_current_physical_branch=sum(by[round(d['anchor_t'],8)]['frontends']['delayed_feedback']['separated_diagnostic']['current'] for d in candidates)))
        if not candidates or case=='negative5':continue
        cfg=actual.CASES[case];desired=cfg['start']+cfg['duration']/2
        d=min(candidates,key=lambda d:abs(d['target_t']-desired));index=round((d['anchor_t']-cfg['start'])*50)-1
        cap=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();cap.release();assert ok
        target=np.rint(d['target_center']).astype(int);pred=np.rint(d['predicted']).astype(int)
        cv2.drawMarker(im,tuple(target),(0,230,100),cv2.MARKER_CROSS,24,2)
        cv2.circle(im,tuple(pred),6,(40,220,240),2)
        cv2.putText(im,f"Future sparse center {d['target_t']:.2f}s | actual horizon {d['actual_h']:.2f}s | evaluation only",(12,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        filename=f'{case}_fixed_future.png';cv2.imencode('.png',im)[1].tofile(str(OUT/filename))
        fixed.append(dict(case=case,anchor_t=d['anchor_t'],target_t=d['target_t'],actual_h=d['actual_h'],error_px=d['error_px'],file=filename))
    video_cases=[c for c in actual.SCORING_CASES if c!='negative5']
    videos=''.join('<article><h3>'+c+'</h3><video controls preload="metadata" src="'+c+'/forecasts.mp4"></video></article>' for c in video_cases)
    images=''.join('<figure><img src="'+r['file']+'"><figcaption>'+html.escape(str({k:v for k,v in r.items() if k!='file'}))+'</figcaption></figure>' for r in fixed)
    headline=[]
    for case in video_cases:
        r={m:next(x for x in main if x['case']==case and x['model']==m) for m in plot_models}
        headline.append(dict(case=case,valid_targets=r['cv']['valid_forecasts'],CV=r['cv']['mean_fde_px'],history=r['trusted_acc']['mean_fde_px'],old_IMM=r['dual_cv_event_increment']['mean_fde_px'],new_isolated=r['mmae_state_fresh']['mean_fde_px'],matched_constant=r['mmae_state_fresh_constant']['mean_fde_px']))
    # Relative improvements must not silently treat missing/absent samples as0.
    headline_html=table(headline)
    page_body='''<h1>融合失败定位与状态隔离：76–78阶段</h1><div class="lead"><b>开发片段改善，额外录像未证实普遍收益；当前不能默认升级。</b><p>本轮没有改检测器、桨盘提取、频率算法或RLS参数。定位旧融合把历史加速度与延迟残差当成区间控制、输入缺失回CV、状态交互传递偏差的问题，固定十路结构对照后接入RAW，补测另外三段固定录像。桨叶转速信息的独立预测收益仍未证明，默认入口保持。</p></div>
<h2>当前0.4秒结果</h2><p>单位像素，越小越好。所有方法使用相同目标、同输出覆盖；有效目标数很少。新主审阅分支是mmae_state_fresh，在78执行前固定；不是按录像分别挑最好的模型。</p>'''+headline_html+'''
<h2>旧融合究竟错在哪</h2><p>76精确重构“原CV预测+滤波状态偏移+未来控制位移”。下图是位移大小，不是可相加的因果误差贡献。26.28时前一事件增量增益为0，错误控制仍主要来自约-938.6px/s²的历史加速度；26.8时轴老0.12秒、周期老0.10秒，却仍用较大控制传播。</p><img src="old_error_components.png">'''+table(failure['table'])+'''
<ul><li>物理缺失时旧融合经常回到CV，丢掉了已存在且更好的历史加速度预测。</li><li>旧代码把previous_history_a同时用于前一个滤波区间和未来外推；当前框已经更新后，未来外推仍在用旧历史输入。</li><li>72学到的是相对历史预测的延迟残差，没有被独立验证成瞬时物理加速度。把它传播到滤波状态，可能把预测偏差再累积成速度偏差。</li><li>IMM模式会交互状态，因此某模式的错误控制可以改动另一模式的状态。独立MMAE保护CV子状态，但最终加权输出仍可能错误。</li></ul>
<h2>固定做了哪些对照</h2><p>保留全部旧39路，再加10路：只刷新未来输入、只改缺失回历史，以及IMM/MMAE×区间控制/仅未来修正×事件/匹配恒定输入。所有噪声、初始概率、Markov转移和增益来自原74/72，不搜索参数。当前轴与周期都实际支持才接物理输出；不支持时继续用强历史预测，不减少发布数量。</p><p>常量与事件分支共用EVENT资格时钟，避免零激励常量因从不更新而被人为拒绝。实际学到的增益/增量仍分别保留。恒定输入取得近似收益说明收益主要来自结构与历史运动学，不能全部归于桨叶RPM。</p>
<h2>开发片段：所有结构消融</h2><img src="development_comparison.png">'''+table(development)+'''
<h2>另外三段固定录像</h2><p>train1 30–35秒、train2 30–35秒、train4 60–65秒由录像时长中点元数据选择，规则和参数在预测前固定，未按标签是否可见或误差筛片。它们用于本轮未调参的迁移检查；不保证检测器训练未见过这些录像，也不能冒称新无人机盲测。失败、缺席和无输出全部保留。</p><p>额外1号/2号的0.4秒有效评分锚点没有当前物理输入，新输出等于历史；2号因此60.50px差于CV46.97px。4号新分支55.86px差于强历史49.91px和匹配常量48.56px。三段不能作为事件预测成功或普遍迁移改善的证据。</p><img src="additional_comparison.png">'''+table(additional)+'''
<h2>多少预测真正用了物理分支</h2>'''+table(availability)+table(causal_target_usage)+'''
<p>回历史是保护机制，不是事件预测成功。这里“当前周期”仍基于过去160ms窗口，不是零延迟真实转速。当前支持少，说明后续重点仍是有用周期/姿态线索的可观测性与可靠性；不能仅靠增加融合模型补出不存在的信息。</p>
<h2>实际预测视频与固定画面</h2><p>蓝色原CV、紫色旧IMM、黄色新隔离分支；无当前物理支持时黄色回强历史且视频贡献权重显示0。视频按源时间50fps播放，处理速度见后表。固定帧按片段中点附近的有效目标选取，不按误差挑成功；绿色十字为评分后未来框中心、黄色圆为实际评分时刻预测。</p><div class="grid">'''+videos+'''</div>'''+images+'''
<h2>数值正确性和RAW验证</h2>'''+table([controls])+'''
<p>复用<a href="https://filterpy.readthedocs.io/en/latest/kalman/MMAEFilterBank.html">FilterPy独立MMAE作者实现</a>与<a href="https://github.com/rlabbe/filterpy/tree/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33">相同固定提交</a>，MIT许可保留。该固定MMAE源的向量混合协方差循环把均值分量当成标量，已由已知向量控制暴露；原源不改，外部适配器重算标准混合协方差及无测量均值。作者概率/独立滤波仍实际执行，不隐瞒适配修正。错误控制不改独立CV子状态的控制逐时刻通过。</p>
<p>77原39路逐时刻保持，588评分行/468旧行精确，refresh-only滤波位置/速度/概率/协方差精确旧74，只改变未来历史输入。78真正RAW入口禁止旧特征和标签读取；原区域全部49预测/状态精确77、当前观测精确75，两个实际RAW前缀通过。含额外15秒共2185父/4370原生窗口、1850视频帧全部解码；旧区间/前缀重复，不算独立样本。1029评分行中588原区域逐项精确77。</p>
<h2>耗时与仍未解决的问题</h2>'''+table(timing)+'''
<p>端到端仍不实时，本轮还同时计算49路用于研究对照，没有声称加速。稳定高精度、真实机械RPM/质量/风速、独立姿态误差仍未证明。下一步先改善周期与姿态输入的可靠覆盖、独立变向段标注和观测链延迟，避免继续用相同几个稀疏目标反复调规则。SDK原生FrequencyAlgorithm此前只核查API，尚未运行，本轮没有冒用它的结果。</p>
<p><a href="../stage76_fusion_failure_audit_20261021/audit76.json">逐目标失败分解</a> · <a href="../stage77_separated_filter_forecast_20261021/protocol.json">77固定协议和控制</a> · <a href="checks78.json">RAW/前缀/视频检查</a> · <a href="scoring/analysis.json">全部49路与额外录像评分</a> · <a href="protocol.json">78固定片段协议</a> · <a href="delivery_data78.json">本页数据</a> · <a href="../stage75_RAW_parallel_forecast_20261021/第七十四至七十五阶段双分支融合报告.html">保留的旧失败与视频</a></p>'''
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>状态隔离与未来预测修正报告</title><style>body{font:16px/1.7 system-ui;background:#f2f5f8;color:#203044;max-width:1300px;margin:32px auto;padding:0 24px}h1{font-size:30px}h2{margin-top:36px}.lead,article,figure{background:white;padding:20px;border-radius:12px;margin:18px 0}.lead{border-left:6px solid #cda72b}img{max-width:100%;height:auto;background:white}video{width:100%;background:#111}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}.scroll{overflow:auto;margin:20px 0}table{border-collapse:collapse;background:white;font-size:13px;width:100%}th,td{border:1px solid #d4dde6;text-align:left;padding:7px;vertical-align:top}th{background:#e6edf5}a{color:#245e99}figcaption{font-size:13px;overflow-wrap:anywhere}</style>'+page_body+'</html>'
    report=OUT/'第七十六至七十八阶段状态隔离与预测修正报告.html';report.write_text(page,encoding='utf-8')
    save(OUT/'delivery_data78.json',dict(headline=headline,development=development,additional=additional,
        availability=availability,target_usage=causal_target_usage,timing=timing,fixed=fixed))
    save(OUT/'delivery78.json',dict(helper_sha256=sha(__file__),report=str(report),actual_RAW49_completed=True,
        default_promoted=False,structure_improved_on_development=True,event_RPM_advantage_proven=False,
        independent_MMAE_source=actual.cache.SNAPSHOT))
    print('76–78 report delivered',report,flush=True)
if __name__=='__main__':run()
