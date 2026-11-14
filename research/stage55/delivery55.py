"""Summarize the frozen paired experiment; no inference or parameter selection."""
from pathlib import Path
import argparse, datetime, html, json
import cv2, numpy as np
import study55 as current
import recorded_run55 as recorded
OUT = current.OUT
CASES = ['clear6', 'side6', 'train3']
MODELS = ['trusted_acc', 'adaptive_axis_freq', 'adaptive_all_mix']
SUPPLEMENT = OUT/'delivery_supplement.json'

def freeze():
    recorded.verify()
    assert not SUPPLEMENT.exists()
    current.save(SUPPLEMENT, dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        helper_sha256=current.sha(Path(__file__)),
        design='All saved off/on cases, all15 models and all horizons; fixed0.4s figures for history/axis-frequency/all-mixture; fixed25/50/75 percent source-time examples for every positive recording. No parameter fitting or selected success cases.',
        labels='Only precomputed analysis after all formal predictions saved. No new labels or dense ADE claims.',
        limits='Internal four-frequency availability can include recently stored frequencies from non-visible tracks; it is not RPM or identity truth. Extra diagnostic run and original failed run stay separate from the frozen1715/3430 counts.'))
    print('Stage55 delivery supplement frozen', flush=True)

def verify():
    recorded.verify()
    assert current.load(SUPPLEMENT)['helper_sha256'] == current.sha(Path(__file__))

def table(rows, cols):
    def value(x):
        if isinstance(x, float): return f'{x:.4f}'
        return str(x)
    return '<table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(value(r.get(c)))+'</td>' for c in cols)+'</tr>' for r in rows)+'</table>'

def render():
    verify()
    for case in current.CASES:
        assert (OUT/case/'completion.json').exists() and (OUT/case/'regression.json').exists(), case
    prefix=current.load(OUT/'raw_prefix_checks.json')
    assert prefix['actual_RAW_prefix'] and prefix['complete_forecasts_lifecycle_ROI_hint_exact']
    checks=current.load(OUT/'score_checks.json')
    assert checks['all_coverage_identical54'] and checks['scoring_rows']==360
    analysis=current.load(OUT/'analysis.json')
    lookup={(r['case'],r['model'],r['nominal_h']):r for r in analysis['table']}
    paired=[]
    for case in CASES+['negative5']:
        for model in current.previous.previous.runner.MODELS:
            for horizon in [.1,.2,.4]:
                off=lookup['off_'+case,model,horizon];on=lookup[case,model,horizon]
                assert all(off[k]==on[k] for k in ['present_targets','valid_forecasts','anchors_available'])
                paired.append(dict(case=case,model=model,horizon=horizon,targets=on['present_targets'],valid=on['valid_forecasts'],
                    off_fde_px=off['mean_fde_px'],on_fde_px=on['mean_fde_px'],
                    delta_fde_px=None if on['mean_fde_px'] is None else on['mean_fde_px']-off['mean_fde_px'],
                    off_success=off['joint_success'],on_success=on['joint_success']))
    current.save(OUT/'paired_results.json',dict(table=paired,interpretation='Negative delta is lower conditional mean endpoint error, not independent significance or dense ADE; coverage and exact target times must also match.'))
    internal=[]
    for case in current.SCORING:
        rr=current.rows(OUT/case/'observations_forecasts.jsonl');tt=current.rows(OUT/case/'rotor_ticks.jsonl')
        feats=[r['frontends']['delayed_feedback']['feature_diagnostic'] for r in rr]
        internal.append(dict(case=case,native10=len(tt),parent20=len(rr),
            four_frequency_without_four_current=sum(bool(r['sensing']['four_frequencies'] and not r['sensing']['four_visible']) for r in tt),
            reference_retirements=sum(len(r['slot_audit']['retired_ids']) for r in tt),
            inferred_shape_candidates=sum(bool(x.get('inferred_shape')) for r in tt for x in r['observation'].get('rotors',[])),
            active_parent=sum(x['active'] for x in feats),
            four_visible_parent=sum(x['four_visible'] for x in feats),
            active_four_visible_quality_gt045=sum(x['active'] and x['four_visible'] and x['quality']>.45 for x in feats),
            quality_median=float(np.median([x['quality'] for x in feats])),
            frequency_updates=sum(r['lifecycle']['delayed_feedback']['actual_update_increments']['adaptive_axis_freq'] for r in rr)))
    current.save(OUT/'observability_diagnostic.json',dict(table=internal,limits='Quality>0.45 is the unchanged legacy physical learning gate, not a threshold chosen from these results. Counts are internal availability. Stale frequencies may be fresh for160ms; partial shapes have current event support but inferred shape parameters. No virtual constellation points are serialized as measured rotors.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#277da1','#e76f51'];x=np.arange(3);width=.35
    summaries=current.load(OUT/'summary.json')['table'];sl={r['case']:r for r in summaries}
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.4),layout='constrained')
    for j,(prefix_name,label) in enumerate([('off_','ROI off'),('','Past body ROI')]):
        for ax,field,title in [(axes[0],'four_visible','Four currently visible tracks'),(axes[1],'four_frequency','Four available frequencies')]:
            values=[100*sl[prefix_name+c][field]/sl[prefix_name+c]['native10_ticks'] for c in CASES]
            bars=ax.bar(x+(j-.5)*width,values,width,color=colors[j],label=label)
            ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8);ax.set_title(title);ax.set_ylabel('Native10ms ticks (%)');ax.set_ylim(0,110)
        values=[sl[prefix_name+c]['actual_frequency_updates'] for c in CASES]
        bars=axes[2].bar(x+(j-.5)*width,values,width,color=colors[j],label=label)
        axes[2].bar_label(bars,fmt='%d',padding=3,fontsize=8)
    axes[2].set_title('Actual delayed physical updates');axes[2].set_ylabel('Updates')
    for ax in axes:ax.set_xticks(x,['Clear6','Side6','Train3']);ax.legend(fontsize=8)
    fig.suptitle('Internal availability is not true rotor / RPM accuracy',fontsize=12)
    fig.savefig(OUT/'availability_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.6),layout='constrained')
    for ax,case in zip(axes,CASES):
        for j,prefix_name in enumerate(['off_','']):
            rows=[lookup[prefix_name+case,m,.4] for m in MODELS]
            bars=ax.bar(x+(j-.5)*width,[r['mean_fde_px'] if r['mean_fde_px'] is not None else np.nan for r in rows],width,color=colors[j],label='ROI off' if j==0 else 'Past body ROI')
            for bar,r in zip(bars,rows):
                if r['mean_fde_px'] is not None:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{r['mean_fde_px']:.1f}\n{r['joint_success']}/{r['present_targets']}",ha='center',va='bottom',fontsize=8)
        reference=lookup[case,'trusted_acc',.4];ax.set_title(f"{case}: valid {reference['valid_forecasts']}/{reference['present_targets']}")
        ax.set_xticks(x,['History','Axis+frequency','All mixture'],rotation=12,fontsize=8);ax.set_ylabel('Mean future endpoint error (px)');ax.margins(y=.25);ax.legend(fontsize=8)
    fig.suptitle('Same sparse future targets, nominal0.4s; labels above bars: error / joint success',fontsize=11)
    fig.savefig(OUT/'future_comparison.png',dpi=160);plt.close(fig)
    examples=[]
    for case in CASES:
        rr=current.rows(OUT/case/'observations_forecasts.jsonl');canvas=np.full((3*(360+34),1280,3),248,np.uint8)
        for row,fraction in enumerate([.25,.5,.75]):
            index=min(len(rr)-1,max(0,round(len(rr)*fraction)-1))
            for col,prefix_name in enumerate(['off_','']):
                capture=cv2.VideoCapture(str(OUT/(prefix_name+case)/'forecasts.mp4'));capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read();capture.release();assert ok
                image=cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA);y=row*394
                cv2.putText(canvas,f"{case} | {'ROI off' if col==0 else 'Past body ROI'} | t={rr[index]['t']:.2f}s",(col*640+8,y+24),cv2.FONT_HERSHEY_SIMPLEX,.5,(30,45,55),1)
                canvas[y+34:y+394,col*640:(col+1)*640]=image
            examples.append(dict(case=case,fraction=fraction,frame_index=index,t=rr[index]['t']))
        assert cv2.imencode('.png',canvas)[0]
        cv2.imencode('.png',canvas)[1].tofile(str(OUT/f'{case}_fixed_examples.png'))
    current.save(OUT/'fixed_example_selection.json',dict(rule='25/50/75% of each complete source-time clip, paired identical index, no choice by error',records=examples))
    selected=[r for r in paired if r['horizon']==.4 and r['case']!='negative5' and r['model'] in MODELS]
    failed=current.load(OUT/'ipc_observation_supplement.json')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>55：整机框引导桨盘的实际收益与局限</title><style>body{max-width:1400px;margin:28px auto;padding:0 16px;font:17px/1.8 "Microsoft YaHei",Arial;color:#203747}img,video{width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ccd8df}a{color:#1670a0}</style><h1>55：整机框引导桨盘提取，预测效果逐段核对</h1><p>固定过去的NN整机框，25%余量、160ms过期，仅裁剪当前10ms几何；完整原始频率历史、阈值、15模型与在线学习不变。当前20ms整机定位和六历史预测与54精确相同。本轮为已有ROI方法的受控整合，不宣称新算法。</p><img src="availability_comparison.png"><p>四频率可用包含160ms以内的历史频率，并不要求四桨此刻均可见；不能将这两列互换，也不能视为真实RPM精度。真实转轴身份、法向和RPM仍无独立真值。</p><img src="future_comparison.png"><p>同一批未来目标、相同有效覆盖；误差为稀疏可见框中心的未来端点误差，联合成功阈值沿用未来框对角线25%。不是密集ADE或真实重心误差。三段都属于开发已见数据，不是新的盲测。</p>'''+table(selected,['case','model','targets','valid','off_fde_px','on_fde_px','delta_fde_px','off_success','on_success'])+'''<h2>物理更新与可观测性</h2>'''+table(internal,['case','four_frequency_without_four_current','reference_retirements','active_parent','active_four_visible_quality_gt045','frequency_updates'])+'''<p>增加候选或延长频率可用不一定提高真实推力估计，仍需要可靠的四桨归属、谐波选择及姿态。原门槛没有按结果修改。假定五圆刚体有助于关联，但四点任意单应性零残差本身不能证明它们属于真实四桨。</p><h2>固定时刻画面：左关闭，右开启</h2>'''+''.join(f'<h3>{c}</h3><img src="{c}_fixed_examples.png">' for c in CASES)+'''<p>每段固定25/50/75%时刻，不挑成功画面。绿线历史、紫线轴向频率；active=False时紫线退回CV。视频50fps是采集时间，不代表处理实时。</p><h2>完整侧视预测视频</h2><video controls src="side6/forecasts.mp4"></video><h2>失败与验证</h2><p>首次正式运行在首条预测写出前JSON解析失败，原失败完整保留。单步严格解析诊断通过，补充只记录下一次异常原文，不放宽解析规则；原始异常行未保留，根因尚未确认。正式ROI关闭完整15模型回归、开启同位置/六历史、空场零输出、独立225/450 RAW前缀及全部未来评分检查均单独保存。</p><p><a href="第五十五阶段整机框引导桨盘报告.html">完整统计与原报告</a> · <a href="paired_results.json">全模型/时距配对结果</a> · <a href="observability_diagnostic.json">内部可观测性诊断</a> · <a href="raw_prefix_checks.json">真实RAW前缀</a> · <a href="score_checks.json">评分回归</a> · <a href="ipc_observation_supplement.json">原失败与补充</a></p><p>原始研究：<a href="https://arxiv.org/abs/2604.18289">Relative State Estimation using Event-Based Propeller Sensing</a> · <a href="https://arxiv.org/abs/2609.11920">EVPeriscope</a>。实际复用本地17裁剪、54共享传输、53/52完整推理，不是执行上述作者完整系统。</p></html>'''
    (OUT/'桨盘提取对照结果与固定时刻画面.html').write_text(page,encoding='utf-8')
    current.save(OUT/'delivery_counts.json',dict(parent=sum(current.load(OUT/c/'regression.json')['parent_frames'] for c in current.CASES),
        native10=sum(current.load(OUT/c/'regression.json')['native10_ticks'] for c in current.CASES),
        video=sum(current.load(OUT/c/'regression.json')['video_decoded_frames'] for c in current.CASES),
        separate_diagnostic_parent=1,original_failed_run_preserved=True,strict_parser_supplement_verified=True))
    print('Stage55 figures, paired analysis and fixed examples complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','render']);globals()[parser.parse_args().action]()
