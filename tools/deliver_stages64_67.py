"""Deliver compensation, projected-direction forecasts and local periodicity tests."""
from pathlib import Path
from collections import Counter
import html,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research/stage64'),str(ROOT/'research/stage65'),str(ROOT/'research/stage67')]
import study64 as motion
import study65 as forecast
import study67 as periodic
OUT=periodic.OUT;load=motion.load;save=motion.save;sha=motion.sha;rows=motion.rows


def table(rr):
    if not rr:return '<p>无记录。</p>'
    cols=list(rr[0]);return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[c],4) if isinstance(r[c],float) else r[c]))+'</td>' for c in cols)+'</tr>' for r in rr)+'</table></div>'


def run():
    motion.verify();forecast.verify();periodic.verify()
    raw_checks=load(motion.OUT/'raw_prefix_checks.json');assert raw_checks['totals']==dict(parent=985,native10=1970)
    score_checks=load(forecast.OUT/'score_checks.json');assert score_checks==dict(rows=540,baseline_rows_exact62=180,all_coverage_exact62=True)
    control=load(motion.OUT/'synthetic64.json');assert control['zero_velocity_support_exact'] and control['future_velocity_refused'] and control['stale_velocity_refused']
    motion_table=[];synthetic_table=[];motion_deltas=[]
    for case in motion.MAIN:
        c=load(motion.OUT/case/'completion.json');assert c['static_complete_measurements_exact63'] and not c['labels_read']
        for method,d in c['methods'].items():motion_table.append(dict(case=case,method=method,parent=c['parent'],candidates=d.get('valid',0),during62_inactive=d.get('while62_axis_inactive',0)))
        rr=rows(motion.OUT/case/'observations.jsonl')
        for rep in ['total','bipolar']:
            diffs=[float(np.degrees(motion.angle_distance(r['methods']['static_'+rep]['angle_rad'],r['methods']['aligned_'+rep]['angle_rad']))) for r in rr if r['methods']['static_'+rep]['valid'] and r['methods']['aligned_'+rep]['valid']]
            motion_deltas.append(dict(case=case,representation=rep,paired_candidates=len(diffs),median_direction_change_deg=float(np.median(diffs)) if diffs else None,
                                      warning='Change is not measured improvement: real plane normal unknown.'))
    for estimate in ['correct','wrong_sign']:
        for method in ['static_total','aligned_total','static_bipolar','aligned_bipolar']:
            rr=[r for r in control['rows'] if r['estimate']==estimate and r['method']==method]
            thin=[r for r in rr if r['ratio']==.06];accepted=[r for r in thin if r['accepted']]
            thick=[r for r in rr if r['ratio']>.4]
            synthetic_table.append(dict(velocity=estimate,method=method,thin_cases=len(thin),thin_accepted=len(accepted),
                thin_accepted_mean_error_deg=float(np.mean([r['axis_error_deg'] for r in accepted])) if accepted else None,
                thick_controls=len(thick),thick_accepted=sum(r['accepted'] for r in thick)))
    analysis=load(forecast.OUT/'analysis.json');predictions={c:rows(forecast.OUT/c/'forecasts.jsonl') for c in forecast.CASES}
    comparison=[];usage=[];corrections=[]
    for r in analysis['table']:
        if r['case']!='negative5' and r['nominal_h']==.4 and r['model'] in ['trusted_acc','adaptive_axis_only','adaptive_all_mix']:
            comparison.append({k:r[k] for k in ['case','frontend','model','present_targets','valid_forecasts','mean_fde_px','joint_success']})
    for case,rr in predictions.items():
        by_t={round(r['t'],8):r for r in rr}
        for method,d in load(forecast.OUT/case/'completion.json')['methods'].items():
            corrections.append(dict(case=case,method=method,applied=d['applied'],reasons=d['reasons']))
            dd=[d for d in analysis['details'] if d['case']==case and d['frontend']==method and d['model']=='adaptive_axis_only' and d['nominal_h']==.4 and d['target_state']=='present' and d['forecast_valid']]
            usage.append(dict(case=case,method=method,valid04_targets=len(dd),corrected_anchor_targets=sum(by_t[round(d['anchor_t'],8)]['correction_audit'][method]['applied'] for d in dd)))
    periodic_table=[]
    for case in periodic.CASES:
        a=load(periodic.OUT/case/'regression67.json');assert a['original66_current_spectra_velocity_and_sampling_exact']
        periodic_table+=a['table']
    controls67=load(periodic.OUT/'controlled67.json');extra=load(periodic.OUT/'holdout_rate_controls67.json')
    assert extra['helper_sha256']==sha(ROOT/'research/stage67/holdout67.py') and extra['parameters_unchanged']
    rate_table=[]
    for kind in ['periodic','mixed_periodic','constant_poisson','single_burst','one_step']:
        rr=[r for r in controls67['rows'] if r['kind']==kind]
        rate_table.append(dict(process=kind,cases=len(rr),FFT_peak=sum(r['valid'] for r in rr),
            dominant_half=sum(r['stable'] for r in rr),multi_half=sum(r['multi_half_stable'] for r in rr)))
    for r in extra['table']:rate_table.append(dict(process=r['kind'],cases=r['cases'],FFT_peak=r['spectral'],dominant_half=r['dominant_half'],multi_half=r['multi_half']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.1),layout='constrained')
    for ax,case in zip(axes,['clear6','side6','train3']):
        for offset,method,label,color in [(-.18,'static_total','Original count support','#f4a261'),(.18,'aligned_total','Motion aligned counts','#277da1')]:
            r=next(v for v in motion_table if v['case']==case and v['method']==method)
            ax.bar(np.array([0,1])+offset,[r['candidates'],r['during62_inactive']],.36,label=label,color=color)
            for x,v in zip([0,1],[r['candidates'],r['during62_inactive']]):ax.text(x+offset,v+3,str(v),ha='center',fontsize=9)
        ax.set(title=case,xticks=[0,1],xticklabels=['Direction candidates','During62 axis inactive'],ylabel='Current20ms steps');ax.set_ylim(0,max(r['parent']*1.1,1));ax.legend(fontsize=7)
    fig.suptitle('Actual RAW geometry support: more candidates do not establish true attitude accuracy')
    fig.savefig(OUT/'motion_support64.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    for ax,case in zip(axes,['clear6','side6','train3']):
        rr=[next(r for r in comparison if r['case']==case and r['frontend']==method and r['model']=='adaptive_axis_only') for method in forecast.METHODS]
        history=next(r for r in comparison if r['case']==case and r['frontend']=='axis62' and r['model']=='trusted_acc')
        values=[r['mean_fde_px'] for r in rr]+[history['mean_fde_px']]
        ax.bar(['Axis62','Body static','Body aligned','History'],values,color=['#7b2cbf','#f4a261','#277da1','#2a9d8f'])
        for j,v in enumerate(values):ax.text(j,v+max(values)*.025,f'{v:.3f}',ha='center',fontsize=8)
        ax.set(title=case+f" ({rr[0]['valid_forecasts']}/{rr[0]['present_targets']} targets)",ylabel='0.4s sparse endpoint error (px)',ylim=(0,max(values)*1.18));ax.tick_params(axis='x',labelrotation=15)
    fig.suptitle('Projected heading-only cache ablation: identical learning/RPM/history states')
    fig.savefig(OUT/'partial_heading_forecast65.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,case in zip(axes,['clear6','side6','train3']):
        for method,label,color in [('axis62','Axis62','#7b2cbf'),('body_aligned','Aligned heading','#277da1')]:
            dd=[r for r in analysis['details'] if r['case']==case and r['frontend']==method and r['model']=='adaptive_axis_only' and r['nominal_h']==.4 and r['target_state']=='present' and r['forecast_valid']]
            ax.plot([r['target_t'] for r in dd],[r['error_px'] for r in dd],'o-',color=color,label=label,ms=4)
        ax.set(title=case,xlabel='Future target time(s)',ylabel='Endpoint error(px)');ax.legend(fontsize=8)
    fig.suptitle('All valid0.4s targets, no success selection; unavailable anchors kept in denominators')
    fig.savefig(OUT/'all_target_errors65.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,4.8),layout='constrained');xx=np.arange(len(rate_table))
    for offset,key,label,color in [(-.25,'FFT_peak','Spectral peak','#f4a261'),(0.,'dominant_half','Dominant halves','#277da1'),(.25,'multi_half','Multi-peak halves','#7b2cbf')]:
        ax.bar(xx+offset,[r[key]/r['cases'] for r in rate_table],.25,label=label,color=color)
    ax.set(xticks=xx,xticklabels=[r['process'] for r in rate_table],ylabel='Fraction accepted by periodicity diagnostic',ylim=(0,1.13));ax.tick_params(axis='x',labelrotation=25);ax.legend(ncol=3,fontsize=8)
    ax.set_title('Known rate-process controls: colored aperiodic variation exposes false acceptance')
    fig.savefig(OUT/'periodicity_controls66_67.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,case in zip(axes,['clear6','side6','train3']):
        rr=[next(r for r in periodic_table if r['case']==case and r['method']==method) for method in ['raw','aligned']]
        for offset,key,label,color in [(-.22,'original_spectral_frames','Any peak','#f4a261'),(0.,'dominant_half_stable_frames','Dominant half','#277da1'),(.22,'multi_half_stable_frames','Multi-peak half','#7b2cbf')]:
            ax.bar(np.arange(2)+offset,[r[key] for r in rr],.22,label=label,color=color)
        ax.set(title=case,xticks=[0,1],xticklabels=['Unaligned','Aligned'],ylim=(0,rr[0]['parent']*1.15),ylabel='Current20ms steps with a candidate');ax.legend(fontsize=7)
    fig.suptitle('RAW shadow period candidates without rotor center; not true RPM recall')
    fig.savefig(OUT/'local_periodicity66_67.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,6),layout='constrained')
    for col,case in enumerate(['clear6','side6','train3']):
        cfg=periodic.CASES[case];t=round((cfg['start']+cfg['duration']*.5)*50)/50
        rr=rows(periodic.OUT/case/'observations.jsonl');r=next(r for r in rr if abs(r['t']-t)<1e-8)
        for row,method in enumerate(['raw','aligned']):
            ax=axes[row,col];image=np.full((3,6),np.nan);counts=np.zeros((3,6),int)
            for tile in r['methods'][method]['tiles']:
                if tile['multi_half_stable']:
                    m=max(tile['matched_frequencies'],key=lambda x:x['full_snr']);image[tile['row'],tile['column']]=m['frequency_hz'];counts[tile['row'],tile['column']]=len(tile['matched_frequencies'])
            plot=ax.imshow(np.ma.masked_invalid(image),vmin=60,vmax=1200,cmap='viridis')
            for y in range(3):
                for x in range(6):
                    if np.isfinite(image[y,x]):ax.text(x,y,f'{image[y,x]:.0f}\n({counts[y,x]} hyp)',ha='center',va='center',fontsize=7,color='white' if image[y,x]<650 else 'black')
            ax.set(title=f'{case} {method} t={t:.2f}',xticks=[],yticks=[])
    fig.colorbar(plot,ax=axes.ravel().tolist(),label='Strongest matching local rate hypothesis (Hz)',shrink=.8)
    fig.suptitle('Tile coordinates are ROI cells, not motor IDs; harmonics/multiple origins remain unresolved',fontsize=10)
    fig.savefig(OUT/'fixed_frequency_hypotheses67.png',dpi=150);plt.close(fig)
    common=dict(helper_sha256=sha(__file__),new_default_promoted=False,raw_joint_deployment_added=False)
    save(motion.OUT/'delivery64.json',dict(common,raw_checks=raw_checks,synthetic_table=synthetic_table,table=motion_table,direction_changes=motion_deltas,
         scope='1970 RAW window reads include repeated prefixes; main1400 windows. No real pose truth.'))
    save(forecast.OUT/'delivery65.json',dict(common,score_checks=score_checks,comparison04=comparison,corrections=corrections,scored_correction_usage04=usage,
         conclusion='No substantial future gain; train3 axis138.262->138.107px, side/clear unchanged. '
                    'Same learned parameters/RPM/history; no new RAW joint inference or realtime claim.'))
    save(periodic.previous.OUT/'delivery66.json',dict(common,controlled_counts=rate_table[:5],
         scope='Local spectral and dominant-half candidates only; no true mechanical RPM or thrust.'))
    save(OUT/'delivery67.json',dict(common,original66_spectra_exact=True,controlled_table=rate_table,
         extra_controls_sha256=sha(OUT/'holdout_rate_controls67.json'),RAW_table=periodic_table,
         RAW_counts=dict(native10_per_run=320,parent_per_run=160,total_66_67_reads=640,unique_native_windows=320),
         conclusion='Multi-peak agreement preserves two-rate mixtures but accepts31/40 '
                    'additional aperiodic colored-rate processes and all periodic illumination '
                    'flicker. Do not feed these candidates into thrust/RLS. Train3 has8/50 '
                    'aligned candidate frames vs old0 frequency frames, not certified RPM.',
         next='Reuse existing perception._frequency phase coherence and rotor_sensing '
              'harmonic/rotation-phase hypotheses; add temporal phase and spatial common-mode '
              'background/rotor association checks before any new force feature. Preserve '
              'colored-noise/flicker controls and holdout processes, do not tune solely on '
              'the3.2s RAW. Old physical input/predictor remains unchanged.'))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>64–67：运动补偿、方向预测与局部周期</title><style>body{max-width:1450px;margin:28px auto;padding:0 22px;font:16px/1.8 system-ui;color:#233e4d;background:#f7fafc}.lead{background:#eaf1fa;border-left:5px solid #277da1;padding:16px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:13px}td,th{border:1px solid #cbd6df;padding:6px;vertical-align:top}img{max-width:100%;height:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}a{color:#2457a3}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>64–67：补偿平移，再验证方向与转速信号</h1><div class="lead"><b>这一轮没有明显未来预测提升；找到了“局部周期不能直接当转速”的关键失效条件。</b><p>64复用运动补偿，3号全事件方向候选217→235；65只修正画面内方向，保留原倾角/学习，3号0.4秒138.262→138.107px，侧视113.917px不变。66/67不要求桨心或椭圆即可产生局部周期，但多峰规则在额外有色非周期控制中接受31/40，照明闪烁也通过，因此没有接入推力学习。原默认保持，最新完整自动RAW预测入口仍是62。</p></div><h2>64：只用过去速度，补偿事件与历史支持图</h2><p>先查CVPR2018事件对齐工作，实际复用原21逐事件平移与原6历史图warpAffine、PCA/LSD。速度来自已提交的20ms整机CV状态，逐10ms断言其时间戳不晚于当前包起点，超过0.16秒弃用。不用本包末尾才获得的速度，无新对比度优化器或依赖。静态两表示的完整观测逐项精确63；14秒主段加两个实际RAW前缀，共985父/1970原生窗口读取，前缀精确。全事件与双极表示保持原阈值。</p><p>已知方向点支持模拟中，18个薄形状正确补偿全事件18个通过，平均已接受角误差约0.135°；未补偿9个通过、已接受平均约1.931°。两者接受集合不同，不能只用这两个平均数宣称统一精度倍数。6个厚形状与估反速度的控制完整保留。点支持模拟不是真实事件传感器或无人机姿态真值。</p>'''+table(synthetic_table)+'''<img src="motion_support64.png">'''+table(motion_table)+'''<p>真实候选角度变化并不等于误差下降：没有真实桨盘法线标注。</p>'''+table(motion_deltas)+'''<h3>3号固定28.00秒：左未补偿，右补偿</h3><div class="pair"><img src="../stage64_motion_compensated_axis_20261021/train3/static_total_28.00.png"><img src="../stage64_motion_compensated_axis_20261021/train3/aligned_total_28.00.png"></div><h2>65：线只修正二维投影方向，保留未知倾角</h2><p>比较62基线、静态侧影、补偿侧影三路。只在当前转子方向缺失、原方向仍未过期且仍active时，使用连续3次、变化≤12°的侧影方向修正轴向分支rollout。保留n_z与||n_xy||、原0.16秒时限；线不初始化完整3D法线，不延长姿态寿命，不产生新RLS样本。所有学习参数/协方差/扰动、转速专家与历史系数逐项精确62。700当前缓存输入、三路2100步，独立缓存前缀通过，540行评分、180基线行精确62。这里是缓存因果预测消融，没有新RAW联合部署或延迟测量。</p><p>侧视只改2个时刻，3号静态改3个、补偿改6个。原门槛失效时没有用一条线补造完整姿态，因此大部分断档仍在。候选数增多没有自动变成预测改进。</p><img src="partial_heading_forecast65.png"><img src="all_target_errors65.png">'''+table(comparison)+table(corrections)+table(usage)+'''<h2>66：不依赖桨心/椭圆的局部ON/OFF周期</h2><p>复查原论文III-D与FrequencyCam。实际复用原逐极性时间分箱、Hann FFT和原160ms窗口/100us分箱/60–1200Hz/SNR8；整机ROI分成6×3单元，比较原坐标与过去CV速度对齐。局部格子不是四个电机槽，没有将最强峰换成总推力。原450000全窗事件上限与均匀步长保留。3.2秒RAW（清晰22–22.8、侧视112–113、3号28–29、空场0–0.4秒），使用缓存同截止NN框。空场特意用固定ROI，避免靠没有NN身份让测试自动通过。</p><p>单独FFT强峰在单次突发32/32、亮度台阶29/32中错误接受；要求主峰跨半窗一致能拒绝这些控制，但也拒绝两频混合32/32。因此补充67保留原top5峰，只要求每个候选分别在两个半窗得到SNR≥8支持。66原结果不改，67重读同3.2秒，去掉新增字段后所有旧频谱/速度/采样记录逐项精确66，两轮共640次原生窗口读取，只有320个不同窗口。</p><h2>67：多峰保持了混合信号，也暴露更难的假周期</h2><p>原两频混合32/32恢复；另取未用于初始设计的40组混合频率，接受40/40、匹配74/80个已知分量（以原6.25Hz FFT格距统计）。但40组有色非周期强度变化中也接受31组，周期照明40/40通过。周期检测无法独自判断信号是否由桨叶产生，增加候选数不能作为真实转速精度。</p><img src="periodicity_controls66_67.png">'''+table(rate_table)+'''<h2>真实RAW局部周期候选</h2><p>3号同1秒原逐桨频率为0，本轮未补偿有5/50、补偿有8/50个半窗稳定候选时刻；尚无同步转速真值，这不是新增真实RPM召回率。清晰/侧视很多格子存在多个频率或谐波，不能重复计为多个电机或直接平方求和。</p><img src="local_periodicity66_67.png">'''+table(periodic_table)+'''<h3>固定中点频率假设地图</h3><p>清晰22.40、侧视112.50、3号28.50秒。色值为匹配候选中SNR最大的局部频率，括号是候选数；不表示机械转速或电机身份。空白表示没有通过该诊断。</p><img src="fixed_frequency_hypotheses67.png"><h2>下一步与保持的边界</h2><p>优先复用已存在的perception._frequency时间相位一致性，以及rotor_sensing的谐波/旋转相位候选，给无椭圆局部周期增加时间相位、空间共同照明背景与桨叶关联约束。必须保留有色变化、闪烁、混合频率与频率切换控制；在这些检查前不把频谱候选输入推力/RLS。方向链还需解决当前NN身份缺失和长时间未知倾角，不能仅靠放宽期限达到连续。</p><p>全部试验属于已知开发录像。稀疏可见框不是重心，尚无真实姿态/RPM/推力/风速真值；没有新的实时证明或超过论文的结论。质量、推力系数和像素尺度仍作为有效组合参数，不能把观测错误都解释为风。没有G输入、没有修改yolov10-main、没有新依赖。</p><p><a href="../stage64_motion_compensated_axis_20261021/protocol.json">64协议</a> · <a href="../stage64_motion_compensated_axis_20261021/raw_prefix_checks.json">64真实RAW前缀</a> · <a href="../stage65_partial_heading_20261021/analysis.json">65全部未来评分</a> · <a href="../stage66_center_free_periodicity_20261021/controlled66.json">66原控制</a> · <a href="controlled67.json">67多峰控制</a> · <a href="holdout_rate_controls67.json">额外不调参控制</a> · <a href="delivery67.json">交付与下一步</a></p><p>原始相关研究：<a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Gallego_A_Unifying_Contrast_CVPR_2018_paper.html">事件运动补偿</a> · <a href="https://arxiv.org/html/2604.18289v1">桨叶状态估计</a> · <a href="https://github.com/ros-event-camera/frequency_cam">FrequencyCam作者代码</a>。本轮执行的是本地既有组件与补充诊断，没有运行完整FrequencyCam或声称新颖算法。</p></html>'''
    (OUT/'第六十七阶段多峰周期与本轮预测总览.html').write_text(page,encoding='utf-8')
    print('64–67 report, complete comparisons and all controlled failures saved',flush=True)


if __name__=='__main__':run()
