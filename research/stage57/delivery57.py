"""Reuse55 numeric figures; add three-way results and unchanged4/50 gate audit."""
from pathlib import Path
from collections import Counter
import argparse,datetime,hashlib,html,inspect,sys
import numpy as np
import study57 as current
sys.path[:0]=[str(ROOT/'core'),str(current.ROOT/'research/stage55'),str(current.ROOT/'research/stage50')]
import delivery55 as old
import layout50 as layout
OUT=current.OUT;CASES=['clear6','side6','train3'];MODELS=['trusted_acc','adaptive_axis_freq','adaptive_all_mix']
SUPPLEMENT=OUT/'delivery_supplement.json'

def figure_source():
    source=inspect.getsource(old.render)
    start=source.index("    failed=current.load(OUT/'ipc_observation_supplement.json')")
    end=source.index("    current.save(OUT/'delivery_counts.json'")
    source=source[:start]+source[end:]
    source=source.replace('Past body ROI','Candidate gate').replace('Stage55 figures','Stage57 reused figures')
    source=source.replace('separate_diagnostic_parent=1,original_failed_run_preserved=True,strict_parser_supplement_verified=True',
        'separate_diagnostic_parent=0,strict_parser_error_line_recording=True')
    return source

def sources():
    return {str(p.relative_to(current.ROOT)):current.sha(p) for p in [Path(__file__),current.ROOT/'research/stage55/delivery55.py',current.ROOT/'research/stage50/layout50.py',current.ROOT/'research/stage4/tracker.py']}

def freeze():
    current.verify();assert not SUPPLEMENT.exists();source=figure_source()
    current.save(SUPPLEMENT,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),derived_numeric_render_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(),
        design='Execute actual55 plotting/paired-sparse metrics/fixed25-50-75% examples with newOUT and labels. Remove only old HTML/failed55 bookkeeping, not arithmetic. Add all three record/model0.4s comparisons vs55 hard ROI, plus original4/50 initialization eligibility on same current57 geometry. No fitting or new thresholds.',
        hard55_analysis_sha256=current.sha(current.previous.OUT/'analysis.json'),old56_results_sha256=current.sha(current.BASE/'stage56_roi_layout_association_20261021/results.json')))
    (OUT/'derived_numeric_render.py').write_text(source,encoding='utf-8');print('Stage57 three-way delivery frozen',flush=True)

def verify():
    current.verify();p=current.load(SUPPLEMENT)
    assert p['source_hashes']==sources() and p['derived_numeric_render_sha256']==hashlib.sha256(figure_source().encode('utf-8')).hexdigest()
    assert p['hard55_analysis_sha256']==current.sha(current.previous.OUT/'analysis.json')
    assert p['old56_results_sha256']==current.sha(current.BASE/'stage56_roi_layout_association_20261021/results.json')

def render():
    verify();namespace=dict(old.__dict__);namespace.update(OUT=OUT,verify=verify)
    exec(compile(figure_source(),'<57 actual55 numeric plots>','exec'),namespace);namespace['render']()
    assert not list(OUT.rglob('ipc_failure.json'))
    now=current.load(OUT/'analysis.json');hard=current.load(current.previous.OUT/'analysis.json')
    lookup={(r['case'],r['model'],r['nominal_h']):r for r in now['table']};hard_lookup={(r['case'],r['model'],r['nominal_h']):r for r in hard['table']}
    comparisons=[]
    for case in CASES:
        for model in MODELS:
            baseline=lookup['off_'+case,model,.4];clipped=hard_lookup[case,model,.4];candidate=lookup[case,model,.4]
            for r in [clipped,candidate]:assert r['present_targets']==baseline['present_targets'] and r['valid_forecasts']==baseline['valid_forecasts']
            comparisons.append(dict(case=case,model=model,targets=baseline['present_targets'],valid=baseline['valid_forecasts'],
                global_fde_px=baseline['mean_fde_px'],hard55_fde_px=clipped['mean_fde_px'],candidate57_fde_px=candidate['mean_fde_px'],
                global_success=baseline['joint_success'],hard55_success=clipped['joint_success'],candidate57_success=candidate['joint_success']))
    current.save(OUT/'three_way_results.json',dict(table=comparisons,coverage_identical=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14.8,4.8),layout='constrained');x=np.arange(3);width=.25
    for ax,case in zip(axes,CASES):
        selected=[r for r in comparisons if r['case']==case]
        for j,(field,label,color) in enumerate([('global_fde_px','Global54','#277da1'),('hard55_fde_px','Hard ROI55','#e76f51'),('candidate57_fde_px','Candidate57','#43aa8b')]):
            values=[r[field] for r in selected];bars=ax.bar(x+(j-1)*width,values,width,label=label,color=color);ax.bar_label(bars,fmt='%.1f',fontsize=7,padding=3)
        r=selected[0];ax.set_title(f"{case}: valid{r['valid']}/{r['targets']}");ax.set_xticks(x,['History','Axis+frequency','All mixture'],rotation=12,fontsize=8);ax.set_ylabel('Future endpoint error (px)');ax.margins(y=.18);ax.legend(fontsize=7)
    fig.suptitle('Nominal0.4s, same sparse future targets; no rotor/pose ground truth',fontsize=11);fig.savefig(OUT/'three_way_comparison.png',dpi=160);plt.close(fig)
    gates=[];tester=layout.LayoutSlots(layout.PaperConfig())
    for case in CASES:
        parent=current.rows(OUT/case/'observations_forecasts.jsonl');hints={}
        for r in parent:
            for t in [r['t']-.01,r['t']]:hints[round(t*1e6)]=r['body_hint_sent']
        native=current.rows(OUT/case/'rotor_ticks.jsonl');reasons=Counter();eligible=streak=maximum=four=0;pending=None
        for r in native:
            rotors=r['observation'].get('rotors',[]);four+=len(rotors)==4
            initial,reason=layout.skeleton.initialization(rotors)
            if initial is not None and not tester.inside(rotors,hints[round(r['t']*1e6)],r['t']):initial=None;reason='body_hint_missing_or_outside'
            reasons[reason]+=1
            if initial is None:streak=0;pending=None;continue
            eligible+=1;center=initial[0].mean(0);streak=streak+1 if pending is not None and np.linalg.norm(center-pending)<50 else 1;pending=center;maximum=max(maximum,streak)
        gates.append(dict(case=case,native10=len(native),four_current_candidates=four,eligible_raw_current_ticks=eligible,maximum_raw_current_streak=maximum,required_streak=layout.skeleton.PARAMS['init_frames'],reasons=dict(reasons)))
    current.save(OUT/'initialization_diagnostic.json',dict(table=gates,actual_original4_50_gate=True,no_frequency_or_association_change=True,limits='Gate availability is not actual rotor identity/pose accuracy; perspective may violate affine parallelogram assumption.'))
    cols=['case','model','targets','valid','global_fde_px','hard55_fde_px','candidate57_fde_px','global_success','hard55_success','candidate57_success']
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>57：原方案、硬ROI与完整轮廓候选筛选</title><style>body{max-width:1450px;margin:28px auto;padding:0 16px;font:17px/1.8 "Microsoft YaHei",Arial;color:#203747}img,video{width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd8df}</style><h1>57：三种桨盘提取的真实未来位置对照</h1><p>本轮保留所有当前事件、原全图密度分位数和完整连通轮廓；已到达的NN整机框仅筛选拟合后的候选中心，随后原分组/跟踪/频率/15模型不变。25%余量和160ms期限不调。框提示不晚于20ms包开始，不用未来框。</p><img src="three_way_comparison.png"><p>三路相同稀疏未来目标/有效覆盖，联合成功阈值仍为未来框对角线25%；不是密集ADE、真实重心或新盲测。误差变动需与成功数一起看。原54基线与55硬裁结果完整保留。</p>'''+old.table(comparisons,cols)+'''<h2>旧机架初始化条件是否恢复</h2>'''+old.table(gates,['case','native10','four_current_candidates','eligible_raw_current_ticks','maximum_raw_current_streak','required_streak','reasons'])+'''<p>调用原4/50初始化，不放宽五帧、尺寸/闭合/扁平度门槛。它反映旧模型的可启用性，不能证明机械转子身份或姿态正确。候选筛选联合避免了硬裁的轮廓截断与阈值变化；尚未把两种影响单独归因。</p><img src="availability_comparison.png"><p>四频率可包含160ms内历史值，不等于当前四桨同时可见；当前RPM/法向缺少独立真值。所有输入及六历史预测与基线固定，空场和实际RAW前缀单独验证。</p><h2>固定25/50/75%时刻：左全图，右候选筛选</h2>'''+''.join(f'<h3>{c}</h3><img src="{c}_fixed_examples.png">' for c in CASES)+'''<p>不挑成功帧；绿色历史、紫色轴向频率，active=False时退回CV；50fps是源时间，不代表处理实时。</p><h2>侧视完整视频</h2><video controls src="side6/forecasts.mp4"></video><h2>3号完整视频</h2><video controls src="train3/forecasts.mp4"></video><p><a href="第五十七阶段完整轮廓候选ROI报告.html">完整统计和全部时距</a> · <a href="three_way_results.json">三路数值</a> · <a href="initialization_diagnostic.json">原机架门槛</a> · <a href="raw_prefix_checks.json">真实RAW前缀</a> · <a href="score_checks.json">评分回归</a> · <a href="protocol.json">冻结协议</a></p><p>已有基础：<a href="https://arxiv.org/html/2604.18289v1">事件桨叶状态估计</a>、<a href="https://arxiv.org/html/2609.11920v1">EVPeriscope</a>。实际复用原本地检测/55边界/54传输和完整物理推理，不称新颖方法或作者完整复现。</p></html>'''
    (OUT/'三路预测对照与固定时刻画面.html').write_text(page,encoding='utf-8');print('Stage57 three-way comparison and initialization audit complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','render']);globals()[p.parse_args().action]()
