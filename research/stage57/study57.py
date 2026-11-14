"""Frozen actual RAW comparison of complete-contour candidate ROI against54/55."""
from pathlib import Path
import argparse,datetime,hashlib,html,inspect,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage55')]
import run_candidate_forecast as runner
import study55 as previous
from gating57 import PARAMS
from rotor_worker57 import worker_source
from checks57 import run as checks
BASE=ROOT.parent/'output';OUT=BASE/'stage57_candidate_roi_20261021';REFERENCE=previous.REFERENCE
CASES=previous.CASES;SCORING=previous.SCORING
load=previous.load;save=previous.save;sha=previous.sha;rows=previous.rows

def sources():
    result=previous.sources()
    for n in ['run_candidate_forecast.py','research/stage57/gating57.py','research/stage57/rotor_worker57.py','research/stage57/checks57.py','research/stage57/study57.py']:
        result[n]=sha(ROOT/n)
    return result

def shared_source(name):
    source=inspect.getsource(getattr(previous,name))
    if name=='audit':
        old="d['mode']=='body_roi'";assert source.count(old)==1;source=source.replace(old,"d['mode']=='body_candidate_roi'")
    return source

def derived():
    return {k:hashlib.sha256(v.encode('utf-8')).hexdigest() for k,v in dict(parent=runner.pipeline_source(),worker=worker_source(),audit=shared_source('audit'),prefix=shared_source('prefix'),score=shared_source('score')).items()}

def freeze():
    assert not OUT.exists();module=checks();previous.verify()
    old=load(previous.OUT/'protocol.json')
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),derived_source_hashes=derived(),
        raw_metadata=old['raw_metadata'],label_hashes=old['label_hashes'],cases=CASES,parameters=PARAMS,module_checks=module,
        reference_hashes=old['reference_hashes'],previous55_results_hashes={n:sha(previous.OUT/n) for n in ['summary.json','analysis.json','geometry_current_diagnostic.json']},
        design='Actual RAW, unchanged54 full global contour inputs/threshold. Past55-style NN box screens complete fitted candidates before ranking/grouping; no current events removed and no clipped contour fit. All original full frequency/15 models. Off must exactly54; both modes same body/history. Control uses identical debug SHA and strict JSON parsing with error-line capture.',
        schedule=['pilot_off_clear6','pilot_body_clear6','pilot_negative5']+SCORING+['prefix_side6'],
        scoring='Reuse actual55 audit/prefix/score with OUT/CASES/verify routed. Audit changes only name of ROI diagnostic mode. Original22 sparse-future math and negative alias unchanged; predictions saved before labels.',
        prior_art=['https://arxiv.org/html/2604.18289v1','https://arxiv.org/html/2609.11920v1'],
        actual_reuse='Original9/8/PaperPerception full contour extraction;55 GeometryROI validated bounds,54 shared packets,53/52 complete inference,55 audit and scoring.',
        limits='Not a new method claim, not independent RPM/pose truth, no realtime guarantee, known development recordings. Parent box precedes packet start. Clipping/threshold causal attribution requires diagnostics; candidate gate intervention tests their combined removal.'))
    texts=dict(parent=runner.pipeline_source(),worker=worker_source(),audit=shared_source('audit'),prefix=shared_source('prefix'),score=shared_source('score'))
    for name,source in texts.items():(OUT/f'derived_{name}.py').write_text(source,encoding='utf-8')
    print('Stage57 frozen: complete contours/global threshold and past-box candidate gate',flush=True)

def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['derived_source_hashes']==derived();previous.verify()
    for name,value in p['previous55_results_hashes'].items():assert sha(previous.OUT/name)==value
    return p

def reused(name,*args):
    namespace=dict(previous.__dict__)
    namespace.update(OUT=OUT,REFERENCE=REFERENCE,CASES=CASES,SCORING=SCORING,verify=verify,runner=runner)
    exec(compile(shared_source(name),f'<57 reused55 {name}>','exec'),namespace)
    return namespace[name](*args)

def run(case):
    verify();cfg=CASES[case]
    runner.run(cfg['raw'],OUT/case,cfg['start'],cfg['duration'],precision=32,preview=case not in ['prefix_side6','pilot_negative5'],verify_frames=True,geometry_roi=cfg['geometry_roi'])

def audit(case):
    reused('audit',case)
    rr=rows(OUT/case/'rotor_ticks.jsonl')
    assert all(r['geometry_event_count']==r['event_count'] and r['roi_diagnostic']['geometry_events']==r['event_count'] for r in rr)
    result=load(OUT/case/'regression.json')
    result.update(all_current_events_preserved=True,candidate_calls=sum(r['roi_diagnostic']['candidate_calls'] for r in rr),
        candidates_before_gate=sum(r['roi_diagnostic']['candidates_before_gate'] for r in rr),candidates_after_gate=sum(r['roi_diagnostic']['candidates_after_gate'] for r in rr))
    save(OUT/case/'regression.json',result)

def prefix():return reused('prefix')
def score():return reused('score')

def report():
    verify();a=load(OUT/'analysis.json');old=load(REFERENCE/'analysis.json')
    refs={(r['case'],r['model'],r['nominal_h']):r for r in old['table'] if r['case'] in ['clear6','side6','train3','negative5']};exact=0
    for r in a['table']:
        compare=dict(r);compare['case']=CASES[r['case']]['reference'];ref=refs[compare['case'],r['model'],r['nominal_h']]
        assert all(compare[k]==ref[k] for k in ['present_targets','anchors_available','valid_forecasts','absent_targets','known_negative_published_anchors'])
        if CASES[r['case']]['geometry_roi']=='off' or r['model'] in ['hold','cv','cv_hist_acc','bounded_acc','huber_acc','trusted_acc']:assert compare==ref;exact+=1
    save(OUT/'score_checks.json',dict(all_coverage_identical54=True,old_baseline_or_six_history_exact_rows=exact,scoring_rows=len(a['table'])))
    summary=[]
    for case in SCORING:
        r=load(OUT/case/'regression.json');c=load(OUT/case/'completion.json')
        summary.append(dict(r,processing_median_ms=c['processing_median_ms'],physics_active=c['methods']['delayed_feedback']['physics_active'],actual_frequency_updates=c['methods']['delayed_feedback']['actual_updates']['adaptive_axis_freq']))
    save(OUT/'summary.json',dict(table=summary))
    select=[r for r in a['table'] if r['nominal_h']==.4 and CASES[r['case']]['reference']!='negative5' and r['model'] in ['trusted_acc','adaptive_axis_freq','adaptive_all_mix']]
    def tab(rr,cols):return '<table><tr>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[x],5) if isinstance(r[x],float) else r[x]))+'</td>' for x in cols)+'</tr>' for r in rr)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>57：完整轮廓与候选ROI</title><style>body{max-width:1450px;margin:28px auto;font:16px/1.8 Arial;color:#233e4c}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd8df}video{width:100%}</style><h1>57：保留全图阈值和完整轮廓，只筛选候选</h1><p>原生20ms整机检测和位置、独立10ms所有事件、全图密度分位数及连通轮廓保持；过去整机框只过滤拟合后的候选中心，之后仍是原排名/分组/转子频率及15模型。25%余量/160ms期限，父提示不晚于包开始，无未来框或补造桨心。</p>'''+tab(summary,['case','native10_ticks','body_roi_ticks','candidates_before_gate','candidates_after_gate','four_visible','four_frequency','actual_frequency_updates','processing_median_ms'])+'''<h2>相同目标的0.4秒未来端点</h2>'''+tab(select,['case','model','present_targets','valid_forecasts','mean_fde_px','joint_success'])+'''<p>原22稀疏可见框评分，不是密集ADE或真实重心误差；预测全部保存后读取标签。ROI关闭完整15精确54，两路六历史与覆盖相同。候选/频率可用性不是独立桨轴身份或RPM精度。</p><video controls src="side6/forecasts.mp4"></video><video controls src="train3/forecasts.mp4"></video><p>绿历史，紫轴向频率；active=False时回退CV。50fps为源时间，不是实时处理吞吐。</p><p><a href="analysis.json">全模型/时距评分</a> · <a href="summary.json">统计</a> · <a href="raw_prefix_checks.json">真实RAW前缀</a> · <a href="protocol.json">冻结协议</a></p><p>已有ROI/跟踪基础：<a href="https://arxiv.org/html/2604.18289v1">事件桨叶状态估计</a>、<a href="https://arxiv.org/html/2609.11920v1">EVPeriscope</a>。实际复用本地原检测、55边界、54传输和完整推理，不称新算法或作者系统完整复现。</p></html>'''
    (OUT/'第五十七阶段完整轮廓候选ROI报告.html').write_text(page,encoding='utf-8');print('Stage57 report complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','audit','prefix','score','report']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action in ['run','audit']:globals()[a.action](a.case)
    else:globals()[a.action]()
