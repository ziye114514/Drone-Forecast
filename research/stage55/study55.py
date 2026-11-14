from pathlib import Path
import argparse,copy,datetime,hashlib,html,importlib.util,inspect,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage54')]
import run_roi_forecast as runner
import study54 as previous
from roi55 import PARAMS
from checks55 import run as checks
from rotor_worker55 import worker_source
BASE=ROOT.parent/'output';OUT=BASE/'stage55_body_guided_rotors_20261021';REFERENCE=previous.OUT
CASES={c:dict(previous.CASES[c],geometry_roi='body') for c in ['clear6','side6','train3','negative5']}
CASES.update({f'off_{c}':dict(cfg,geometry_roi='off') for c,cfg in list(CASES.items())})
CASES.update(pilot_off_clear6=dict(CASES['off_clear6'],duration=.8),pilot_body_clear6=dict(CASES['clear6'],duration=.8),pilot_negative5=dict(CASES['negative5'],duration=.2),prefix_side6=dict(CASES['side6'],duration=4.5))
SCORING=['off_clear6','clear6','off_side6','side6','off_train3','train3','off_negative5','negative5']
load=previous.load;save=previous.save;sha=previous.sha;rows=previous.rows
def sources():
    result=previous.sources()
    for n in ['run_roi_forecast.py','research/stage55/roi55.py','research/stage55/rotor_worker55.py','research/stage55/checks55.py','research/stage55/study55.py','research/stage17/frontend.py']:result[n]=sha(ROOT/n)
    return result
def scorer_module():
    spec=importlib.util.spec_from_file_location('scorer55_original22',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def score_source():
    source=inspect.getsource(scorer_module().score);assert source.count("case=='negative5'")==2
    return source.replace("case=='negative5'","CASE_SPECS[case]['reference']=='negative5'")
def derived():return {name:hashlib.sha256(text.encode('utf-8')).hexdigest() for name,text in [('parent',runner.pipeline_source()),('worker',worker_source()),('score',score_source())]}
def freeze():
    assert not OUT.exists();check=checks();previous.verify();save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        raw_metadata=load(REFERENCE/'protocol.json')['raw_metadata'],cases=CASES,parameters=PARAMS,module_checks=check,derived_source_hashes=derived(),
        label_hashes=load(REFERENCE/'protocol.json')['label_hashes'],reference_hashes={c:{n:sha(REFERENCE/c/n) for n in ['observations_forecasts.jsonl','rotor_ticks.jsonl']} for c in ['clear6','side6','train3','negative5']},
        design='Actual RAW paired ROI-off/body-ROI, identical native20 whole-body inputs and15 models; only native10 geometry events cropped by past body-only box. Original raw frequency history/stride unchanged. No fabricated rotors, no different pose/RLS thresholds. Body hint only known by packet start. Both modes add named-field canonical SHA for validation, opaque padding SHA remains transfer-only.',
        test_order=['pilot_off_clear6','pilot_body_clear6','pilot_negative5']+SCORING+['prefix_side6'],
        scoring='Save every prediction before reading sparse future truth. Original22 math, only case alias maps negative identity for off_negative5. ROI-off complete15 must exact54; both modes native20/body/six history/validity exact54. Same future target sets; report internal availability as proxies, not true motor identity/RPM.',
        limits='Known development recordings, no pose/RPM truth, no novelty claim or guaranteed realtime. Current20ms NN never guides earlier10ms subframes; no future boxes/GT ROI. Additional canonical-field hashing debug cost included in timing.',
        prior_art=['https://arxiv.org/abs/2604.18289','https://arxiv.org/abs/2609.11920'],actual_reuse='stage17.inside,54 packet/isolated worker and complete53/52 inference math'))
    for n,text in [('parent',runner.pipeline_source()),('worker',worker_source()),('score',score_source())]:(OUT/f'derived_{n}.py').write_text(text,encoding='utf-8')
    print('Stage55 frozen: causal body ROI, unchanged frequency history and complete15 models',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['derived_source_hashes']==derived()
    for c,files in p['reference_hashes'].items():
        for name,value in files.items():assert sha(REFERENCE/c/name)==value
    for name,metadata in p['raw_metadata'].items():
        s=Path(name).stat();assert metadata==dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return p
def run(case):
    verify();cfg=CASES[case];runner.run(cfg['raw'],OUT/case,cfg['start'],cfg['duration'],precision=32,preview=case not in ['prefix_side6','pilot_negative5'],verify_frames=True,geometry_roi=cfg['geometry_roi'])
def audit(case):
    verify();cfg=CASES[case];ref=cfg['reference'];rr=rows(OUT/case/'observations_forecasts.jsonl');old={round(r['t'],8):r for r in rows(REFERENCE/ref/'observations_forecasts.jsonl')};tt=rows(OUT/case/'rotor_ticks.jsonl');old_ticks={round(r['t'],8):r for r in rows(REFERENCE/ref/'rotor_ticks.jsonl')}
    assert len(rr)==round(cfg['duration']*50) and len(tt)==len(rr)*2
    native_changed=0;cropped=0;kept=raw=0;reasons={};four_visible=four_ellipses=four_frequency=published=0
    for r in tt:
        key=round(r['t'],8);o=old_ticks[key];d=r['roi_diagnostic'];assert r['event_count']==o['event_count'] and r['geometry_event_count']==d['geometry_events']
        if d['hint_t'] is not None:assert round(d['hint_t']*1e6)<=round(r['t']*1e6)//20000*20000 if round(r['t']*1e6)%20000 else round(d['hint_t']*1e6)<=round(r['t']*1e6)-20000
        if cfg['geometry_roi']=='off':
            for field in ['observation','sensing','slot_audit']:assert r[field]==o[field],(case,r['t'],field,'off native differs')
        native_changed+=r['observation']!=o['observation'] or r['sensing']!=o['sensing'];cropped+=d['mode']=='body_roi';kept+=r['geometry_event_count'];raw+=r['event_count'];reasons[d['reason']]=reasons.get(d['reason'],0)+1
        s=r['sensing'];four_visible+=s['four_visible'];four_ellipses+=s['four_ellipses'];four_frequency+=s['four_frequencies'];published+=r['observation'].get('detected',False)
    for r in rr:
        o=old[round(r['t'],8)]
        for field in ['event_count','filtered_event_count','geometry20','body_observation','body_boxes','frame_sha256']:assert r[field]==o[field],(case,r['t'],field,'position input differs')
        q=r['frontends']['delayed_feedback'];qold=o['frontends']['delayed_feedback'];assert q['valid']==qold['valid']
        for name in ['hold','cv','cv_hist_acc','bounded_acc','huber_acc','trusted_acc']:assert q['coefficients'].get(name)==qold['coefficients'].get(name),(case,r['t'],name)
        if cfg['geometry_roi']=='off':assert q==qold and r['lifecycle']==o['lifecycle'],(case,r['t'],'off full15 differs')
        if r['body_hint_sent'] is not None:assert round(r['body_hint_sent']['t']*1e6)<=round((r['t']-.02)*1e6)
        assert len(r['event_fields_sha256'])==len(r['event_packet_sha256'])==64
    video_frames=0
    if (OUT/case/'forecasts.mp4').exists():
        cap=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'))
        while True:
            ok,frame=cap.read()
            if not ok:break
            video_frames+=1
        cap.release();assert video_frames==len(rr)
    if ref=='negative5':assert not any(r['observation'].get('detected') or r['frontends']['delayed_feedback']['valid'] for r in rr)
    summary=dict(case=case,mode=cfg['geometry_roi'],parent_frames=len(rr),native10_ticks=len(tt),raw_events=raw,geometry_events=kept,geometry_events_fraction=kept/max(raw,1),body_roi_ticks=cropped,
        geometry_published=published,four_visible=four_visible,four_ellipses=four_ellipses,four_frequency=four_frequency,native_changed_ticks=int(native_changed),roi_reasons=reasons,
        native20_body_pixels_six_history_and_validity_exact54=True,off_complete15_exact54=cfg['geometry_roi']=='off',hint_causality_checked=True,video_decoded_frames=video_frames,
        interpretation='Four-visible/frequency are internal proxy counts, not certified actual motors or RPM truth.')
    save(OUT/case/'regression.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
def prefix():
    verify();a=rows(OUT/'prefix_side6/observations_forecasts.jsonl');b=rows(OUT/'side6/observations_forecasts.jsonl');assert len(a)==225
    keys=['t','event_count','filtered_event_count','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256','frontends','lifecycle','event_fields_sha256','event_packet_nbytes','body_hint_sent','roi_diagnostic']
    for x,y in zip(a,b):
        for key in keys:assert x[key]==y[key],(x['t'],key,'prefix differs')
    a=rows(OUT/'prefix_side6/rotor_ticks.jsonl');b=rows(OUT/'side6/rotor_ticks.jsonl');assert len(a)==450
    for x,y in zip(a,b):
        for key in ['t','event_count','geometry_event_count','observation','sensing','slot_audit','roi_diagnostic']:assert x[key]==y[key],(x['t'],key)
    save(OUT/'raw_prefix_checks.json',dict(parent_ticks=225,native10_ticks=450,canonical_all_named_event_fields_hash_exact=True,complete_forecasts_lifecycle_ROI_hint_exact=True,
        per_transfer_opaque_byte_checksum_preserved=True,cross_decode_opaque_padding_sha_not_required=True,actual_RAW_prefix=True));print('ROI actual RAW prefix225/450, named fields and complete models exact',flush=True)
def score():
    verify();m=scorer_module();namespace=dict(m.__dict__);namespace.update(OUT=OUT,INPUT=OUT,CASES=SCORING,CASE_SPECS=CASES,FRONTENDS=['delayed_feedback'],MODELS=previous.previous.runner.MODELS,verify=verify,evaluate=previous.previous.previous.prior.evaluate)
    namespace['label_path']=lambda c:BASE/'raw_only_project/evaluation_data'/('train3.json' if CASES[c]['reference']=='train3' else 'train6.json')
    for c in SCORING:
        p=OUT/c/'forecasts.jsonl';assert not p.exists();p.write_text('\n'.join(json.dumps(dict(t=r['t'],frontends=r['frontends']),allow_nan=False) for r in rows(OUT/c/'observations_forecasts.jsonl'))+'\n',encoding='utf-8')
    exec(compile(score_source(),'<55 original22 score alias only>','exec'),namespace);namespace['score']()
def report():
    verify();a=load(OUT/'analysis.json');old=load(REFERENCE/'analysis.json');refs={(r['case'],r['model'],r['nominal_h']):r for r in old['table'] if r['case'] in ['clear6','side6','train3','negative5']}
    exact=0
    for r in a['table']:
        compare=copy.deepcopy(r);cfg=CASES[r['case']];compare['case']=cfg['reference'];expected=refs[(compare['case'],compare['model'],compare['nominal_h'])]
        for field in ['present_targets','anchors_available','valid_forecasts','absent_targets','known_negative_published_anchors']:assert compare[field]==expected[field]
        if cfg['geometry_roi']=='off' or r['model'] in ['hold','cv','cv_hist_acc','bounded_acc','huber_acc','trusted_acc']:assert compare==expected;exact+=1
    save(OUT/'score_checks.json',dict(old_baseline_or_six_history_exact_rows=exact,all_coverage_identical54=True,scoring_rows=len(a['table']),case_alias_only=True))
    summary=[]
    for c in SCORING:
        r=load(OUT/c/'regression.json');s=load(OUT/c/'completion.json');summary.append(dict(r,processing_median_ms=s['processing_median_ms'],processing_p95_ms=s['processing_p95_ms'],physics_active=s['methods']['delayed_feedback']['physics_active'],actual_frequency_updates=s['methods']['delayed_feedback']['actual_updates']['adaptive_axis_freq']))
    save(OUT/'summary.json',dict(table=summary));metrics=[r for r in a['table'] if r['nominal_h']==.4 and CASES[r['case']]['reference']!='negative5' and r['model'] in ['trusted_acc','adaptive_axis_freq','adaptive_all_mix']]
    def tab(rr,cols):return '<table><tr>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[x],5) if isinstance(r[x],float) else r[x]))+'</td>' for x in cols)+'</tr>' for r in rr)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>55：过去整机框引导桨盘提取</title><style>body{max-width:1450px;margin:30px auto;font:16px/1.8 Arial;color:#233e4c}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd8df}video{width:100%}</style><h1>55：只改变当前桨盘几何检索区域</h1><p>复用17 inside及既有25%框余量/160ms期限。独立10ms桨叶进程仅使用父进程在本20ms包开始前已经取得的NN body-only框；不会用TT的图像指导TT-.01的几何。过期/缺失/无效框回到原全局。原始频率历史、抽取预算、FFT/相位/转子关联、RLS与全部15模型保持不变；没有制造缺失桨心。整机NN和20ms几何始终全图，同六历史位置和覆盖固定，故未来差异只来自桨叶物理输入及其在线权重。</p>'''+tab(summary,['case','mode','native10_ticks','geometry_events_fraction','body_roi_ticks','geometry_published','four_visible','four_frequency','physics_active','actual_frequency_updates','processing_median_ms'])+'''<p>四可见/四频率是内部可用性，不是独立真实转轴/RPM精度。额外命名字段哈希只用于调试前缀，忽略无语义padding，传输内完整字节校验仍保留。计时包含该研究开销，跨运行负载未控制。</p><h2>同位置0.4秒真实未来端点</h2>'''+tab(metrics,['case','model','present_targets','valid_forecasts','mean_fde_px','joint_success'])+'''<p>所有预测保存后才读取人工未来稀疏框，原22计分数学不变，只把off_negative5别名映射到实际空场身份。ROI关完整15精确54，两路原生20ms、NN、六历史和覆盖精确54；模型参数没有按结果调节。该数据已开发见过，不是全流程盲测，框中心不是真实质心。</p><h2>实际RAW画面</h2><p>绿：历史收缩；紫：轴向频率（active=False时回退CV）。完整数据/全时距评分含负结果，不挑成功帧。</p><video controls src="side6/forecasts.mp4"></video><video controls src="train3/forecasts.mp4"></video><p><a href="summary.json">可用性与耗时</a> · <a href="analysis.json">全部未来评分</a> · <a href="raw_prefix_checks.json">真实RAW前缀</a> · <a href="protocol.json">冻结协议</a></p><p>相关原始研究<a href="https://arxiv.org/abs/2604.18289">事件桨叶状态估计</a>和<a href="https://arxiv.org/abs/2609.11920">EVPeriscope</a>已有检测/ROI/几何追踪基础；本轮为本项目已实现模块整合消融，不称新算法或作者系统完整复现。</p></html>'''
    (OUT/'第五十五阶段整机框引导桨盘报告.html').write_text(page,encoding='utf-8');print('Stage55 report complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','audit','prefix','score','report']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action in ['run','audit']:globals()[a.action](a.case)
    else:globals()[a.action]()
