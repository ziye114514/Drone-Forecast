from pathlib import Path
import argparse,copy,datetime,hashlib,html,importlib.util,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research/stage53')]
import run_deploy_forecast as runner
import study53 as previous
from checks54 import run as checks
from rotor_worker54 import worker_source
BASE=ROOT.parent/'output';OUT=BASE/'stage54_single_decode_deploy_20261021';REFERENCE=previous.OUT
CASES={c:dict(previous.CASES[c],transport='shared') for c in previous.SCORING}
CASES.update(dual_side6=dict(CASES['side6'],transport='dual'),
    pilot_dual_clear6=dict(CASES['clear6'],transport='dual',duration=.8),pilot_shared_clear6=dict(CASES['clear6'],duration=.8),
    pilot_negative5=dict(CASES['negative5'],duration=.2),prefix_side6=dict(CASES['side6'],duration=4.5))
SCORING=['clear6','side6','train3','negative5','dual_side6'];load=previous.load;save=previous.save;sha=previous.sha;rows=previous.rows
def sources():
    result=previous.sources()
    for name in ['run_deploy_forecast.py','research/stage54/packets54.py','research/stage54/rotor_worker54.py','research/stage54/checks54.py','research/stage54/study54.py']:result[name]=sha(ROOT/name)
    return result
def derived():return {name:hashlib.sha256(source.encode('utf-8')).hexdigest() for name,source in [('dual',runner.pipeline_source('dual')),('shared',runner.pipeline_source('shared')),('worker',worker_source())]}
def freeze():
    assert not OUT.exists();check=checks();previous.verify();save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        raw_metadata=load(REFERENCE/'protocol.json')['raw_metadata'],cases=CASES,scoring_cases=SCORING,label_hashes=load(REFERENCE/'protocol.json')['label_hashes'],
        parameters=dict(precision=32,event_capacity_mib=32,verify_frame_and_event_pixels=True,physical_models=15,final_group_only=True,standalone_history_regression_kept=True),module_checks=check,derived_source_hashes=derived(),
        reference_hashes={c:{name:sha(REFERENCE/c/name) for name in ['observations_forecasts.jsonl','rotor_ticks.jsonl']} for c in previous.SCORING},
        design='Preserve all original53 feature/physics arithmetic and final15 models. Dual mode removes only two redundant research groups. Shared mode also reuses one RAW20 decode, sends complete structured native bytes with dtype offsets+itemsize and ack, splits exactly10ms in isolated worker. No raw input decoder/path in that worker. No cache/truth in runtime; separate full current/forecast audit to53 after each run.',
        test_order=['pilot_dual_clear6','pilot_shared_clear6','pilot_negative5','clear6','side6','train3','negative5','dual_side6','prefix_side6'],
        regression='All observations, native10 geometry/frequency, final15 forecasts and lifecycle must exactly match53. Independent4.5s true RAW prefix; actual sparse future endpoints after all forecasts saved. No prediction semantic reduction.',
        timing_limits='Cross-run machine load uncontrolled. Shared verification additionally hashes raw event payload; report overhead. Separately report53 three-group,54 dual one-group,54 shared one-group; do not infer strict acceleration from different scopes.',
        prior_art=['https://docs.python.org/3.11/library/multiprocessing.shared_memory.html','https://numpy.org/doc/stable/reference/arrays.dtypes.html']))
    for name,text in [('dual',runner.pipeline_source('dual')),('shared',runner.pipeline_source('shared')),('worker',worker_source())]:(OUT/f'derived_{name}.py').write_text(text,encoding='utf-8')
    print('Stage54 frozen: native event packet controls passed; all15 models retained',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources() and p['derived_source_hashes']==derived()
    for c,items in p['reference_hashes'].items():
        for name,value in items.items():assert sha(REFERENCE/c/name)==value
    for name,metadata in p['raw_metadata'].items():
        st=Path(name).stat();assert metadata==dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return p
def run(case):
    verify();cfg=CASES[case];runner.run(cfg['raw'],OUT/case,cfg['start'],cfg['duration'],precision=32,preview=case not in ['prefix_side6','pilot_negative5'],verify_frames=True,transport=cfg['transport'])
def audit(case):
    verify();cfg=CASES[case];reference=cfg['reference'];rr=rows(OUT/case/'observations_forecasts.jsonl');original={round(r['t'],8):r for r in rows(REFERENCE/reference/'observations_forecasts.jsonl')};rotors=rows(OUT/case/'rotor_ticks.jsonl');old_rotors={round(r['t'],8):r for r in rows(REFERENCE/reference/'rotor_ticks.jsonl')}
    assert len(rr)==round(cfg['duration']*50) and len(rotors)==len(rr)*2
    keys=['t','event_count','filtered_event_count','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256']
    for r in rr:
        old=original[round(r['t'],8)]
        for key in keys:assert r[key]==old[key],(case,r['t'],key,'parent differs')
        assert list(r['frontends'])==['delayed_feedback'] and r['frontends']['delayed_feedback']==old['frontends']['delayed_feedback'],(case,r['t'],'full15 forecast differs')
        assert r['lifecycle']['delayed_feedback']==old['lifecycle']['delayed_feedback'],(case,r['t'],'lifecycle differs')
        if cfg['transport']=='shared':assert len(r['event_packet_sha256'])==64
        else:assert r['event_packet_sha256'] is None
    for r in rotors:
        old=old_rotors[round(r['t'],8)]
        for key in ['t','event_count','observation','sensing','slot_audit','eof_empty_padding']:assert r[key]==old[key],(case,r['t'],key,'native10 differs')
    for j,r in enumerate(rr):assert r['event_count']==sum(q['event_count'] for q in rotors[j*2:j*2+2]),(case,r['t'],'packet lost events')
    video_frames=0
    if (OUT/case/'forecasts.mp4').exists():
        cap=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'))
        while True:
            ok,frame=cap.read()
            if not ok:break
            video_frames+=1
        cap.release();assert video_frames==len(rr)
    if reference=='negative5':assert not any(r['observation'].get('detected') or r['frontends']['delayed_feedback']['valid'] for r in rr)
    save(OUT/case/'regression.json',dict(frames=len(rr),native10_ticks=len(rotors),all_parent_observation_pixels_boxes_exact_to53=True,
        complete_final15_forecast_and_lifecycle_exact_to53=True,native10_geometry_and_sensing_exact_to53=True,event_count_conserved=True,shared_event_sha_verified=cfg['transport']=='shared',video_decoded_frames=video_frames))
    print(case,'all current data, native10, complete15 forecasts and lifecycle exact53; frames='+str(len(rr)),flush=True)
def prefix():
    verify();aa=rows(OUT/'prefix_side6/observations_forecasts.jsonl');bb=rows(OUT/'side6/observations_forecasts.jsonl');assert len(aa)==225
    keys=['t','event_count','filtered_event_count','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256','frontends','lifecycle','event_packet_sha256','event_packet_nbytes']
    for a,b in zip(aa,bb):
        for key in keys:assert a[key]==b[key],(a['t'],key,'prefix differs')
    a=rows(OUT/'prefix_side6/rotor_ticks.jsonl');b=rows(OUT/'side6/rotor_ticks.jsonl');assert len(a)==450
    for x,y in zip(a,b):
        for key in ['t','event_count','observation','sensing','slot_audit']:assert x[key]==y[key],(x['t'],key)
    save(OUT/'raw_prefix_checks.json',dict(parent_ticks=225,native10_ticks=450,actual_RAW_decode=True,complete_data_forecasts_lifecycle_exact=True,shared_payload_bytes_hash_exact=True));print('True RAW shared prefix225/450 exact',flush=True)
def score():
    verify();spec=importlib.util.spec_from_file_location('score54',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=OUT;m.INPUT=OUT;m.CASES=SCORING;m.FRONTENDS=['delayed_feedback'];m.MODELS=previous.runner.MODELS;m.verify=verify;m.evaluate=previous.previous.prior.evaluate
    m.label_path=lambda c:BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json')
    for c in SCORING:
        p=OUT/c/'forecasts.jsonl';assert not p.exists();p.write_text('\n'.join(json.dumps(dict(t=r['t'],frontends=r['frontends']),allow_nan=False) for r in rows(OUT/c/'observations_forecasts.jsonl'))+'\n',encoding='utf-8')
    m.score()
def report():
    verify();a=load(OUT/'analysis.json');old=load(REFERENCE/'analysis.json');reference={(r['case'],r['model'],r['nominal_h']):r for r in old['table'] if r['frontend']=='delayed_feedback'}
    for r in a['table']:
        compare=copy.deepcopy(r);compare['case']=CASES[r['case']]['reference'];assert compare==reference[(compare['case'],compare['model'],compare['nominal_h'])]
    save(OUT/'score_regression.json',dict(full_table_rows_exact_to53=len(a['table']),all_horizons=[.1,.2,.4],coverage_and_joint_success_unchanged=True))
    summary=[];components=[]
    for c in SCORING:
        s=load(OUT/c/'completion.json');cfg=CASES[c];rr=rows(OUT/c/'observations_forecasts.jsonl')
        summary.append(dict(case=c,transport=cfg['transport'],frames=s['frames'],median_ms=s['processing_median_ms'],p95_ms=s['processing_p95_ms'],wall_s=s['wall_including_cold_decode_video_s'],maximum_event_packet_bytes=s['maximum_event_packet_bytes'],updates=s['methods']['delayed_feedback']['actual_updates']['adaptive_axis_freq']))
        for field in ['parent_geometry_ms','image_preparation_ms','communication_ms','physics_ms','event_packet_publish_ms']:
            values=[r[field] for r in rr];components.append(dict(case=c,field=field,median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95))))
    save(OUT/'runtime_summary.json',dict(table=summary,components=components))
    def tab(rr,cols):return '<table><tr>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[x],4) if isinstance(r[x],float) else r[x]))+'</td>' for x in cols)+'</tr>' for r in rr)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>54：单次RAW解码与完整物理部署</title><style>body{max-width:1300px;margin:30px auto;font:16px/1.8 Arial;color:#233e4c}video{width:100%}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd8df;padding:6px}pre{white-space:pre-wrap;background:#eef3f6;padding:15px}</style><h1>54：保留完整桨叶物理，减少重复研究计算与RAW解码</h1><p>新run_deploy_forecast.py保留最终delayed_feedback完整15模型（六历史、四物理、五组合）及独立历史回归，省去重复的reset/stable两组研究对照。dual仍两解码器；shared只由父进程解码一次，将完整原生结构字节交给独立10ms worker，保持OpenCV状态隔离。字段名/类型/offset/itemsize与事件顺序保留，32MiB有界缓冲、id/ack背压，不采样、不丢包；超容量明确失败。只处理截止前事件，空子桶也按10ms推进。</p>'''+tab(summary,['case','transport','frames','median_ms','p95_ms','wall_s','maximum_event_packet_bytes','updates'])+'''<p>shared研究模式额外做双端原始事件SHA，所有模式仍做图像SHA；此开销包含在计时中。跨次负载未严格控制。53三组与54单组的差别含研究计算减少，54dual/shared才是同完整物理范围的传输对照；不把不同范围直接宣传为算法加速。</p><h2>分项耗时</h2>'''+tab(components,['case','field','median_ms','p95_ms'])+'''<p>communication包含事件发布及两worker等待，worker可能重叠；分项中位数不能相加作为总中位数。处理计时仍不含父下一RAW块、JSON/视频，wall含冷启动/解码/写盘。视频50fps为源时间，是否实时看实际20ms预算。</p><h2>预测质量</h2><p>所有当前图像/几何/频率、最终15模型预测及生命周期精确53，全部未来0.1/.2/.4评分、覆盖、联合成功保持。此轮只改善工程入口，不宣称新准确率优势；侧视物理仍未胜历史。实际RAW独立前缀和空场另有证据。缺四、错桨与姿态质量瓶颈仍待NN引导提取解决。active=False时紫色物理回退CV。</p><video controls src="side6/forecasts.mp4"></video><video controls src="train3/forecasts.mp4"></video><h2>使用</h2><pre>&amp; "D:/drone forecast/.venv-event311/Scripts/python.exe" "D:/drone forecast/event_drone_forecast/run_deploy_forecast.py" --raw "E:/dataset/train/6/media/event.raw" --start 108 --duration 5 --output "D:/drone forecast/output/my_deploy_prediction" --preview</pre><p>默认shared/FP32；--transport dual保留单分支两解码对照；--verify-frames同时核验图像/事件双端SHA；事件容量可用--event-capacity-mib配置，超限不会降采样。旧入口/结果保留，RAW与固定模型以外无需质量/尺寸/风速参数。不用G。</p><p>复用<a href="https://docs.python.org/3.11/library/multiprocessing.shared_memory.html">Python共享内存</a>与<a href="https://numpy.org/doc/stable/reference/arrays.dtypes.html">NumPy结构dtype</a>，并执行原53/52全部物理算术，非新估计算法。</p><p><a href="raw_prefix_checks.json">真实RAW前缀</a> · <a href="score_regression.json">全部未来评分回归</a> · <a href="protocol.json">冻结协议与传输控制</a></p></html>'''
    (OUT/'第五十四阶段单次解码完整物理部署报告.html').write_text(page,encoding='utf-8');print('Stage54 report complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','audit','prefix','score','report']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action in ['run','audit']:globals()[a.action](a.case)
    else:globals()[a.action]()
