from pathlib import Path
import argparse,copy,datetime,hashlib,html,importlib.util,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
import run_joint_forecast as runner
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage52'));import study52 as previous
BASE=ROOT.parent/'output';OUT=BASE/'stage53_raw_joint_physics_20261020';POS=BASE/'stage45_cuda_graph_20261019/sequence';SENSING=BASE/'stage34_slot_frequency_forecast_20261018'
CASES={c:dict(cfg,raw='E:/dataset/train/3/aligned/event.raw' if c=='train3' else ('E:/dataset/train/5/media/event.raw' if c=='negative5' else 'E:/dataset/train/6/media/event.raw'),reference=c) for c,cfg in previous.CASES.items()}
CASES.update(pilot_clear6=dict(CASES['clear6'],duration=.8),pilot_negative5=dict(CASES['negative5'],duration=.2),prefix_side6=dict(CASES['side6'],duration=4.5))
SCORING=['clear6','side6','train3','negative5'];load=previous.load;save=previous.save;sha=previous.sha
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def sources():
    result=previous.sources();result.update(runner.fingerprints());result['research/stage53/study53.py']=sha(Path(__file__));return result
def freeze():
    assert not OUT.exists();raws={}
    for cfg in CASES.values():
        p=Path(cfg['raw']);s=p.stat();raws[str(p)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),raw_metadata=raws,cases=CASES,
        precision=32,verify_frame_pixels=True,scoring_cases=SCORING,label_hashes=load(previous.OUT/'protocol.json')['label_hashes'],
        design='Actual RAW-only joint entry: unchanged47 native20 body input/GPU graph plus independent bounded RAW10 recovery/ReplaceStale frequency worker, both current cutoff only. Full three physical variants, six history coefficients exact independent filter. Actual predicted outputs first, audit against34/45/51/52 only in a separate process, then true sparse future labels.',
        regression='Pilot .8s clear+.2s empty before full13s positive+1s empty. Independent4.5s side RAW prefix includes real delayed updates. Parent/worker pixel SHA; all native10 geometry/sensing against34 and native20 geometry/pixels against45/41. Original controls preserved, no cached arrays read in runtime.',
        limits='Two RAW decoders for numerical isolation, may cost IO/CPU. Model still learned using1/2/4, selected6, weight-held3 was earlier development-seen. No independent pose/RPM/COM truth, no guarantee realtime or stable physical advantage.',
        prior_art=['https://arxiv.org/abs/2604.18289','https://docs.python.org/3/library/multiprocessing.shared_memory.html'],
        expected_sample_count=dict(full_frames=700,prefix_frames=225,pilot_frames=50)))
    print('Stage53 RAW joint protocol frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for name,metadata in p['raw_metadata'].items():
        s=Path(name).stat();assert metadata==dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return p
def run(case):
    verify();cfg=CASES[case];runner.run(cfg['raw'],OUT/case,cfg['start'],cfg['duration'],precision=32,preview=case not in ['prefix_side6','pilot_negative5'],verify_frames=True)
def audit(case):
    verify();cfg=CASES[case];reference=cfg['reference'];rr=rows(OUT/case/'observations_forecasts.jsonl');rotors=rows(OUT/case/'rotor_ticks.jsonl')
    assert len(rr)==round(cfg['duration']*50) and len(rotors)==len(rr)*2
    pos={round(r['t'],8):r for r in rows(POS/reference/'observations.jsonl')};original52={round(r['t'],8):r for r in rows(previous.OUT/reference/'forecasts.jsonl')};original51={round(r['t'],8):r for r in rows(previous.REFERENCE_OUT/reference/'forecasts.jsonl')}
    if cfg['geometry']:
        geom={round(r['t'],8):r['observation'] for r in rows(BASE/cfg['geometry'])};sensing={round(r['t'],8):r for r in rows(SENSING/reference/'sensing.jsonl')}
        for r in rotors:
            key=round(r['t'],8);assert r['observation']==geom[key],(case,r['t'],'native10 geometry differs')
            assert r['sensing']==sensing[key]['methods']['replace_stale'] and r['slot_audit']==sensing[key]['slot_audit'],(case,r['t'],'native10 sensing differs')
    images={round(r['t'],8):r for r in load(BASE/'stage41_continuous_body_forecast_20261018/inputs.json')['images'] if r['case']==reference}
    bbox_delta=[];exact_body=0;exact_forecast={m:0 for m in runner.METHODS};exact_history=0
    for r in rr:
        key=round(r['t'],8);old=pos[key];assert r['geometry20']==old['methods']['geometry']['observation'],(case,r['t'],'native20 geometry differs')
        image=cv2.imdecode(np.fromfile(str(BASE/'stage41_continuous_body_forecast_20261018'/images[key]['image']),np.uint8),cv2.IMREAD_COLOR)
        assert hashlib.sha256(image.tobytes()).hexdigest()==r['frame_sha256'],(case,r['t'],'native pixels differ')
        ob=old['methods']['body_only']['observation'];assert r['body_observation']['detected']==ob['detected']
        exact_body+=r['body_observation']==ob
        if ob['detected']:bbox_delta.append(float(np.max(abs(np.asarray(ob['bbox'])-r['body_observation']['bbox']))))
        for name in runner.METHODS:
            expected=original51[key]['frontends']['reset_all'] if name=='reset_all' else original52[key]['frontends'][name]
            exact_forecast[name]+=r['frontends'][name]==expected
        if all(r['frontends']['delayed_feedback']['coefficients'].get(m)==original52[key]['frontends']['delayed_feedback']['coefficients'].get(m) for m in runner.HISTORY):exact_history+=1
    video_frames=0
    if (OUT/case/'forecasts.mp4').exists():
        video=cv2.VideoCapture(str(OUT/case/'forecasts.mp4'))
        while True:
            ok,frame=video.read()
            if not ok:break
            video_frames+=1
        video.release();assert video_frames==len(rr)
    if reference=='negative5':assert not any(q['valid'] for r in rr for q in r['frontends'].values())
    result=dict(frames=len(rr),native10_ticks=len(rotors),native10_geometry_sensing_exact_to34=bool(cfg['geometry']),native20_geometry_and_pixels_exact_to45_41=True,
        body_observation_exact_to45=exact_body,maximum_body_bbox_difference_px=max(bbox_delta,default=0.),full_forecast_exact_counts=exact_forecast,six_history_exact_to52=exact_history,video_decoded_frames=video_frames,
        notes='Exact counts report actual arithmetic, do not assume GPU numerical identity. All saved future predictions also scored independently. Negative has no34 reference.')
    save(OUT/case/'regression.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
def prefix():
    verify();a=rows(OUT/'prefix_side6/observations_forecasts.jsonl');b=rows(OUT/'side6/observations_forecasts.jsonl');assert len(a)==225
    keys=['t','event_count','filtered_event_count','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256','frontends','lifecycle']
    for first,second in zip(a,b):
        for key in keys:assert first[key]==second[key],(first['t'],key,'RAW prefix differs')
    aa=rows(OUT/'prefix_side6/rotor_ticks.jsonl');bb=rows(OUT/'side6/rotor_ticks.jsonl');assert len(aa)==450
    for first,second in zip(aa,bb):
        for key in ['t','event_count','observation','sensing','slot_audit','eof_empty_padding']:assert first[key]==second[key],(first['t'],key)
    save(OUT/'raw_prefix_checks.json',dict(parent_ticks=225,rotor_native10_ticks=450,complete_forecasts_lifecycle_observations_pixels_exact=True,independent_actual_RAW_decode=True))
    print('Actual RAW prefix225 body/450 rotor complete results exact',flush=True)
def score():
    verify();spec=importlib.util.spec_from_file_location('score53',ROOT/'research/stage22/study22.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT;m.INPUT=OUT;m.CASES=SCORING;m.FRONTENDS=runner.METHODS;m.MODELS=runner.MODELS;m.verify=verify;m.evaluate=previous.prior.evaluate
    m.label_path=lambda c:BASE/'raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json')
    # Reuse scorer expecting forecasts.jsonl; copies are made only after RAW run.
    for c in SCORING:
        dest=OUT/c/'forecasts.jsonl';assert not dest.exists();dest.write_text('\n'.join(json.dumps(dict(t=r['t'],frontends=r['frontends']),allow_nan=False) for r in rows(OUT/c/'observations_forecasts.jsonl'))+'\n',encoding='utf-8')
    m.score()
def report():
    verify();a=load(OUT/'analysis.json');summary=[]
    for case in SCORING:
        q=load(OUT/case/'completion.json');summary.append(dict(case=case,frames=q['frames'],median_ms=q['processing_median_ms'],p95_ms=q['processing_p95_ms'],over20=q['fraction_over20ms'],wall_s=q['wall_including_cold_decode_video_s'],physical_updates=q['methods']['delayed_feedback']['actual_updates']['adaptive_axis_freq']))
    table=[r for r in a['table'] if r['case']!='negative5' and r['nominal_h']==.4 and r['model'] in ['trusted_acc','adaptive_axis_freq','adaptive_all_mix']]
    def tab(rr,cols):return '<table><tr>'+''.join('<th>'+x+'</th>' for x in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[x],5) if isinstance(r[x],float) else r[x]))+'</td>' for x in cols)+'</tr>' for r in rr)+'</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>53：真实RAW整机与桨叶联合预测</title><style>body{max-width:1300px;margin:30px auto;font:16px/1.8 Arial;color:#233e4c}video{width:100%}table{border-collapse:collapse;font-size:13px}th,td{padding:6px;border:1px solid #ccd8df}pre{white-space:pre-wrap;background:#eef3f6;padding:15px}</style><h1>53：固定参考与延迟物理学习接入真实RAW</h1><p>新run_joint_forecast.py仅输入RAW及固定本地权重：20ms整机定位/历史，独立10ms桨叶几何/逐桨频率；实际执行姿态轴、有效推力/RLS与延迟加权。原重置、固定参考及延迟反馈三路保存。两个独立RAW解码器用于隔离OpenCV几何数值状态，两个子进程只处理当前请求截止；预测入口禁止读取旧观测/频率/预测缓存和人工标签。缓存只在运行结束后的独立审计使用。</p>'''+tab(summary,['case','frames','median_ms','p95_ms','over20','wall_s','physical_updates'])+'''<p>处理计时含当前几何/图像准备/双worker等待和物理模型，排除父进程下一RAW块/JSON/视频；wall含冷启动、双RAW解码和视频。两worker可能重叠，不把分项相加当端到端。视频50fps是源时间，不代表实时吞吐。</p><h2>实际预测视频</h2><p>橙框整机；绿色历史收缩、紫色轴向+频率物理未来轨迹。预测0.1/0.2/0.4秒；未叠加未来真值。</p><video controls src="clear6/forecasts.mp4"></video><video controls src="side6/forecasts.mp4"></video><video controls src="train3/forecasts.mp4"></video><h2>真实未来0.4秒端点</h2>'''+tab(table,['case','frontend','model','present_targets','valid_forecasts','mean_fde_px','joint_success'])+'''<p>整机位置不等于真实质心，没有独立姿态/机械RPM/质量/风速真值。真实RAW整合不自动意味着超过历史模型，全部时距/覆盖和负例见分析。该数据此前开发见过，不称全流程盲测；不把像素模型称为作者三维复现。</p><h2>使用入口</h2><pre>&amp; "D:/drone forecast/.venv-event311/Scripts/python.exe" "D:/drone forecast/event_drone_forecast/run_joint_forecast.py" --raw "E:/dataset/train/6/media/event.raw" --start 108 --duration 5 --output "D:/drone forecast/output/my_joint_prediction" --preview</pre><p>输出目录需未存在；默认FP32，研究逐像素核验可加--verify-frames。原47/30入口及旧默认保留，不用G。<a href="raw_prefix_checks.json">独立RAW前缀</a> · <a href="analysis.json">全部评分</a> · <a href="protocol.json">冻结协议</a></p></html>'''
    (OUT/'第五十三阶段真实RAW联合预测报告.html').write_text(page,encoding='utf-8');save(OUT/'runtime_summary.json',dict(table=summary));print('Stage53 RAW report complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','audit','prefix','score','report']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action in ['run','audit']:globals()[a.action](a.case)
    else:globals()[a.action]()
