"""RAW deployment of exact80 with fresh NN and rotor workers."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage80')]
import run_exact_forecast as entry
import study80 as base
load=base.load;save=base.save;sha=base.sha;rows=base.rows
PREV=base.PREV;OUT=ROOT.parent/'output/stage81_RAW_exact_compute_20261021'
def cases():
    original=load(PREV/'protocol.json')['cases'];durations=dict(clear6=2.,side6=5.,train3=3.,negative5=1.,extra2=2.,prefix_side6=2.5,prefix_train3=1.2)
    return {c:dict(original[c],duration=d) for c,d in durations.items()}
def sources():
    ns=entry.build();h=ns['fingerprints']();ns.clear();h['research/stage81/study81.py']=sha(__file__);return h
def freeze():
    base.verify();assert (base.OUT/'benchmark80.json').is_file() and not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=cases(),
        raw_metadata=load(PREV/'protocol.json')['raw_metadata'],reference_hashes=load(base.OUT/'protocol.json')['input_hashes'],
        exact80_checks_sha256=sha(base.OUT/'checks80.json'),benchmark80_sha256=sha(base.OUT/'benchmark80.json'),
        design='Fresh RAW decode/current NN/current rotor workers; no inference caches. '
            'All49 and original current observations exact78 prefix; two separate actual '
            'truncation prefixes and video full decoding. Same FP32 weights/noise/horizons. '
            'Fixed short prefixes limit repeat RAW cost, no cherry-picked errors.',
        timing='Compare matching old78 prefixes as recorded, not simultaneous randomized '
            'full-pipeline timing. Controlled module benchmark is separate80.50fps video '
            'uses source time, not measured processing throughput.',default_promoted=False))
    print('81 actual RAW optimized deployment frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for f,m in p['raw_metadata'].items():
        st=Path(f).stat();assert m==dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return p
def run(case):
    p=verify();cfg=p['cases'][case];args=[sys.executable,str(ROOT/'core/run_exact_forecast.py'),'--raw',cfg['raw'],
        '--output',str(OUT/case),'--start',str(cfg['start']),'--duration',str(cfg['duration']),'--verify-frames']
    if not case.startswith('prefix') and case!='negative5':args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as f:
        r=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert r.returncode==0,(case,r.returncode);check(case);print(case,'81 actual RAW verified',flush=True)
def check(case):
    p=verify();cfg=p['cases'][case];n=round(cfg['duration']*50);rr=rows(OUT/case/'observations_forecasts.jsonl')
    reference=case.removeprefix('prefix_');path=PREV/reference/'observations_forecasts.jsonl';assert sha(path)==p['reference_hashes'][reference]
    old=rows(path)[:n];assert len(rr)==len(old)==n
    keys=['t','event_activity','body_axis','event_measurement_audit','geometry20','body_observation','observation','sensing','slot_audit',
        'body_boxes','frame_sha256','event_fields_sha256','frontends','lifecycle']
    try:
        for i,(r,o) in enumerate(zip(rr,old)):
            for k in keys:assert r[k]==o[k],(r['t'],k)
        if case.startswith('prefix'):
            full=rows(OUT/reference/'observations_forecasts.jsonl')[:n]
            for r,o in zip(rr,full):
                for k in keys:assert r[k]==o[k],(r['t'],'actualprefix',k)
    except Exception as e:
        save(OUT/case/'regression_failure81.json',dict(error=str(e),actual=r,expected=o));raise
    frames=0;video=OUT/case/'forecasts.mp4'
    if video.is_file():
        cap=cv2.VideoCapture(str(video))
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(720,1280,3);frames+=1
        cap.release();assert frames==n
    c=load(OUT/case/'completion.json');assert c['source_hashes_unchanged'] and not c['labels_read']
    proto=load(OUT/case/'protocol.json');assert proto['exact_compute_deployment_stage']==81 and not proto['native_sdk_frequency79_enabled']
    timing={}
    for name,data in [('old78_matching_prefix',old),('exact81',rr)]:
        timing[name]={k:dict(median_ms=float(np.median([r[k] for r in data])),p95_ms=float(np.percentile([r[k] for r in data],95))) for k in ['processing_ms','physics_ms','gpu_inference_ms','rotor_request_ms']}
    save(OUT/case/'regression81.json',dict(case=case,parent=n,native10=2*n,all_current_and49_exact78=True,
        actual_RAW_prefix=case.startswith('prefix'),video_frames_decoded=frames,timing=timing))
def summarize_all():
    p=verify();table=[load(OUT/c/'regression81.json') for c in p['cases']]
    save(OUT/'checks81.json',dict(table=table,parent=sum(r['parent'] for r in table),native10=sum(r['native10'] for r in table),
        video_frames_decoded=sum(r['video_frames_decoded'] for r in table),actual_RAW_only=True,all49_exact78=True,
        accuracy_unchanged=True,no_new_sparse_truth_scoring_needed=True,default_promoted=False))
    print('81 fresh RAW observations/full49/prefixes/videos complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check','summarize']);p.add_argument('--case');a=p.parse_args()
    if a.action in ['run','check']:globals()[a.action](a.case)
    elif a.action=='summarize':summarize_all()
    else:globals()[a.action]()
