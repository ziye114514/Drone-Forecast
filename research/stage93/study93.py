"""Actual RAW integration/regression, separately processes each CLI run."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage88'),str(ROOT/'research/stage93')]
from inventory88 import save,load,sha
from online93 import NAMES
import run_complementary_forecast as entry
OUT=ROOT.parent/'output/stage93_RAW_complementary_20261022'
OLD=ROOT.parent/'output/stage78_RAW_separated_forecast_20261021'
CACHE=ROOT.parent/'output/stage92_complementary_router_20261022'
rows=lambda p:[json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]


def sources():
    ns=entry.build();h=ns['fingerprints']();ns.clear();h['research/stage93/study93.py']=sha(__file__);return h


def freeze():
    assert not OUT.exists();old=load(OLD/'protocol.json');cases={c:old['cases'][c] for c in ['clear6','side6','train3']}
    cases['prefix_clear6']=dict(cases['clear6'],duration=.8);cases['prefix_side6']=dict(cases['side6'],duration=2.)
    save(OUT/'protocol93.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=cases,
        raw_metadata=old['raw_metadata'],old_input_sha256={c:sha(OLD/c/'observations_forecasts.jsonl') for c in ['clear6','side6','train3']},
        cache92_sha256={c:sha(CACHE/'self'/c/'forecasts.jsonl') for c in ['clear6','side6','train3']},
        design='Real RAW/NN/rotor loop. All original49/current states exact78; six new coefficients exact92 including learning weights. '
            'Actual separate RAW truncated prefixes, no inference cache or labels. No promotion on mixed cached scores.',
        default_promoted=False))
    print('93 actual RAW55 integration frozen',flush=True)


def verify():
    p=load(OUT/'protocol93.json');assert p['source_hashes']==sources()
    for path,meta in p['raw_metadata'].items():
        st=Path(path).stat();assert meta==dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return p


def check(case):
    p=verify();ref=case.removeprefix('prefix_');cfg=p['cases'][case];n=round(cfg['duration']*50)
    actual=rows(OUT/case/'observations_forecasts.jsonl');old=rows(OLD/ref/'observations_forecasts.jsonl')[:n]
    cached=rows(CACHE/'self'/ref/'forecasts.jsonl')[:n];assert len(actual)==len(old)==len(cached)==n
    current_keys=['t','event_activity','body_axis','event_measurement_audit','geometry20','body_observation','observation','sensing','slot_audit','body_boxes','frame_sha256','event_fields_sha256','lifecycle']
    for r,o,c in zip(actual,old,cached):
        for key in current_keys:assert r[key]==o[key],(case,r['t'],key)
        q=r['frontends']['delayed_feedback'];oldq=o['frontends']['delayed_feedback']
        for key,value in oldq.items():
            if key in ['coefficients','predictions']:
                for m,v in value.items():assert q[key][m]==v,(case,r['t'],key,m)
            else:assert q[key]==value,(case,r['t'],key)
        if q['valid']:
            assert len(q['coefficients'])==55
            assert q['coefficients']['kinematic_train6400']==c['coefficients']['kf_tuned']
            for k,v in NAMES.items():assert q['coefficients'][v]==c['coefficients'][k],(case,r['t'],v)
        assert q['complement_diagnostic']==c['diagnostics']
    if case.startswith('prefix_'):
        full=rows(OUT/ref/'observations_forecasts.jsonl')[:n]
        for r,o in zip(actual,full):assert r['frontends']['delayed_feedback']==o['frontends']['delayed_feedback']
    completion=load(OUT/case/'completion.json');assert completion['source_hashes_unchanged'] and not completion['labels_read']
    decoded=0;video=OUT/case/'forecasts.mp4'
    if video.exists():
        cap=cv2.VideoCapture(str(video));writer=cv2.VideoWriter(str(OUT/case/'complement_forecasts.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50.,(1280,720))
        assert writer.isOpened()
        while True:
            ok,image=cap.read()
            if not ok:break
            r=actual[decoded];q=r['frontends']['delayed_feedback']
            if q['valid']:
                points=[q['predictions']['complement_fallback'][str(h)] for h in [.1,.2,.4]]
                xy=np.array(points)
                if np.isfinite(xy).all() and abs(xy).max()<100000:
                    cv2.polylines(image,[np.int32(xy)],False,(255,255,0),2)
                    cv2.circle(image,tuple(np.int32(xy[-1])),6,(255,255,0),-1)
            cv2.putText(image,'cyan=new fallback router | yellow=original primary',(15,690),0,.6,(255,255,0),2)
            writer.write(image);decoded+=1
        cap.release();writer.release();assert decoded==n
        cap=cv2.VideoCapture(str(OUT/case/'complement_forecasts.mp4'));verified=0
        while True:
            ok,image=cap.read()
            if not ok:break
            assert image.shape==(720,1280,3);verified+=1
        cap.release();assert verified==n
    result=dict(case=case,parent=n,native10=2*n,current_and_original49_exact78=True,new6_and_learning_exact92=True,
        actual_RAW_prefix=case.startswith('prefix_'),video_frames_decoded=decoded,labels_read=False,default_promoted=False)
    save(OUT/case/'regression93.json',result);print(case,'93 real RAW verified',flush=True)


def run(case):
    p=verify();cfg=p['cases'][case]
    args=[sys.executable,str(ROOT/'core/run_complementary_forecast.py'),'--raw',cfg['raw'],'--output',str(OUT/case),
        '--start',str(cfg['start']),'--duration',str(cfg['duration']),'--verify-frames']
    if not case.startswith('prefix_'):args.append('--preview')
    with (OUT/(case+'_parent.log')).open('x',encoding='utf-8') as f:
        result=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.returncode==0,(case,result.returncode);check(case)


def summarize():
    p=verify();table=[load(OUT/c/'regression93.json') for c in p['cases']]
    save(OUT/'checks93.json',dict(records=table,parent=sum(r['parent'] for r in table),native10=sum(r['native10'] for r in table),
        video_frames_decoded=sum(r['video_frames_decoded'] for r in table),actual_RAW=True,all_original49_exact78=True,
        all_new6_exact92=True,actual_prefixes=2,labels_read=False,default_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check','summarize']);p.add_argument('--case');a=p.parse_args()
    if a.action in ['run','check']:globals()[a.action](a.case)
    else:globals()[a.action]()
