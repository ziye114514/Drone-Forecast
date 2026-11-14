"""Frozen ON/OFF-only rotor extraction pilot. No G, YOLO or external project writes."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.raw_io import open_raw,raw_chunks
from polarity import PolarityFrontend,METHODS,counts,tests
OUT=ROOT.parent/'output/stage8_onoff_rotors_20261017'
OWN='E:/dataset/train/6/media/event.raw'
FRED=ROOT.parent/'output/stage7_localization_budget_20261017/data'
CASES={'absent':dict(raw=OWN,start=0.,duration=2.),
       'clear':dict(raw=OWN,start=22.,duration=4.),
       'side':dict(raw=OWN,start=109.,duration=4.),
       'exit':dict(raw=OWN,start=114.5,duration=1.5),
       'fred4':dict(raw=str(FRED/'sequence4/events.raw'),start=12.,duration=4.),
       'fred6':dict(raw=str(FRED/'sequence6/events.raw'),start=0.,duration=4.)}
SNAPSHOTS={'absent':[1.],'clear':[23.,24.,25.],'side':[110.,111.,112.],
           'exit':[115.],'fred4':[13.,15.],'fred6':[1.,3.]}

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashes():
    files=[Path(__file__),Path(__file__).with_name('polarity.py'),*(ROOT/'src/event_drone_forecast').glob('*.py')]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def freeze():
    assert not (OUT/'protocol.json').exists(),'Already frozen'
    save(OUT/'module_checks.json',tests())
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=hashes(),
        cases=CASES,snapshots=SNAPSHOTS,methods=METHODS,step_s=.01,publish_s=.02,
        feature='Strict native-pixel min(ON,OFF) versus min(GaussianBlur(ON),GaussianBlur(OFF)); 3x3 sigma.8. Existing downstream3x3 blur/morphology/geometry/grouping unchanged.',
        temporal='10ms versus past30ms ON/OFF sums divided by3 to preserve10ms-equivalent event-rate units; past-only, same cutoff, up to30ms warmup.',
        thresholds='Original PaperConfig fixed; no parameter search or test tuning.',
        baseline='strict10 source adapter must reproduce unchanged PaperPerception every processed frame, before accepting any result.',
        candidate_metrics='Candidate centers inside/outside existing whole-UAV boxes are diagnostic proxies, NOT per-rotor recall or proof of true four rotors.',
        scoring='Existing sparse own labels / FRED original coordinates used only after extraction; no annotation in detector. Current position error, missing outputs, observed candidate counts and known absent-label detections.',
        scope='Small previously inspected development clips, no independent test claim. No forecast/RPM/pose accuracy improvement claim.',
        excluded='No reconstructed G, no HyperE2VID/PCBS/YOLO, no writes or code execution in D:/yolov10-main, no modification of prior src/results.',
        prior_work=['https://github.com/prgumd/EVPropNet','https://github.com/radimspetlik/effqpdet','https://docs.prophesee.ai/stable/concepts.html'],
        reuse='Installed OpenCV Gaussian/morphology/fitEllipse, unchanged existing PaperPerception geometry and grouping; custom adapter is not a novel algorithm claim.'))
    print('Frozen ON/OFF pilot',flush=True)

def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==hashes(),'Frozen source changed';return p

def run(case):
    p=verify();cfg=p['cases'][case];folder=OUT/case;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'completion.json').exists(),'Do not overwrite completed run'
    reader=open_raw(cfg['raw']);h,w=reader.get_size();fronts={m:PolarityFrontend(w,h,m) for m in METHODS}
    reference=PaperPerception(w,h,PaperConfig());history=collections.deque();snapshots=[];all_times=[];profiles={m:[] for m in METHODS}
    count=0;events_total=0;exact=0;started=time.perf_counter();last_print=started
    warm_start=max(0.,cfg['start']-.03);end=cfg['start']+cfg['duration']
    with (folder/'observations.jsonl').open('w',encoding='utf-8') as f:
        for t,e in raw_chunks(reader,warm_start,end-warm_start,10000):
            native=counts(e,w,h);history.append((t,native));events_total+=len(e)
            while history and history[0][0]<t-.020001:history.popleft()
            long=dict(total=native['total'],on=sum(r['on'] for _,r in history)/3,off=sum(r['off'] for _,r in history)/3)
            record=dict(t=t,methods={});snap=any(abs(t-s)<.0001 for s in p['snapshots'][case])
            images={}
            for m,front in fronts.items():
                prepared=native if METHODS[m]['window_ms']==10 else long
                tick=time.perf_counter();o,_=front.detect_prepared(e,t,prepared);profiles[m].append((time.perf_counter()-tick)*1000)
                record['methods'][m]=dict(observation=o,candidates=front.candidates)
                if snap:images[m]=front.feature(prepared['on'],prepared['off'])
            original,_=reference.detect(e,t)
            assert original==record['methods']['strict10']['observation'],f'Baseline mismatch at{case}:{t}'
            exact+=1
            if t>cfg['start']+1e-8 and round(t*100)%2==0:
                f.write(json.dumps(record,allow_nan=False)+'\n');count+=1
            if snap:
                snapshot=dict(t=t,on=native['on'],off=native['off'],**images)
                np.savez_compressed(folder/f'snapshot_{t:.2f}.npz',**snapshot);all_times.append(t)
            if time.perf_counter()-last_print>12:
                print(json.dumps(dict(case=case,t=t,phase='ON/OFF-only extraction')),flush=True);last_print=time.perf_counter()
    save(folder/'completion.json',dict(case=case,published=count,baseline_exact_frames=exact,snapshots=all_times,
        events_including_warmup=events_total,wall_s=time.perf_counter()-started,width=w,height=h,labels_in_inference=False,
        inputs=[cfg['raw']],G_used=False,yolov10_main_accessed=False,
        feature_detection_ms={m:dict(median=float(np.median(v)),p95=float(np.percentile(v,95))) for m,v in profiles.items()}))
    print('Completed',case,flush=True)

def labels(case):
    cfg=CASES[case];lo=cfg['start'];hi=lo+cfg['duration']
    if case.startswith('fred'):
        seq=int(case[4:]);path=ROOT.parent/f'output/stage7_localization_budget_v3_20261017/data/sequence{seq}/coordinates.txt';result=[]
        for line in path.read_text().splitlines():
            ts,vs=line.split(':',1);t=float(ts)
            if lo<t<=hi:
                box=list(map(float,vs.split(',')[:4]));result.append(dict(t=t,state='present',bbox=box))
        return result
    return [r for r in load(ROOT.parent/'output/paper_reproduction_v030/evaluation_data/train6.json')['labels'] if lo<r['t']<=hi]

def analyze():
    verify();table=[];details=[]
    for case in CASES:
        folder=OUT/case;records=[json.loads(s) for s in (folder/'observations.jsonl').read_text().splitlines()];times=np.array([r['t'] for r in records])
        for label in labels(case):
            k=int(np.searchsorted(times,label['t'],side='right')-1)
            if k<0 or label['t']-times[k]>.025:continue
            for m in METHODS:
                q=records[k]['methods'][m];o=q['observation'];r=dict(case=case,method=m,label_t=label['t'],observation_t=float(times[k]),state=label['state'],detected=o['detected'],
                    selected_count=len(o['rotors']),candidate_count=len(q['candidates']))
                if label['state']=='present':
                    box=np.array(label['bbox']);center=(box[:2]+box[2:])/2
                    inside=lambda c:box[0]<=c['cx']<=box[2] and box[1]<=c['cy']<=box[3]
                    r.update(bbox=box.tolist(),candidates_inside=sum(inside(c) for c in q['candidates']),
                        candidates_outside=sum(not inside(c) for c in q['candidates']),selected_inside=sum(inside(c) for c in o['rotors']),
                        center_error_px=float(np.linalg.norm(np.array(o['center'])-center)) if o['detected'] else None)
                details.append(r)
        for m in METHODS:
            subset=[r for r in details if r['case']==case and r['method']==m];present=[r for r in subset if r['state']=='present'];absent=[r for r in subset if r['state']=='absent']
            errors=[r['center_error_px'] for r in present if r['center_error_px'] is not None]
            table.append(dict(case=case,method=m,published_frames=len(records),detected_frames=sum(r['methods'][m]['observation']['detected'] for r in records),
                four_selected_frames=sum(len(r['methods'][m]['observation']['rotors'])==4 for r in records),
                mean_candidates_per_frame=float(np.mean([len(r['methods'][m]['candidates']) for r in records])),
                present_labels=len(present),present_detected=sum(r['detected'] for r in present),
                mean_current_error_px=float(np.mean(errors)) if errors else None,
                candidate_inside_mean=float(np.mean([r['candidates_inside'] for r in present])) if present else None,
                candidate_outside_mean=float(np.mean([r['candidates_outside'] for r in present])) if present else None,
                absent_labels=len(absent),absent_detections=sum(r['detected'] for r in absent)))
    save(OUT/'analysis.json',dict(table=table,details=details,limits='Whole-box diagnostics, NOT per-rotor true-positive metrics.'))
    print(json.dumps(table,ensure_ascii=False),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','tests','run','analyze','all']);a.add_argument('--case',choices=list(CASES));q=a.parse_args()
    if q.action=='tests':print(tests())
    elif q.action=='all':
        for case in CASES:run(case)
        analyze()
    elif q.action=='run':run(q.case)
    else:globals()[q.action]()
