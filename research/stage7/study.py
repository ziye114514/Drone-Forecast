"""Frozen FRED localization intervention. Research diagnostics, not full benchmark."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys,time,collections
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage2'))
from activity_fallback_fixed import ActivityTracker
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_config import PaperConfig
from models import MotionBank,predict,tests,PARAMS,METHODS
OUT=ROOT.parent/'output/stage7_localization_budget_v3_20261017'
FRONTS=('rotor','activity')
INPUTS=('dense_auto','matched_auto','matched_oracle','oracle_all')
HORIZONS=(.1,.2,.4)

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def hashes():
    files=[Path(__file__),Path(__file__).with_name('models.py'),ROOT/'research/stage2/activity_fallback.py',ROOT/'research/stage2/activity_fallback_fixed.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def freeze():
    assert not (OUT/'protocol.json').exists(),'Frozen already'
    checks=tests();assert ActivityTracker.__module__=='activity_fallback_fixed'
    checks['activity_dependency']='Existing Stage2 fixed reacquisition adapter, not the preserved failed original'
    # Exercise actual dependency with a confirmed target, a gap and reacquisition.
    dt=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);ev=np.zeros(30*30*4,dtype=dt)
    ev['x']=np.repeat(np.arange(30,60),120);ev['y']=np.tile(np.repeat(np.arange(30,60),4),30)
    detector=ActivityTracker(320,240)
    assert detector.step(ev,.03)['detected'] is False
    detector.step(ev,.04);assert detector.step(ev,.05)['detected']
    empty=np.zeros(0,dtype=dt);detector.step(empty,.1);detector.step(empty,.2);detector.step(empty,.4)
    detector.step(ev,.5);detector.step(ev,.51);assert detector.step(ev,.52)['detected']
    checks['actual_activity_reacquisition_test']=True;save(OUT/'module_checks.json',checks)
    base=OUT/'upstream/FRED/dataset_splits/canonical'
    splits={s:{int(x.strip().strip('/')) for x in (base/f'{s}_split.txt').read_text().splitlines() if x.strip()} for s in ['train','test']}
    role={str(i):('official_test_subset' if i in splits['test'] else 'previously_inspected_train' if i in [1,2,3] else 'additional_train_transfer') for i in range(11)}
    assert all(i in splits['train']|splits['test'] for i in range(11))
    for name in ['study.py','models.py','prepare.py']:
        p=Path(__file__).with_name(name);dest=OUT/'source_snapshot'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=hashes(),
        upstream=load(OUT/'upstream/audit.json'),roles=role,sequences=list(range(11)),
        duration_s=20.,future_visualization_s=.4,raw_step_us=10000,parameters=PARAMS,horizons_s=HORIZONS,
        frontends='Unchanged original PaperPerception; unchanged Stage2 ActivityTracker from activity_fallback_fixed.py. Separate hypotheses; no semantic claim for activity.',
        reuse='Actual cv2.KalmanFilter predict/correct; existing RAW IO, original rotor and activity front ends; Stage6 history-quadratic principle. No new KF engine.',
        inputs=dict(dense_auto='Automatic measurement stream at10ms',matched_auto='At each past label timestamp, latest automatic measurement no older than50ms',
            matched_oracle='Same update times and missing mask as matched_auto, replacing only center with current/past box center',oracle_all='All past box centers; diagnostic upper-input reference'),
        scoring='Original coordinates.txt, no interpolated_coordinates. Select last past label anchor <= target-h, allow <=75ms extra lead. Same target/anchor per all arms. No future labels in state estimation.',
        warmup_s=.5,oracle_age_limit_s=.05,lead_slack_s=.075,
        metrics='Endpoint Euclidean error, coverage, common support, median/p95, error / target box diagonal, joint coverage-and-accuracy <=.25 target diagonal. Not dense ADE.',
        tuning='No parameter search. All eleven20s intervals and all parameters fixed before predictions or forecast scoring; test8/9 never used for tuning.',
        label_limits='Legacy four-field boxes lack IDs; no identity association to tracks.txt assumed. Reset after.5s label gap. Evaluate provided box stream, not verified multi-object tracking.',
        scope='First20s per local archive. Official test SUBSET8/9 only, not full canonical FRED result. No actual pose/RPM/thrust ground truth.',
        revision='v1 stopped before scoring at out-of-interval zero-area label. v2 imported the preserved failed activity implementation instead of fixed adapter, causing reacquisition failure; stopped before test8/9. v3 reuses existing corrected dependency, with explicit dependency/reacquisition test. No predictor parameters changed. Completed v2 rotor-only observations may be reused; all activity rerun from RAW.',
        performance='Decode, rotor, activity, file IO separately measured in one research pipeline; not standalone per-detector end-to-end benchmark.'))
    print('Frozen; official test subset:',[i for i,r in role.items() if r=='official_test_subset'],flush=True)

def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==hashes(),'Source changed after freeze';return p

def run(sequence):
    p=verify();inventory=next(r for r in load(OUT/'local_inventory.json') if r['sequence']==sequence)
    folder=OUT/'runs'/f'sequence{sequence}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists():print('verified existing complete',sequence,flush=True);return
    raw=Path(inventory['raw']);assert sha(raw)==inventory['raw_sha256']
    cv2.setNumThreads(1)
    reader=open_raw(raw);height,width=reader.get_size()
    rotor=PaperPerception(width,height,PaperConfig());activity=ActivityTracker(width,height)
    cached=None;old=ROOT.parent/'output/stage7_localization_budget_v2_20261017'
    previous=old/'runs'/f'sequence{sequence}'
    if (previous/'completion.json').exists():
        oldp=load(old/'protocol.json');current=hashes()
        assert all(v==current[k] for k,v in oldp['source_hashes'].items() if k.startswith('src/'))
        cached=[json.loads(s) for s in (previous/'observations.jsonl').read_text().splitlines()]
        assert len(cached)==2000 and load(previous/'completion.json')['raw_sha256']==inventory['raw_sha256']
    started=time.perf_counter();step_times={k:[] for k in ['decode','rotor','activity','io']}
    images=[];itimes=[];last_print=started;events_count=0;frames=0;detected=collections.Counter();four=0;frequency=0
    generator=iter(raw_chunks(reader,0.,20.4,10000))
    with (folder/'observations.jsonl').open('w',encoding='utf-8') as f:
        while True:
            clock=time.perf_counter()
            try:t,e=next(generator)
            except StopIteration:break
            step_times['decode'].append((time.perf_counter()-clock)*1000)
            events_count+=len(e)
            if t<=20.+1e-8:
                clock=time.perf_counter()
                if cached is None:o,total=rotor.detect(e,t)
                else:
                    assert abs(cached[frames]['t']-t)<1e-9
                    o=cached[frames]['rotor'];total=None
                step_times['rotor'].append((time.perf_counter()-clock)*1000)
                clock=time.perf_counter();a=activity.step(e,t);step_times['activity'].append((time.perf_counter()-clock)*1000)
                clock=time.perf_counter();f.write(json.dumps(dict(t=t,rotor=o,activity=a),allow_nan=False)+'\n');step_times['io'].append((time.perf_counter()-clock)*1000)
                frames+=1
                for k,z in [('rotor',o),('activity',a)]:detected[k]+=int(z['detected'])
                four+=int(o.get('observed_rotor_count',0)==4)
                frequency+=int(o.get('frequency_quality',0)>=.35)
            else:
                pix=e['y'].astype(int)*width+e['x'];total=np.bincount(pix,minlength=height*width).reshape(height,width)
            if int(round(t*100))%10==0:
                if total is None:
                    pix=e['y'].astype(int)*width+e['x'];total=np.bincount(pix,minlength=height*width).reshape(height,width)
                image=np.uint8(np.clip(np.log1p(total)/4,0,1)*255)
                images.append(cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA));itimes.append(t)
            if time.perf_counter()-last_print>12:
                print(json.dumps(dict(sequence=sequence,t=t,phase='RAW-only inference')),flush=True);last_print=time.perf_counter()
    np.savez_compressed(folder/'event_frames.npz',times=itimes,images=images)
    save(folder/'completion.json',dict(sequence=sequence,role=p['roles'][str(sequence)],frames=frames,events_including_future_display=events_count,
        raw_sha256=inventory['raw_sha256'],input_files=[str(raw)],labels_in_inference=False,width=width,height=height,
        wall_s=time.perf_counter()-started,rotor_cached=cached is not None,rotor_cache_source=str(previous) if cached is not None else None,
        detected_frames=dict(detected),four_rotor_frames=four,frequency_quality_frames=frequency,
        timings_ms={k:dict(total=float(np.sum(v)),median=float(np.median(v)),p95=float(np.percentile(v,95))) for k,v in step_times.items()}))
    print('RAW complete',sequence,flush=True)

def read_labels(sequence):
    p=OUT/'data'/f'sequence{sequence}'/'coordinates.txt';labels=[]
    for line in p.read_text().splitlines():
        ts,vs=line.split(':',1);box=np.array(list(map(float,vs.split(',')[:4])));t=float(ts)
        if 0<=t<=20:
            assert np.isfinite(box).all() and box[2]>box[0] and box[3]>box[1]
            labels.append(dict(t=t,bbox=box.tolist(),center=((box[:2]+box[2:])/2).tolist()))
    labels.sort(key=lambda r:r['t'])
    assert len(set(r['t'] for r in labels))==len(labels),'Need multi-object ID protocol for repeated timestamp'
    return labels

def states(sequence):
    p=verify();folder=OUT/'runs'/f'sequence{sequence}'
    records=[json.loads(x) for x in (folder/'observations.jsonl').read_text().splitlines()]
    times=np.array([r['t'] for r in records]);labels=read_labels(sequence)
    assert np.all(np.diff(times)>0)
    dense={};label_states={};fresh={};oracle=MotionBank();clock=time.perf_counter()
    for front in FRONTS:
        bank=MotionBank();stream=[];seen=[];last=None
        for record in records:
            o=record[front];z=o.get('center') if o['detected'] else None
            stream.append(bank.step(record['t'],z))
            if z is not None:last=dict(t=record['t'],center=z)
            seen.append(last)
        dense[front]=stream;fresh[front]=seen
        auto=MotionBank();paired=MotionBank();rr=[]
        for label in labels:
            t=label['t'];k=int(np.searchsorted(times,t+1e-10,side='right')-1)
            recent=seen[k] if k>=0 else None
            available=recent is not None and t-recent['t']<=p['oracle_age_limit_s']+1e-10
            rr.append(dict(t=t,matched_auto=auto.step(t,recent['center'] if available else None),
                matched_oracle=paired.step(t,label['center'] if available else None),measurement_used=available,
                measurement_t=recent['t'] if available else None))
            assert not available or recent['t']<=t+1e-10
        label_states[front]=rr
    oracle_states=[oracle.step(l['t'],l['center']) for l in labels]
    # No forecast target is used until every causal state above has been generated.
    save(folder/'state_generation.json',dict(estimated_from_past_only=True,wall_s=time.perf_counter()-clock,
        no_future_interpolation=True,automatic_rate_hz=100.,oracle_uses_past_annotations_not_raw_only=True))
    return records,times,labels,dense,label_states,oracle_states

def summarize(rows):
    ok=[r for r in rows if r['error_px'] is not None];e=[r['error_px'] for r in ok]
    return dict(targets=len(rows),available=len(ok),coverage=len(ok)/max(1,len(rows)),
        mean_px=float(np.mean(e)) if e else None,median_px=float(np.median(e)) if e else None,
        p95_px=float(np.percentile(e,95)) if e else None,
        mean_normalized=float(np.mean([r['error_normalized'] for r in ok])) if ok else None,
        within_quarter_diagonal=sum(r['error_normalized']<=.25 for r in ok)/max(1,len(rows)))

def analyze(sequence):
    p=verify();folder=OUT/'runs'/f'sequence{sequence}'
    records,times,labels,dense,paired,oracle=states(sequence);lt=np.array([l['t'] for l in labels]);details=[]
    for target in labels:
        for h in HORIZONS:
            j=int(np.searchsorted(lt,target['t']-h+1e-10,side='right')-1)
            if j<0:continue
            anchor=lt[j];lead=target['t']-anchor
            if lead>h+p['lead_slack_s']+1e-8 or anchor-lt[0]<p['warmup_s']:continue
            k=int(np.searchsorted(times,anchor+1e-10,side='right')-1)
            if k<0:continue
            diagonal=float(np.linalg.norm(np.array(target['bbox'])[2:]-np.array(target['bbox'])[:2]))
            for front in FRONTS:
                sources=dict(dense_auto=(times[k],dense[front][k]),
                    matched_auto=(anchor,paired[front][j]['matched_auto']),matched_oracle=(anchor,paired[front][j]['matched_oracle']),
                    oracle_all=(anchor,oracle[j]))
                for source,(state_t,bank) in sources.items():
                    assert state_t<=anchor+1e-9
                    for method in METHODS:
                        pred=predict(bank[method],target['t']-state_t) if bank is not None else None
                        err=float(np.linalg.norm(pred-target['center'])) if pred is not None else None
                        details.append(dict(sequence=sequence,role=p['roles'][str(sequence)],front=front,input=source,method=method,
                            horizon_s=h,actual_lead_s=float(lead),anchor_t=float(anchor),target_t=target['t'],
                            prediction=None if pred is None else pred.tolist(),target_center=target['center'],target_bbox=target['bbox'],
                            error_px=err,error_normalized=err/diagonal if err is not None else None))
    with (folder/'endpoint_details.jsonl').open('w',encoding='utf-8') as f:
        for r in details:f.write(json.dumps(r,allow_nan=False)+'\n')
    metrics=[]
    for front in FRONTS:
        for source in INPUTS:
            for method in METHODS:
                for h in HORIZONS:
                    rows=[r for r in details if r['front']==front and r['input']==source and r['method']==method and r['horizon_s']==h]
                    metrics.append(dict(front=front,input=source,method=method,horizon_s=h,**summarize(rows)))
    common=[]
    for front in FRONTS:
        for method in METHODS:
            for h in HORIZONS:
                groups={}
                for r in details:
                    if r['front']==front and r['method']==method and r['horizon_s']==h:groups.setdefault(r['target_t'],{})[r['input']]=r
                support=[g for g in groups.values() if all(g[s]['error_px'] is not None for s in ['matched_auto','matched_oracle'])]
                common.append(dict(front=front,method=method,horizon_s=h,n=len(support),
                    matched_auto=summarize([g['matched_auto'] for g in support]),matched_oracle=summarize([g['matched_oracle'] for g in support])))
    save(folder/'evaluation.json',dict(sequence=sequence,metrics=metrics,common=common,label_count=len(labels),
        lead_mean_s={str(h):float(np.mean([r['actual_lead_s'] for r in details if r['horizon_s']==h])) if any(r['horizon_s']==h for r in details) else None for h in HORIZONS},
        limits='No dense trajectory ADE; diagnostic matched_oracle is not a deployable RAW-only system.'))
    print('scored sequence',sequence,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['tests','freeze','run','analyze','all']);parser.add_argument('--sequence',type=int);a=parser.parse_args()
    if a.action=='tests':print(tests())
    elif a.action=='freeze':freeze()
    elif a.action=='all':
        for i in list(range(8))+[10,8,9]:run(i);analyze(i)
    else:globals()[a.action](a.sequence)
