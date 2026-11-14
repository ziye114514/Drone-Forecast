"""Freeze before running either sequence; labels enter only the later scoring."""
from pathlib import Path
import argparse,copy,datetime,json,shutil,sys,time,zipfile
import cv2,numpy as np
from study import ROOT,OUT,FIRST,save,csvsave,digest,verify
from activity_fallback import ActivityTracker
from event_drone_forecast.config import Config
from event_drone_forecast.motion import ForecastBank,extrapolate
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.raw_io import open_raw,raw_chunks
DEST=OUT/'activity_fallback'
METHODS=('original_rotor_cv','activity_cv','rotor_or_activity_cv')

def freeze():
    verify();DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'protocol.json').exists():raise RuntimeError('Already frozen')
    files=[Path(__file__),Path(__file__).with_name('activity_fallback.py')]
    for p in files:(DEST/p.name).write_bytes(p.read_bytes())
    save(DEST/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={p.name:digest(p) for p in files},methods=list(METHODS),sequences=['1','2'],duration_s=20,
        roles={'1':'Previously inspected development train sequence','2':'Local train sequence not used to tune these rules; secondary transfer check, not official test'},
        no_label_input=True,no_parameter_tuning_between_sequences=True,
        activity_rules=dict(bin_px=2,window_ms=30,min_events_per_bin=3,min_component_bins=6,min_activity=40,
            max_extent_px=320,initial_confirmation_frames=3,reset_s=.3),
        hybrid_rule='Use original rotor position only when four rotors are currently observed and geometry_quality>.45; otherwise activity position. No pose/force invented.',
        forecasting='Same original ForecastBank CV implementation for each stream; detector-specific declared observation confidence/covariance retained.',
        evaluation='Both all-available coverage/error and common available endpoints; original provided label timestamps, no GT interpolation.',
        conclusion_limits='A generic detection fallback is an engineering baseline, not a claimed new rotor-physics prediction algorithm. Single compact active target assumption.'))
    print('Activity fallback experiment frozen',flush=True)

def check():
    verify();p=json.loads((DEST/'protocol.json').read_text())
    for name,h in p['sources'].items():assert digest(Path(__file__).with_name(name))==h
    return p

def prepare(sequence):
    folder=DEST/f'sequence{sequence}';folder.mkdir(exist_ok=True)
    if sequence=='1':return FIRST/'public_truth_substitutes/fred_sequence1/events.raw'
    raw=folder/'input/events.raw';raw.parent.mkdir(exist_ok=True)
    if not raw.exists():
        zpath=Path(r'E:\FRED')/f'{sequence}.zip'
        with zipfile.ZipFile(zpath) as z:
            names=z.namelist();info=[]
            for suffix,target in [('Event/events.raw','events.raw'),('coordinates.txt','coordinates.txt')]:
                matches=[p for p in names if p.endswith(suffix) and not (suffix=='coordinates.txt' and p.endswith('rgb_coordinates.txt'))]
                assert len(matches)==1,matches
                member=matches[0];dest=raw.parent/target
                with z.open(member) as source,dest.open('wb') as f:shutil.copyfileobj(source,f)
                info.append(dict(member=member,target=target,bytes=dest.stat().st_size,sha256=digest(dest)))
            save(raw.parent/'inventory.json',dict(archive=str(zpath),members=info))
    return raw

def run(sequence):
    protocol=check();raw=prepare(sequence);folder=DEST/f'sequence{sequence}'
    if (folder/'completion.json').exists():raise RuntimeError('Complete run exists')
    reader=open_raw(raw);height,width=reader.get_size();front=PaperPerception(width,height,PaperConfig());activity=ActivityTracker(width,height)
    banks={name:ForecastBank(Config(step_us=10000)) for name in METHODS};traces=[];observations=[];preview=[];pt=[]
    started=time.perf_counter();lastprint=started;step_times=[]
    for frame,(t,e) in enumerate(raw_chunks(reader,0,protocol['duration_s'],10000)):
        begin=time.perf_counter();o,total=front.detect(e,t);a=activity.step(e,t)
        qualified=o['detected'] and o.get('observed_rotor_count',0)==4 and o.get('geometry_quality',0)>.45
        mixed=copy.deepcopy(o if qualified else a);mixed['position_source']='four_rotors' if qualified else 'activity_only'
        obs=dict(zip(METHODS,(o,a,mixed)));record=dict(t=t,states={},active={})
        for name,bank in banks.items():
            result=bank.step(obs[name]);record['active'][name]=result is not None and result['active']
            record['states'][name]=None if result is None else result['states']['cv']
        traces.append(record);observations.append(dict(t=t,observations=obs))
        step_times.append((time.perf_counter()-begin)*1000)
        if (frame+1)%5==0:
            image=np.uint8(np.clip(np.log1p(total)/4,0,1)*255)
            preview.append(cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA));pt.append(t)
        if time.perf_counter()-lastprint>12:
            print(json.dumps(dict(sequence=sequence,t=t,phase='raw_only_activity_comparison')),flush=True);lastprint=time.perf_counter()
    for filename,records in [('forecasts.jsonl',traces),('observations.jsonl',observations)]:
        with (folder/filename).open('w',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    np.savez_compressed(folder/'preview_frames.npz',times=pt,images=preview)
    # Original stream must match the earlier v0.3 baseline on the development clip.
    if sequence=='1':
        old=[json.loads(s) for s in (raw.parent/'raw_only_baseline/forecasts.jsonl').read_text().splitlines()]
        assert len(old)==len(traces)
        for a,b in zip(old,traces):
            assert a['active']==b['active']['original_rotor_cv']
            if b['states']['original_rotor_cv'] is not None:
                for key in ('position','velocity','acceleration'):
                    np.testing.assert_array_equal(a['states']['cv'][key],b['states']['original_rotor_cv'][key])
    save(folder/'completion.json',dict(sequence=sequence,frames=len(traces),raw=str(raw),raw_sha256=digest(raw),
        inference_inputs=[str(raw)],labels_used=False,wall_s=time.perf_counter()-started,
        processing_median_ms=float(np.median(step_times)),processing_p95_ms=float(np.percentile(step_times,95)),
        detected_frames={m:sum(r['observations'][m]['detected'] for r in observations) for m in METHODS},
        no_thrust_estimation_from_activity=True))
    print('RAW comparison complete sequence'+sequence,flush=True)

def score(sequence):
    check();folder=DEST/f'sequence{sequence}';meta=json.loads((folder/'completion.json').read_text());raw=Path(meta['raw'])
    traces=[json.loads(s) for s in (folder/'forecasts.jsonl').read_text().splitlines()];times=np.array([r['t'] for r in traces])
    labels=[]
    for line in (raw.parent/'coordinates.txt').read_text().splitlines():
        stamp,values=line.split(':',1);box=np.array(list(map(float,values.split(',')[:4])))
        if 0<float(stamp)<=20:labels.append(dict(t=float(stamp),bbox=box,center=(box[:2]+box[2:])/2))
    details=[]
    for label in labels:
        for h in (.1,.2,.4):
            k=np.searchsorted(times,label['t']-h,side='right')-1
            if k<0:continue
            q=traces[k]
            for m in METHODS:
                row=dict(sequence=sequence,target_t=label['t'],anchor_t=times[k],horizon_s=h,method=m,available=q['active'][m])
                if row['available']:row['error_px']=float(np.linalg.norm(extrapolate(q['states'][m],label['t']-times[k])-label['center']))
                details.append(row)
    csvsave(folder/'endpoint_details.csv',details);metrics=[]
    for h in (.1,.2,.4):
        # Common support requires all three variants to publish, and is reported separately.
        groups={}
        for r in details:
            if r['horizon_s']==h:groups.setdefault(r['target_t'],[]).append(r)
        common={t for t,rs in groups.items() if len(rs)==3 and all(r['available'] for r in rs)}
        for m in METHODS:
            subset=[r for r in details if r['horizon_s']==h and r['method']==m];ok=[r for r in subset if r['available']];co=[r for r in ok if r['target_t'] in common]
            metrics.append(dict(method=m,horizon_s=h,targets=len(subset),available=len(ok),coverage=len(ok)/max(1,len(subset)),
                mean_error_px=float(np.mean([r['error_px'] for r in ok])) if ok else None,
                common_available=len(co),common_mean_error_px=float(np.mean([r['error_px'] for r in co])) if co else None))
    save(folder/'scores.json',dict(sequence=sequence,metrics=metrics,labels_used_in_inference=False,gt_interpolated=False))
    print(json.dumps(dict(sequence=sequence,metrics=metrics)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','score']);parser.add_argument('--sequence',choices=['1','2']);a=parser.parse_args()
    if a.action=='freeze':freeze()
    else:globals()[a.action](a.sequence)
