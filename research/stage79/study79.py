"""Fixed module experiment: installed SDK frequency on causal RAW inputs."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys,time
import numpy as np
from native79 import *
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage64')]
from event_drone_forecast.raw_io import open_raw,raw_chunks
from compensation64 import PastVelocity
PREV=ROOT.parent/'output/stage78_RAW_separated_forecast_20261021'
OUT=ROOT.parent/'output/stage79_native_frequency_20261021'
load=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def save(p,q):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(q,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=str(args[0]).lower().replace('\\','/')
        if 'evaluation_data' in p or '/labels' in p or p.endswith('/analysis.json'):
            raise PermissionError('79 feature extraction must not read future truth or scores: '+p)
def sources():
    files=[Path(__file__),Path(__file__).with_name('native79.py'),Path(__file__).with_name('RELATED_WORK.md'),ROOT/'research/stage64/compensation64.py',ROOT/'src/event_drone_forecast/raw_io.py']
    return {str(p):sha(p) for p in files}
def freeze():
    if OUT.exists():
        assert (OUT/'attempt1_range_failure.json').is_file() and not (OUT/'controls79.json').exists()
        assert not any((OUT/c).exists() for c in load(PREV/'protocol.json')['cases'])
        assert not (OUT/'protocol_attempt1.json').exists()
        (OUT/'protocol_attempt1.json').write_bytes((OUT/'protocol.json').read_bytes())
    p=load(PREV/'protocol.json');cases=p['cases']
    import metavision_sdk_cv_internal as internal
    installed=[Path(internal.__file__),Path(sdk.__file__),Path('D:/Prophesee/include/metavision/sdk/cv/configs/frequency_estimation_config.h'),Path('D:/Prophesee/include/metavision/sdk/cv/events/event_frequency.h')]
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=cases,
        input_hashes={c:sha(PREV/c/'observations_forecasts.jsonl') for c in cases},raw_metadata=p['raw_metadata'],
        parameters=PARAMS,installed_sdk_hashes={str(f):sha(f) for f in installed},numpy_version=np.__version__,
        design='Three persistent image-coordinate SDK engines; strict7 past160ms velocity-compensated replay. '
            'Only saved current NN box and previously committed CV velocity select current ROI/warp. '
            'Original160ms spectral input retained for coverage comparison. No future labels read.',
        decision='Module reliability/latency experiment only. Do not connect to force by coverage alone. '
            'All synthetic negatives and all recordings retained; no parameter search.',
        default_promoted=False,attempt=2 if (OUT/'protocol_attempt1.json').is_file() else 1,
        failure_correction='First synthetic-only run stopped on SDK default7 frequency outside requested range. '
            'Retain failure evidence and apply explicit60..1200Hz selection with exclusion counts; '
            'no actual RAW or future labels were run before this revision.'))
    print('79 protocol frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for f,h in p['installed_sdk_hashes'].items():assert sha(f)==h
    for f,m in p['raw_metadata'].items():
        st=Path(f).stat();assert m==dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return p
def controls():
    verify();sys.addaudithook(guard)
    # Layout/ownership/chunking controls use known coordinates/timestamps,
    # not a filter which discards corrupt rows from the original binding.
    e,_,freq=synthetic('tone',2);a=engine(128,96,'strict7');o=a.get_empty_output_buffer()
    a.process_events(e,o);raw=o.numpy();info=o._buffer_info();q=copy_frequency(o)
    invalid=int(np.sum((raw['x']>=128)|(raw['y']>=96)|~np.isfinite(raw['frequency'])|(raw['frequency']<60)|(raw['frequency']>1200)))
    expected_xy=set(map(tuple,np.c_[e['x'],e['y']]))
    assert all(tuple(x) in expected_xy for x in np.c_[q['x'],q['y']])
    assert len(q)>0 and np.max(abs(q['frequency']-freq))<10
    b=engine(128,96,'strict7');out=b.get_empty_output_buffer();pieces=[]
    for k in range(40):pieces.append(checked_process(b,e[(e['t']>=k*10000)&(e['t']<(k+1)*10000)],out,128,96))
    split=np.concatenate(pieces)
    for field in q.dtype.names:np.testing.assert_array_equal(q[field],split[field])
    result=[]
    for kind in ['tone','moving_tone','poisson','colored','burst','switch','common_flicker']:
        for seed in range(12):
            ev,bbox,f=synthetic(kind,seed);model=NativeFeatures(128,96);final=None;first={};d=[]
            for k in range(40):
                t=(k+1)*.01;ee=ev[(ev['t']>=k*10000)&(ev['t']<(k+1)*10000)]
                model.feed(ee,t)
                if (k+1)%2:continue
                st=time.perf_counter();final=model.publish(bbox,t,[100.,0.] if kind=='moving_tone' else [0.,0.]);d.append((time.perf_counter()-st)*1000)
                for name,value in final.items():
                    if value['qualified'] and name not in first:first[name]=t
            target=f*1.5 if kind=='switch' else f
            for name,value in final.items():
                measured=value.get('frequency_hz')
                result.append(dict(kind=kind,seed=seed,method=name,accepted=value['qualified'],
                    relative_error=None if measured is None else abs(measured/target-1),first_qualified_s=first.get(name),final=value))
        print('79 synthetic',kind,flush=True)
    table=[]
    for kind in sorted({r['kind'] for r in result}):
        for method in sorted({r['method'] for r in result}):
            rr=[r for r in result if r['kind']==kind and r['method']==method]
            accepted=[r for r in rr if r['accepted']]
            table.append(dict(kind=kind,method=method,n=len(rr),accepted=len(accepted),
                accepted_and_within3percent=sum(r['relative_error']<.03 for r in accepted),
                median_first_qualified_s=float(np.median([r['first_qualified_s'] for r in rr if r['first_qualified_s'] is not None])) if any(r['first_qualified_s'] is not None for r in rr) else None))
    refused=False
    bad=np.zeros(1,sdk_base.EventCD);bad['t']=20000
    try:NativeFeatures(128,96).feed(bad,.02)
    except AssertionError:refused=True
    assert refused
    save(OUT/'controls79.json',dict(layout=dict(native_stride=list(info.strides),numpy_stride=list(raw.strides),record_bytes=info.itemsize,
        naive_invalid_rows=invalid,native_rows=len(q),known_coordinates_valid=True,known_frequency_valid=True,
        all_output_fields_exact_across_chunks=True),future_events_refused=True,table=table,rows=result,
        caveat='Known synthetic pixel-transition processes, not hardware rotor truth; colored random phase processes can be locally periodic.'))
    print('79 controls complete',flush=True)
def run(case):
    p=verify();assert (OUT/'controls79.json').is_file();cfg=p['cases'][case]
    assert sha(PREV/case/'observations_forecasts.jsonl')==p['input_hashes'][case]
    sys.addaudithook(guard);rr=rows(PREV/case/'observations_forecasts.jsonl');folder=OUT/case;folder.mkdir(exist_ok=False)
    reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();native=NativeFeatures(w,h);past=PastVelocity();timing=[];output=[]
    with (folder/'features.jsonl').open('x',encoding='utf-8') as f:
        feed_ms=0.
        for i,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
            st=time.perf_counter();native.feed(e,t);feed_ms+=(time.perf_counter()-st)*1000
            if (i+1)%2:continue
            r=rr[len(output)];assert abs(r['t']-t)<1e-8
            v,va=past.before(t-.02);body=r['body_observation'];bbox=body.get('bbox') if body.get('detected') and body.get('confidence',0)>=.45 else None
            st=time.perf_counter();features=native.publish(bbox,t,v);publish_ms=(time.perf_counter()-st)*1000
            original=r['event_activity'];q=dict(t=t,features=features,velocity_audit=va,
                original_spectral_qualified=original.get('frequency_squared') is not None,
                original_density=original.get('density'),body_detected=bbox is not None,
                timing_ms=dict(native_feed=feed_ms,publish=publish_ms))
            output.append(q);f.write(json.dumps(q,allow_nan=False)+'\n');timing.append(feed_ms+publish_ms);feed_ms=0.;past.commit(r)
    assert len(output)==len(rr)
    save(folder/'completion.json',dict(case=case,parent=len(output),native10=2*len(output),labels_read=False,
        body_supported=sum(r['body_detected'] for r in output),original_periodic_supported=sum(r['original_spectral_qualified'] for r in output),
        supported={n:sum(r['features'][n]['qualified'] for r in output) for n in output[0]['features']},
        median_module_ms=float(np.median(timing)),p95_module_ms=float(np.percentile(timing,95)),
        source_hashes_unchanged=sources()==p['source_hashes']))
    print(case,load(folder/'completion.json'),flush=True)
def summarize_all():
    p=verify();table=[load(OUT/c/'completion.json') for c in p['cases']]
    prefix=[]
    for case in ['side6','train3']:
        a=rows(OUT/case/'features.jsonl');b=rows(OUT/('prefix_'+case)/'features.jsonl')
        for x,y in zip(a,b):
            for k in ['t','features','velocity_audit','original_spectral_qualified','body_detected']:assert x[k]==y[k],(case,x['t'],k)
        prefix.append(dict(case=case,parent=len(b),full_features_exact=True))
    save(OUT/'checks79.json',dict(table=table,prefix=prefix,parent=sum(r['parent'] for r in table),native10=sum(r['native10'] for r in table),
        labels_read=False,no_new_future_forecasts=True,default_promoted=False))
    print('79 all actual RAW module checks complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','controls','run','summarize']);p.add_argument('--case');a=p.parse_args()
    if a.action=='run':run(a.case)
    elif a.action=='summarize':summarize_all()
    else:globals()[a.action]()
