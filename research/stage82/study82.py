"""RAW current-activity parity and independent streamed-selection benchmark."""
from pathlib import Path
import argparse,datetime,json,sys,time
import numpy as np
from stream82 import *
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage79'),str(ROOT/'research/stage64')]
import study79 as utils
from compensation64 import PastVelocity
from event_drone_forecast.raw_io import open_raw,raw_chunks
load=utils.load;save=utils.save;sha=utils.sha;rows=utils.rows
PREV=utils.PREV;OUT=ROOT.parent/'output/stage82_streamed_window_20261021'
def sources():
    return {str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('stream82.py'),Path(__file__).with_name('RELATED_WORK.md'),ROOT/'research/stage80/exact80.py']}
def freeze():
    assert not OUT.exists();p=load(PREV/'protocol.json')
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        frozen78_sources=p['source_hashes'],cases=p['cases'],raw_metadata=p['raw_metadata'],parameters=PARAMS,controls=controls(),
        input_hashes={c:sha(PREV/c/'observations_forecasts.jsonl') for c in p['cases']},
        benchmark_cases=['clear6','side6','extra2'],benchmark_duration_s=2.,
        design='RAW past events with saved current body ROI/past CV velocity; '
            'every activity field exact78. No future label/score reads. Genuine '
            'truncated RAW prefixes. No new accuracy or end-to-end realtime claim.',
        benchmark='Old80 vs streaming82 activity only, alternating tick order; '
            'wait for81 complete before regression/benchmark to avoid CPU contention.',default_promoted=False))
    print('82 exact streaming protocol frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for f,h in p['frozen78_sources'].items():assert sha(ROOT/f)==h
    for f,m in p['raw_metadata'].items():
        st=Path(f).stat();assert m==dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return p
def run(case):
    p=verify();assert (ROOT.parent/'output/stage81_RAW_exact_compute_20261021/checks81.json').is_file()
    assert sha(PREV/case/'observations_forecasts.jsonl')==p['input_hashes'][case]
    sys.addaudithook(utils.guard);cfg=p['cases'][case];rr=rows(PREV/case/'observations_forecasts.jsonl');folder=OUT/case;folder.mkdir(exist_ok=False)
    reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();activity=StreamingActivity(w,h);past=PastVelocity();packets=[];n=0;timing=[]
    with (folder/'activity.jsonl').open('x',encoding='utf-8') as f:
        for i,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
            packets.append(e.copy())
            if (i+1)%2:continue
            r=rr[n];assert abs(r['t']-t)<1e-8;v,va=past.before(t-.02);body=r['body_observation']
            bbox=body.get('bbox') if body.get('detected') and body.get('confidence',0)>=.45 else None
            packet=np.concatenate(packets);packets=[];st=time.perf_counter();q=activity.step(packet,bbox,t,v);timing.append((time.perf_counter()-st)*1000)
            assert q==r['event_activity'] and va==r['event_measurement_audit'],(case,t,'current activity')
            f.write(json.dumps(dict(t=t,activity=q,velocity_audit=va),allow_nan=False)+'\n');past.commit(r);n+=1
    assert n==len(rr)
    summary=dict(case=case,parent=n,native10=2*n,all_activity_fields_exact78=True,labels_read=False,
        module_median_ms=float(np.median(timing)),module_p95_ms=float(np.percentile(timing,95)))
    save(folder/'completion.json',summary);print(summary,flush=True)
def summarize_all():
    p=verify();table=[load(OUT/c/'completion.json') for c in p['cases']];prefix=[]
    for case in ['side6','train3']:
        full=rows(OUT/case/'activity.jsonl');short=rows(OUT/('prefix_'+case)/'activity.jsonl');assert full[:len(short)]==short
        prefix.append(dict(case=case,parent=len(short),activity_fields_exact=True))
    save(OUT/'checks82.json',dict(table=table,prefix=prefix,parent=sum(r['parent'] for r in table),native10=sum(r['native10'] for r in table),
        current_exact78=True,no_new_future_forecasts=True,default_promoted=False))
    print('82 all current activity and real RAW prefixes complete',flush=True)
def benchmark():
    p=verify();assert (OUT/'checks82.json').is_file();sys.addaudithook(utils.guard);table=[]
    for case in p['benchmark_cases']:
        cfg=p['cases'][case];rr=rows(PREV/case/'observations_forecasts.jsonl');reader=open_raw(Path(cfg['raw']));h,w=reader.get_size()
        models=dict(exact80=exact80.CachedActivity(w,h),stream82=StreamingActivity(w,h));past=PastVelocity();packets=[];n=0;timing=[]
        for i,(t,e) in enumerate(raw_chunks(reader,cfg['start'],p['benchmark_duration_s'],10000)):
            packets.append(e.copy())
            if (i+1)%2:continue
            r=rr[n];v,va=past.before(t-.02);body=r['body_observation'];bbox=body.get('bbox') if body.get('detected') and body.get('confidence',0)>=.45 else None
            packet=np.concatenate(packets);packets=[];order=['exact80','stream82'] if n%2==0 else ['stream82','exact80'];elapsed={}
            for name in order:
                st=time.perf_counter();q=models[name].step(packet,bbox,t,v);elapsed[name]=(time.perf_counter()-st)*1000
                assert q==r['event_activity'],(case,t,name)
            timing.append(elapsed);past.commit(r);n+=1
        row=dict(case=case,parent=n,timing_rows=timing,
            median_paired_ratio=float(np.median([r['exact80']/r['stream82'] for r in timing])))
        for name in models:row[name]=dict(median_ms=float(np.median([r[name] for r in timing])),p95_ms=float(np.percentile([r[name] for r in timing],95)))
        table.append(row);print({k:v for k,v in row.items() if k!='timing_rows'},flush=True)
    save(OUT/'benchmark82.json',dict(table=table,activity_exact78=True,
        scope='Activity only, excludes packet decode/current NN/support map/body axis/all forecasting; no claim of full pipeline speed.'))
    print('82 paired activity benchmark complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','summarize','benchmark']);p.add_argument('--case');a=p.parse_args()
    if a.action=='run':run(a.case)
    elif a.action=='summarize':summarize_all()
    else:globals()[a.action]()
