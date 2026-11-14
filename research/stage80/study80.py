"""Exact RAW module regression and paired performance experiment."""
from pathlib import Path
import argparse,cProfile,datetime,hashlib,io,json,pstats,sys,time
import numpy as np
from exact80 import *
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage79'),str(ROOT/'research/stage62')]
import study79 as utils
from independent62 import Managed62,Selector
from event_drone_forecast.raw_io import open_raw,raw_chunks
load=utils.load;save=utils.save;sha=utils.sha;rows=utils.rows
PREV=utils.PREV;OUT=ROOT.parent/'output/stage80_exact_compute_20261021'

def sources():
    return {str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('exact80.py'),Path(__file__).with_name('RELATED_WORK.md')]}
def freeze():
    assert not OUT.exists();p=load(PREV/'protocol.json')
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        frozen78_sources=p['source_hashes'],cases=p['cases'],input_hashes={c:sha(PREV/c/'observations_forecasts.jsonl') for c in p['cases']},
        raw_metadata=p['raw_metadata'],parameters=PARAMS,controls=controls(),
        benchmark_cases=['clear6','side6','extra2'],benchmark_duration_s=2.,
        design='Same RAW/current NN and stored rotor observations as78; reconstruct62 '
            'core15 and all extra34 online. Check every current feature and full49 '
            'against frozen78; no future labels or future scores. Real RAW prefixes.',
        benchmark='Separate fresh paired old/new same ticks, alternating execution order. '
            'Run only after79 completes. Decode/core62/equality/profile overhead excluded '
            'from measured module time; not end-to-end realtime. cProfile separate.',
        default_promoted=False))
    print('80 exact computation protocol frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for f,h in p['frozen78_sources'].items():assert sha(ROOT/f)==h,(f,'frozen78')
    for f,m in p['raw_metadata'].items():
        st=Path(f).stat();assert m==dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return p
def base_forecast(r,managed,selector):
    q,d=managed.step(r['t'],r['observation'],r['sensing'],r['slot_audit']['calibration_epoch'])
    q=selector.step(q,r['observation']);q['association_diagnostic']=r['frontends']['delayed_feedback']['association_diagnostic']
    for m,c in q['coefficients'].items():assert c==r['frontends']['delayed_feedback']['coefficients'][m],(r['t'],m,'base15')
    return q
def evaluate_tick(model,r,e,q,quality):
    st=time.perf_counter();a,b,v=model.measure(r['t'],e,r['body_observation']);measure=(time.perf_counter()-st)*1000
    st=time.perf_counter();new=model.forecast(q,r['observation'],quality);forecast=(time.perf_counter()-st)*1000
    model.commit(r['t'],r['body_observation'],new)
    assert a==r['event_activity'] and b==r['body_axis'] and v==r['event_measurement_audit'],(r['t'],'current')
    assert new==r['frontends']['delayed_feedback'],(r['t'],'all49')
    return new,dict(measure=measure,forecast=forecast,total=measure+forecast)
def run(case):
    p=verify();assert sha(PREV/case/'observations_forecasts.jsonl')==p['input_hashes'][case]
    sys.addaudithook(utils.guard);cfg=p['cases'][case];rr=rows(PREV/case/'observations_forecasts.jsonl');folder=OUT/case;folder.mkdir(exist_ok=False)
    reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();model=OnlineExtras(w,h);managed=Managed62('axis_learn');selector=Selector()
    packets=[];n=0;timing=[];previous=None;snapshot=None;digests=[]
    with (folder/'forecasts.jsonl').open('x',encoding='utf-8') as f:
        for i,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
            packets.append(e.copy())
            if (i+1)%2:continue
            r=rr[n];assert abs(r['t']-t)<1e-8;q=base_forecast(r,managed,selector);before=json.dumps(q,sort_keys=True)
            value,elapsed=evaluate_tick(model,r,np.concatenate(packets),q,managed.predictor.axis.quality);packets=[]
            assert json.dumps(q,sort_keys=True)==before,(t,'input_mutated')
            if previous is not None:assert json.dumps(previous,sort_keys=True)==snapshot,(t,'past_output_mutated')
            previous=value;snapshot=json.dumps(value,sort_keys=True);timing.append(elapsed);n+=1
            digest=hashlib.sha256(snapshot.encode()).hexdigest();digests.append(digest)
            f.write(json.dumps(dict(t=t,frontends=dict(event_force=value)),allow_nan=False)+'\n')
    assert n==len(rr)
    summary=dict(case=case,parent=n,native10=2*n,current_exact78=True,all49_exact78=True,input_unmutated=True,
        previous_outputs_unmutated=True,labels_read=False,forecast_sha256=hashlib.sha256(''.join(digests).encode()).hexdigest(),
        module_median_ms=float(np.median([x['total'] for x in timing])),module_p95_ms=float(np.percentile([x['total'] for x in timing],95)))
    save(folder/'completion.json',summary);print(summary,flush=True)
def summarize_all():
    p=verify();table=[load(OUT/c/'completion.json') for c in p['cases']];prefix=[]
    for case in ['side6','train3']:
        full=rows(OUT/case/'forecasts.jsonl');short=rows(OUT/('prefix_'+case)/'forecasts.jsonl');assert full[:len(short)]==short
        prefix.append(dict(case=case,parent=len(short),all49_exact=True))
    save(OUT/'checks80.json',dict(table=table,prefix=prefix,parent=sum(x['parent'] for x in table),native10=sum(x['native10'] for x in table),
        all49_and_current_exact78=True,labels_read=False,new_accuracy_claim=False,default_promoted=False))
    print('80 all actual RAW module regression and prefixes complete',flush=True)
def benchmark():
    p=verify();assert (utils.OUT/'checks79.json').is_file() and (OUT/'checks80.json').is_file()
    sys.addaudithook(utils.guard);results=[]
    for case in p['benchmark_cases']:
        cfg=p['cases'][case];rr=rows(PREV/case/'observations_forecasts.jsonl')
        reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();old=OriginalExtras(w,h);new=OnlineExtras(w,h)
        managed=Managed62('axis_learn');selector=Selector();packets=[];n=0;timing=[]
        for i,(t,e) in enumerate(raw_chunks(reader,cfg['start'],p['benchmark_duration_s'],10000)):
            packets.append(e.copy())
            if (i+1)%2:continue
            r=rr[n];q=base_forecast(r,managed,selector);packet=np.concatenate(packets);packets=[];times={}
            order=[('old78',old),('exact80',new)] if n%2==0 else [('exact80',new),('old78',old)]
            for name,model in order:
                _,times[name]=evaluate_tick(model,r,packet,q,managed.predictor.axis.quality)
            timing.append(times);n+=1
        summary=dict(case=case,parent=n,timing_rows=timing)
        for name in ['old78','exact80']:
            summary[name]={k:dict(median_ms=float(np.median([r[name][k] for r in timing])),p95_ms=float(np.percentile([r[name][k] for r in timing],95))) for k in ['measure','forecast','total']}
        summary['median_paired_ratio']=float(np.median([r['old78']['total']/r['exact80']['total'] for r in timing]))
        results.append(summary);print(case,{n:summary[n] for n in ['old78','exact80','median_paired_ratio']},flush=True)
    # Profile a separate unmeasured fresh forecast pass using already causal
    # recorded input. Results below do not enter the paired wall-clock table.
    for name,cls in [('old78',OriginalExtras),('exact80',OnlineExtras)]:
        case=p['benchmark_cases'][0];rr=rows(PREV/case/'observations_forecasts.jsonl')[:50];cfg=p['cases'][case]
        reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();model=cls(w,h);managed=Managed62('axis_learn');selector=Selector();packets=[];n=0;prof=cProfile.Profile()
        for i,(t,e) in enumerate(raw_chunks(reader,cfg['start'],1.,10000)):
            packets.append(e.copy())
            if (i+1)%2:continue
            r=rr[n];q=base_forecast(r,managed,selector)
            prof.enable();evaluate_tick(model,r,np.concatenate(packets),q,managed.predictor.axis.quality);prof.disable();packets=[];n+=1
        prof.dump_stats(str(OUT/(name+'_profile.pstats')));stream=io.StringIO();pstats.Stats(prof,stream=stream).sort_stats('cumulative').print_stats(40)
        (OUT/(name+'_profile.txt')).write_text(stream.getvalue(),encoding='utf-8')
    save(OUT/'benchmark80.json',dict(table=results,old_and_new_outputs_exact78=True,paired_order_alternating=True,
        concurrent_project_experiment='79 complete before this benchmark; no GPU detector worker used.',
        scope='Measured event activity/body-axis + extra34 forecasting only, all49 retained; '
              'excludes decode/core62/NN/workers/JSON/equality checks. Not full pipeline realtime.'))
    print('80 paired benchmark and separate profile complete',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','summarize','benchmark']);p.add_argument('--case');a=p.parse_args()
    if a.action=='run':run(a.case)
    elif a.action=='summarize':summarize_all()
    else:globals()[a.action]()
