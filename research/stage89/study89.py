"""Actual SDK reuse and paper-style histogram on fixed88 FRED ROIs and RAW."""
from pathlib import Path
import argparse,datetime,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage79'),str(ROOT/'research/stage88'),str(ROOT/'src')]
from native79 import engine,checked_process,summarize,synthetic
from hist89 import PeriodHistogram,PARAMS
from inventory88 import save,load,sha
from event_drone_forecast.raw_io import open_raw,raw_chunks
BASE=ROOT.parent/'output/stage88_causal_event_controls_20261022'
OUT=ROOT.parent/'output/stage89_small_target_periods_20261022'


def sources():
    return {str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('hist89.py'),Path(__file__).with_name('RELATED_WORK.md'),
        ROOT/'research/stage79/native79.py',ROOT/'src/event_drone_forecast/raw_io.py']}


def freeze():
    assert not OUT.exists()
    save(OUT/'protocol89.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,
        fred_cases=[0,4,6,10,8,9],start_s=20.,duration_s=30.,step_us=20000,
        reference_protocol88_sha256=sha(BASE/'protocol88.json'),native_methods=['default7','strict7'],
        synthetic_kinds=['tone','moving_tone','poisson','colored','burst','switch','common_flicker'],seeds=list(range(6)),
        decision='Module diagnosis only, do not attach unverified higher-coverage inputs to force. No threshold search.',
        interpretation='ON/OFF duty cycle means histogram frequency may be double full-cycle brightness flicker. '
            'Synthetic frequency error reported against expected2f for interval proxy and f for SDK, not true mechanical RPM.',
        no_future_truth_read=True,default_promoted=False))
    print('89 fixed frequency module protocol frozen',flush=True)


def verify():
    p=load(OUT/'protocol89.json');assert p['source_hashes']==sources();return p


def run_stream(e,roi):
    h,w=96,128;hist=PeriodHistogram(w,h);algos={n:engine(w,h,n) for n in ['default7','strict7']};outputs={n:a.get_empty_output_buffer() for n,a in algos.items()};final={}
    for j in range(20):
        t=(j+1)*.02;ev=e[(e['t']>=round((t-.02)*1e6))&(e['t']<round(t*1e6))]
        value=hist.step(ev,roi,t)
        final={name:dict(qualified=value['raw_qualified'] if name=='paper_argmax' else value['guarded_qualified'],frequency_hz=value['frequency_hz']) for name in ['paper_argmax','guarded_hist']}
        for n,a in algos.items():final[n]=summarize(checked_process(a,ev,outputs[n],w,h),roi,t,w,h)
    return final


def controls():
    p=verify();assert not (OUT/'synthetic89.json').exists();details=[]
    for kind in p['synthetic_kinds']:
        for seed in p['seeds']:
            e,roi,f=synthetic(kind,seed);values=run_stream(e,roi)
            for method,value in values.items():
                target=f*(1.5 if kind=='switch' else 1.)*(2 if method in ['paper_argmax','guarded_hist'] else 1)
                hz=value.get('frequency_hz');details.append(dict(kind=kind,seed=seed,method=method,accepted=bool(value['qualified']),
                    expected_proxy_hz=target,relative_proxy_error=None if hz is None else abs(hz/target-1)))
        print('89 synthetic',kind,flush=True)
    save(OUT/'synthetic89.json',dict(details=details,not_real_RPM_accuracy=True))


def run(seq):
    verify();assert (OUT/'synthetic89.json').exists();dest=OUT/f'sequence{seq}';dest.mkdir(parents=True,exist_ok=False)
    inv=next(r for r in load(BASE/'inventory88.json')['sequences'] if r['sequence']==seq)
    assert sha(inv['raw'])==inv['raw_sha256']
    #88's input audit contains past ROI only; no future target or scoring file.
    with (BASE/'fred'/f'sequence{seq}'/'forecasts.jsonl').open(encoding='utf-8') as f:reference=[json.loads(s) for s in f]
    reader=open_raw(inv['raw']);h,w=reader.get_size();hist=PeriodHistogram(w,h)
    algos={n:engine(w,h,n) for n in ['default7','strict7']};outputs={n:a.get_empty_output_buffer() for n,a in algos.items()}
    counts={k:0 for k in ['paper_argmax','guarded_hist','default7','strict7','original_spectrum']};n=0;timing=[];beg=time.perf_counter()
    with (dest/'candidates.jsonl').open('x',encoding='utf-8') as f:
        for j,(t,e) in enumerate(raw_chunks(reader,20.,30.,20000)):
            r=reference[j];assert abs(t-r['t'])<1e-8;box=r['input_audit']['bbox'];start=time.perf_counter()
            value=hist.step(e,box,t);v={'paper_argmax':dict(qualified=value['raw_qualified'],frequency_hz=value['frequency_hz']),
                'guarded_hist':dict(qualified=value['guarded_qualified'],frequency_hz=value['frequency_hz'])}
            for name,a in algos.items():v[name]=summarize(checked_process(a,e,outputs[name],w,h),box,t,w,h)
            ms=(time.perf_counter()-start)*1000;timing.append(ms)
            for k,q in v.items():counts[k]+=bool(q['qualified'])
            counts['original_spectrum']+=r['input_audit']['periodic_candidates']>0;n+=1
            f.write(json.dumps(dict(t=t,methods=v,histogram=value,processing_ms=ms),allow_nan=False)+'\n')
    save(dest/'completion.json',dict(sequence=seq,ticks=n,qualified_ticks=counts,module_median_ms=float(np.median(timing)),
        module_p95_ms=float(np.percentile(timing,95)),wall_s=time.perf_counter()-beg,RAW_only_deployment=False,
        past_provided_ROI=True,no_position_scoring=True,no_thrust_input_enabled=True,source_hashes_unchanged=sources()==load(OUT/'protocol89.json')['source_hashes']))
    print(seq,counts,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','controls','run']);ap.add_argument('--sequence',type=int);a=ap.parse_args()
    if a.action=='run':run(a.sequence)
    else:globals()[a.action]()
