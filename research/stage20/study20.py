"""Sparse cache equivalence/performance study; no target algorithm change."""
from pathlib import Path
import argparse,collections,copy,datetime,hashlib,json,time
import numpy as np
from sparse20 import ROOT,PARAMS,SparseWindow,SparseFrontend,inc
from checks20 import run as checks
from event_drone_forecast.raw_io import open_raw,raw_chunks

OUT=ROOT.parent/'output/stage20_sparse_cache_20261018'
BASE=ROOT.parent/'output/stage18_incremental_track_20261018'
CASES={'clear6':dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=5.),
    'side6':dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=12.),
    'review4':dict(raw='E:/dataset/train/4/aligned/event.raw',start=74.,duration=10.),
    'negative5':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=8.)}
BENCH={'clear6':23.,'side6':110.,'review4':75.,'negative5':2.}


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    def scalar(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(type(v).__name__)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=scalar),encoding='utf-8')


def sources():
    ff=[Path(__file__),Path(__file__).with_name('sparse20.py'),Path(__file__).with_name('checks20.py')]
    result={str(p.relative_to(ROOT)):sha(p) for p in ff}
    for p,v in load(BASE/'protocol.json')['source_hashes'].items():
        assert sha(ROOT/p)==v,(p,'old source changed');result[p]=v
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks())
    raw={}
    for f in set(c['raw'] for c in CASES.values()):
        p=Path(f);st=p.stat()
        with p.open('rb') as stream:head=hashlib.sha256(stream.read(65536)).hexdigest()
        raw[str(p)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,first64k_sha256=head,full_sha_computed=False)
        assert raw[str(p)]==load(BASE/'protocol.json')['raw_metadata'][str(p)]
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,
        cases=CASES,raw_metadata=raw,benchmark=BENCH,trials=7,cache_hashes={c:sha(BASE/c/'observations.jsonl') for c in CASES},
        design='Only histogram storage/update changes: disjoint active-block20ms chunks instead of dense14400x400 ring. Same100ms/250us spectrum and strict global geometry. Assert full observation equality after inference.',
        benchmark_scope='Dense18 vs sparse20 incremental update+FFT;7alternating paired trials, state cloning excluded both. Separate whole-frontend timings and research wall.',
        selection='Same35s development/cross-recording/negative data; not blind, no post-score tuning.',
        scope='Current observations only; no detection improvement claim by numeric equivalence, no future/RPM accuracy claim. No G or access to yolov10-main.'))
    print('Stage20 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def paired(previous,current,earlier,t):
    dense=inc.IncrementalSpectrum(previous.w,previous.h)
    for timestamp,e in earlier:dense.update(e,timestamp)
    trials=[]
    for i in range(7):
        dd=copy.deepcopy(dense);ss=copy.deepcopy(previous);q={};measure={}
        for name,obj in ([('dense',dd),('sparse',ss)] if i%2==0 else [('sparse',ss),('dense',dd)]):
            begin=time.perf_counter();obj.update(current,t);q[name]=obj.step(None,t);measure[name]=(time.perf_counter()-begin)*1000
        for k in ['quality','hz','mask']:np.testing.assert_array_equal(q['dense'][k],q['sparse'][k])
        for k in ['regions','events','active_blocks','eligible_blocks']:assert q['dense'][k]==q['sparse'][k]
        trials.append(measure)
    return dict(t=t,current_events=len(current),window_events=int(ss.totals.sum()),trials_ms=trials,
        dense_median_ms=float(np.median([r['dense'] for r in trials])),sparse_median_ms=float(np.median([r['sparse'] for r in trials])),
        dense_hist_bytes=dd.ring.nbytes+dd.totals.nbytes,sparse_hist_bytes=ss.allocated_histogram_bytes(),spectrum_exact=True)


def run(case):
    protocol=verify();cfg=CASES[case];folder=OUT/case
    assert not (folder/'observations.jsonl').exists();folder.mkdir(parents=True,exist_ok=True)
    assert sha(BASE/case/'observations.jsonl')==protocol['cache_hashes'][case]
    cache={round(r['t'],8):r['methods']['incremental_strict']['observation'] for r in map(json.loads,(BASE/case/'observations.jsonl').read_text().splitlines())}
    began=time.perf_counter();reader=open_raw(cfg['raw']);h,w=reader.get_size();model=SparseFrontend(w,h)
    history=collections.deque();start=max(0,cfg['start']-.1);end=cfg['start']+cfg['duration'];n=0;profiles=[];bench=[];last_print=began;prefix=[];expected=[]
    with (folder/'observations.jsonl').open('x',encoding='utf-8') as stream:
        for t,e in raw_chunks(reader,start,end-start,20000):
            warming=t-start<.1-1e-8;selected=abs(t-BENCH[case])<1e-8
            previous=copy.deepcopy(model.spectral) if selected else None
            earlier=list(history) if selected else None
            q=model.advance(e,t,not warming)
            if not warming:profiles.append(q['timing_ms']['total'])
            if t>cfg['start']+1e-8:
                assert q['observation']==cache[round(t,8)],(case,t,'observation changed')
                stream.write(json.dumps(dict(t=t,**q),allow_nan=False)+'\n');n+=1
            if case=='clear6' and t<=cfg['start']+.2+1e-8:prefix.append((t,e.copy(),warming));expected.append(q['observation'])
            if selected:bench.append(paired(previous,e,earlier,t))
            history.append((t,e))
            while history and history[0][0]<=t-.1+1e-8:history.popleft()
            if time.perf_counter()-last_print>15:print(json.dumps(dict(case=case,t=t)),flush=True);last_print=time.perf_counter()
    wall=time.perf_counter()-began;equal=0
    if prefix:
        twin=SparseFrontend(w,h)
        for (t,e,warming),reference in zip(prefix,expected):assert twin.advance(e,t,not warming)['observation']==reference;equal+=1
    assert n==round(cfg['duration']/.02)
    save(folder/'paired_benchmark.json',bench)
    save(folder/'completion.json',dict(case=case,frames=n,exact_vs_stage18=n,prefix_exact=equal,source_unchanged=sources()==protocol['source_hashes'],
        frontend_median_ms=float(np.median(profiles)),frontend_p95_ms=float(np.percentile(profiles,95)),research_wall_s=wall,
        scope='Research wall includes seek/decode/JSON/paired benchmarks. Frontend time includes ingest+FFT+geometry, excludes decode/JSON.'))
    print('Completed '+case,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','all','run']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action=='all':
        for case in CASES:run(case)
    elif a.action=='run':run(a.case)
    else:freeze()
