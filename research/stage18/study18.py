"""Frozen35s controlled study: numeric-equivalent optimization, then publisher ablation."""
from pathlib import Path
import argparse,collections,copy,datetime,hashlib,json,time
import numpy as np
from frontend18 import IncrementalFrontend,IncrementalSpectrum,TrackPublisher,PARAMS,ROOT,old
from checks18 import run as checks
from event_drone_forecast.raw_io import open_raw,raw_chunks

OUT=ROOT.parent/'output/stage18_incremental_track_20261018'
BASE=ROOT.parent/'output/stage17_sparse_validated_roi_20261018'
CASES={'clear6':dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=5.),
    'side6':dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=12.),
    'review4':dict(raw='E:/dataset/train/4/aligned/event.raw',start=74.,duration=10.),
    'negative5':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=8.)}
SNAPS={'clear6':[23.,24.,25.],'side6':[110.,112.,117.28,117.94,118.,119.94],'review4':[75.,78.,83.],'negative5':[2.,6.]}
BENCH={'clear6':23.,'side6':110.,'review4':75.}


def save(path,x):
    def scalar(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(type(v).__name__)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=scalar),encoding='utf-8')


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    previous=load(BASE/'protocol.json')['source_hashes']
    files=[Path(__file__),Path(__file__).with_name('frontend18.py'),Path(__file__).with_name('checks18.py')]
    result={str(p.relative_to(ROOT)):sha(p) for p in files}
    for path,value in previous.items():
        assert sha(ROOT/path)==value,('prior source changed',path)
        result[path]=value
    return result


def freeze():
    assert not (OUT/'protocol.json').exists(),'Use a new experiment folder'
    module=checks();save(OUT/'module_checks.json',module)
    metadata={}
    for raw in set(c['raw'] for c in CASES.values()):
        path=Path(raw);st=path.stat()
        with path.open('rb') as f:head=hashlib.sha256(f.read(65536)).hexdigest()
        metadata[str(path)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,first64k_sha256=head,full_sha_computed=False)
        assert metadata[str(path)]==load(BASE/'protocol.json')['raw_metadata'][str(path)],'RAW metadata changed'
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,cases=CASES,
        snapshots=SNAPS,paired_spectral_benchmark=BENCH,paired_trials=7,warmup_s=.1,raw_metadata=metadata,
        cache_hashes={case:sha(BASE/case/'observations.jsonl') for case in CASES},
        methods={'cached_stage17_global':'Frozen cached global frontend; verification/deployment reference, no runtime speed comparison.',
            'incremental_strict':'Same global geometry, spectrum and3-frame gate; assert exact current observation versus stage17 global at every output tick.',
            'incremental_track':'Identical current proposal, only replaces external publication with confirmed track latch<=160ms, gated position/scale, reinitialization3ticks; no coast measurements.'},
        design='Single shared geometry/proposal stream; publication variants cannot affect it. No ROI this round. Same cutoffs20ms; old cache assertions occur after inference and never alter it.',
        timing='Every new tick records ingest+spectrum+geometry; shared computation is charged to both publisher variants.7paired spectral trials at23/110/75s include new20ms ingest+FFT vs old full100ms rebuild+FFT. Research wall contains both publishers+JSON+benchmarks; not a single pipeline real-time proof.',
        scope='Detection/position publication only, no mechanical RPM or future prediction accuracy claim. No G or access to yolov10-main.',
        selection='Reuse previously scored35s development/cross-recording/negative cases; not blind. Fixed parameters before new outputs. No post-score tuning.',
        prior_art=[dict(url='https://github.com/tryolabs/norfair/blob/master/norfair/tracker.py',use='Initialization delay and persistent hit-counter semantics reviewed; existing stage9 geometry/state reused. This wrapper does not execute Norfair.'),
            dict(url='https://docs.prophesee.ai/stable/api/python/core/bindings.html',use='SDK rolling event buffer reviewed; retains events rather than this exact250us block histogram. Existing RawReader reused, custom ring is a narrow statistics adapter.'),
            dict(url='https://github.com/radimspetlik/effqpdet',use='Existing periodic evidence branch reused; original C++ not built/run.')]))
    print('Stage18 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'],'Frozen source mismatch';return p


def paired(previous,current,history,t):
    # State copies and input concatenation are outside both timing intervals.
    control=old.SparseSpectrum(previous.w,previous.h);trials=[]
    for i in range(7):
        fresh=copy.deepcopy(previous);measurements={};answers={}
        for method in (['old','incremental'] if i%2==0 else ['incremental','old']):
            start=time.perf_counter()
            if method=='old':q=control.step(history,t)
            else:fresh.update(current,t);q=fresh.step(None,t)
            measurements[method]=(time.perf_counter()-start)*1000;answers[method]=q
        for k in ['quality','hz','mask']:np.testing.assert_array_equal(answers['old'][k],answers['incremental'][k])
        for k in ['regions','events','active_blocks','eligible_blocks']:assert answers['old'][k]==answers['incremental'][k]
        trials.append(measurements)
    return dict(t=t,current_events=len(current),history_events=len(history),trials_ms=trials,
        old_median_ms=float(np.median([x['old'] for x in trials])),incremental_median_ms=float(np.median([x['incremental'] for x in trials])),
        results_exact=True,scope='Spectrum only, new ingest included; not an end-to-end benchmark. Copies outside timing; alternating order.')


def run(case):
    protocol=verify();cfg=CASES[case];folder=OUT/case
    assert not (folder/'observations.jsonl').exists(),'Refuse to overwrite previous/partial inference'
    cache_path=BASE/case/'observations.jsonl';assert sha(cache_path)==protocol['cache_hashes'][case]
    cache={round(r['t'],8):r['methods']['global'] for r in map(json.loads,cache_path.read_text().splitlines())}
    folder.mkdir(parents=True,exist_ok=True);began=time.perf_counter();reader=open_raw(cfg['raw']);h,w=reader.get_size()
    model=IncrementalFrontend(w,h);publisher=TrackPublisher();history=collections.deque()
    start=max(0.,cfg['start']-.1);end=cfg['start']+cfg['duration'];n=equal=0;last_print=began;bench=[];prefix=[];expected=[];durations=[]
    with (folder/'observations.jsonl').open('x',encoding='utf-8') as stream:
        for t,e in raw_chunks(reader,start,end-start,PARAMS['step_us']):
            history.append((t,e))
            while history and history[0][0]<=t-.1+1e-8:history.popleft()
            warming=t-start<.1-1e-8;is_bench=case in BENCH and abs(t-BENCH[case])<1e-8
            before=copy.deepcopy(model.spectral) if is_bench else None
            q=model.advance(e,t,not warming);tick=time.perf_counter();tracked=publisher.apply(q['proposal'],t)
            publication_ms=(time.perf_counter()-tick)*1000
            if not warming:durations.append(q['timing_ms']['total']+publication_ms)
            row=dict(t=t,methods=dict(incremental_strict=q,incremental_track=tracked),raw_events_current=len(e),
                raw_events_history=int(model.spectral.totals.sum()),publication_ms=publication_ms)
            if t>cfg['start']+1e-8:
                assert q['observation']==cache[round(t,8)]['observation'],(case,t,'optimized observation differs')
                equal+=1;n+=1;stream.write(json.dumps(row,allow_nan=False)+'\n')
            if case=='clear6' and t<=cfg['start']+.2+1e-8:
                prefix.append((t,e.copy(),warming));expected.append((q['observation'],tracked['observation']))
            if any(abs(t-s)<1e-8 for s in SNAPS[case]):
                p=e['y'].astype(np.int32)*w+e['x'];total=np.bincount(p,minlength=w*h).reshape(h,w)
                np.savez_compressed(folder/f'snapshot_{t:.2f}.npz',total=total)
            if is_bench:
                bench.append(paired(before,e,np.concatenate([x for _,x in history]),t))
            if time.perf_counter()-last_print>15:print(json.dumps(dict(case=case,t=t)),flush=True);last_print=time.perf_counter()
    raw_wall=time.perf_counter()-began;prefix_equal=0
    if prefix:
        twin=IncrementalFrontend(w,h);gate=TrackPublisher()
        for (t,e,warming),(a,b) in zip(prefix,expected):
            q=twin.advance(e,t,not warming);r=gate.apply(q['proposal'],t)
            assert q['observation']==a and r['observation']==b
            prefix_equal+=1
    assert n==round(cfg['duration']/.02)
    save(folder/'paired_benchmark.json',bench)
    save(folder/'completion.json',dict(case=case,frames=n,optimized_exact_vs_stage17=equal,causal_prefix_exact=prefix_equal,
        width=w,height=h,research_wall_including_open_seek_decode_json_bench_s=raw_wall,
        frontend_plus_track_median_ms=float(np.median(durations)),frontend_plus_track_p95_ms=float(np.percentile(durations,95)),
        source_unchanged=sources()==protocol['source_hashes'],labels_in_inference=False,no_G=True,
        timing_scope='Includes full spectrum update+geometry+track publisher, excludes decode/JSON/benchmark; research wall is not a dedicated real-time benchmark.'))
    print('Completed '+case,flush=True)


def labels(case):
    if case=='negative5':return []
    cfg=CASES[case];name='train4' if case=='review4' else 'train6'
    return [r for r in load(ROOT.parent/'output/raw_only_project/evaluation_data'/f'{name}.json')['labels'] if cfg['start']<r['t']<=cfg['start']+cfg['duration']]


def analyze():
    verify();table=[];details=[];common=[]
    for case in CASES:
        rr=list(map(json.loads,(OUT/case/'observations.jsonl').read_text().splitlines()));times=np.array([r['t'] for r in rr])
        for l in labels(case):
            i=int(np.searchsorted(times,l['t'],side='right')-1)
            if i<0 or l['t']-times[i]>.025:continue
            for m in ['incremental_strict','incremental_track']:
                o=rr[i]['methods'][m]['observation'];d=dict(case=case,method=m,t=l['t'],observation_t=float(times[i]),state=l['state'],detected=o['detected'])
                if l['state']=='present':
                    box=np.array(l['bbox']);diag=np.linalg.norm(box[2:]-box[:2]);center=(box[:2]+box[2:])/2
                    error=float(np.linalg.norm(np.array(o['center'])-center)) if o['detected'] else None
                    d.update(error_px=error,success=bool(o['detected'] and error<=diag*.25),wrong=bool(o['detected'] and error>diag*.5))
                details.append(d)
        for m in ['incremental_strict','incremental_track']:
            dd=[d for d in details if d['case']==case and d['method']==m];pp=[d for d in dd if d['state']=='present'];aa=[d for d in dd if d['state']=='absent'];errors=[d['error_px'] for d in pp if d['detected']]
            table.append(dict(case=case,method=m,frames=len(rr),detected=sum(r['methods'][m]['observation']['detected'] for r in rr),
                present_labels=len(pp),present_detected=len(errors),conditional_error_px=float(np.mean(errors)) if errors else None,
                joint_success=sum(d['success'] for d in pp),wrong_position=sum(d['wrong'] for d in pp),absent_labels=len(aa),absent_detections=sum(d['detected'] for d in aa),
                known_negative_publications=sum(r['methods'][m]['observation']['detected'] for r in rr) if case=='negative5' else None))
        aa={d['t']:d for d in details if d['case']==case and d['method']=='incremental_strict' and d['state']=='present'}
        bb={d['t']:d for d in details if d['case']==case and d['method']=='incremental_track' and d['state']=='present'}
        cc=[t for t in aa if aa[t]['detected'] and bb[t]['detected']]
        common.append(dict(case=case,n=len(cc),strict_error_px=float(np.mean([aa[t]['error_px'] for t in cc])) if cc else None,
            track_error_px=float(np.mean([bb[t]['error_px'] for t in cc])) if cc else None))
    save(OUT/'analysis.json',dict(table=table,details=details,common=common,limits='Sparse image boxes; no independent motor/RPM/pose/future-trajectory truth.'))
    print('Analysis saved',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','all','run','analyze']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action=='all':
        for case in CASES:run(case)
        analyze()
    elif a.action=='run':run(a.case)
    else:globals()[a.action]()
