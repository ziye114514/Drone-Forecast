"""Isolate OpenCV fit calls in separate processes after paired regression failure."""
import argparse,datetime,json,time
import numpy as np
import study39 as s

def freeze():
    assert not (s.OUT/'isolation_supplement.json').exists()
    s.verify();s.save(s.OUT/'isolation_supplement.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=s.sha(__import__('pathlib').Path(__file__)),audit_sha256=s.sha(s.OUT/'baseline_failure_audit.json'),
        reason='Standalone original reproduces all150 train3 prefix ticks; interleaved filtered fit changes one nearly degenerate ellipse at26.17. Preserve paired partial data. Run each branch in its own process for train3/negative only. Original exact regression remains mandatory; no tolerance relaxation.',
        timing='Train3/negative separate processes; clear/side paired. Timing not an unbiased hardware speed comparison. Algorithm, event chunks and all predictor settings unchanged.'))
    folder=s.OUT/'train3';target=s.OUT/'failed_paired_train3';assert not target.exists();folder.rename(target)

def run(case,method):
    s.verify();p=s.load(s.OUT/'isolation_supplement.json');assert s.sha(__import__('pathlib').Path(__file__))==p['source_sha256']
    cfg=s.CASES[case];reader=s.open_raw(cfg['raw']);h,w=reader.get_size();front=s.RecoveryFrontend(w,h,'neighbor10');model=s.Predictor();noise=s.StreamFilter(w,h)
    folder=s.OUT/'isolated'/case/method;folder.mkdir(parents=True,exist_ok=False);old=oldf=None
    if 'observation' in cfg:
        old={round(r['t'],8):r for r in map(json.loads,(s.ROOT.parent/'output'/cfg['observation']).read_text(encoding='utf-8').splitlines())}
        oldf={round(r['t'],8):r['frontends'][cfg['front']] for r in map(json.loads,(s.ROOT.parent/'output'/cfg['forecast']).read_text(encoding='utf-8').splitlines())}
    timing=[];n=raw_count=kept_count=published=0;began=time.perf_counter()
    with (folder/'rows.jsonl').open('x',encoding='utf-8') as f:
        for t,e in s.raw_chunks(reader,cfg['start'],cfg['duration'],10000):
            tick=time.perf_counter();clean=noise.step(e) if method=='activity' else e;o,_=front.detect(clean,t);q,_=model.step(t,o['center'] if o['detected'] else None);elapsed=(time.perf_counter()-tick)*1000
            if old is not None and method=='original':
                assert o==old[round(t,8)]['observation'],(case,t,'isolated baseline differs')
                assert q['valid']==oldf[round(t,8)]['valid']
                for m in s.MODELS:assert q['coefficients'].get(m)==oldf[round(t,8)]['coefficients'].get(m)
            f.write(s.json_text(dict(t=t,event_count=len(e),filtered_events=len(clean),observation=o,forecast=q,processing_ms=elapsed))+'\n')
            n+=1;published+=o['detected'];raw_count+=len(e);kept_count+=len(clean);timing.append(elapsed)
    s.save(folder/'completion.json',dict(frames=n,published=published,events_raw=raw_count,events_retained=kept_count,processing_median_ms=float(np.median(timing)),processing_p95_ms=float(np.percentile(timing,95)),wall_s=time.perf_counter()-began,baseline_exact=n if old is not None and method=='original' else None))
    print(case,method,n,published,flush=True)

def merge():
    for c in ['train3','negative5']:
        streams={m:[json.loads(x) for x in (s.OUT/'isolated'/c/m/'rows.jsonl').read_text(encoding='utf-8').splitlines()] for m in s.METHODS};assert [r['t'] for r in streams['original']]==[r['t'] for r in streams['activity']]
        folder=s.OUT/c;folder.mkdir(exist_ok=False)
        with (folder/'observations.jsonl').open('x',encoding='utf-8') as fo,(folder/'forecasts.jsonl').open('x',encoding='utf-8') as ff:
            for a,b in zip(streams['original'],streams['activity']):
                assert a['event_count']==b['event_count'];rr=dict(zip(s.METHODS,[a,b]))
                fo.write(s.json_text(dict(t=a['t'],event_count=a['event_count'],filtered_events=b['filtered_events'],methods={m:dict(observation=r['observation'],processing_ms=r['processing_ms']) for m,r in rr.items()}))+'\n')
                ff.write(s.json_text(dict(t=a['t'],frontends={m:r['forecast'] for m,r in rr.items()}))+'\n')
        cc={m:s.load(s.OUT/'isolated'/c/m/'completion.json') for m in s.METHODS};s.save(folder/'completion.json',dict(frames=len(streams['original']),published={m:r['published'] for m,r in cc.items()},events_raw=cc['original']['events_raw'],events_retained=cc['activity']['events_retained'],retention=cc['activity']['events_retained']/max(1,cc['original']['events_raw']),baseline_observations_and_six_coefficients_exact=cc['original']['baseline_exact'],processing_median_ms={m:r['processing_median_ms'] for m,r in cc.items()},processing_p95_ms={m:r['processing_p95_ms'] for m,r in cc.items()},isolated_branch_processes=True))
    s.score()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','merge']);p.add_argument('--case');p.add_argument('--method');a=p.parse_args()
    if a.action=='run':run(a.case,a.method)
    else:globals()[a.action]()
