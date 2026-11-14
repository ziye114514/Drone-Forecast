"""Same-input neighbor/soft-evidence/layout RAW ablation, fixed before scoring."""
from pathlib import Path
import argparse,datetime,hashlib,json,time
import numpy as np
from layout19 import ROOT,PARAMS,Observer,inc,counts
from checks19 import run as checks
from event_drone_forecast.raw_io import open_raw,raw_chunks

OUT=ROOT.parent/'output/stage19_soft_layout_20261018'
PREVIOUS=ROOT.parent/'output/stage18_incremental_track_20261018'
CASES={'clear6':dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=5.),
    'side6':dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=12.),
    'review4':dict(raw='E:/dataset/train/4/aligned/event.raw',start=74.,duration=10.),
    'negative5':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=8.)}
METHODS={'neighbor20':(False,False),'soft20':(True,False),'layout20':(True,True)}
SNAPS={'clear6':[23.,24.,25.],'side6':[110.,112.,116.6,117.28,117.94,118.,119.92],'review4':[75.,78.,82.96,83.],'negative5':[2.,6.]}


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    def scalar(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(type(v).__name__)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=scalar),encoding='utf-8')


def sources():
    files=[Path(__file__),Path(__file__).with_name('layout19.py'),Path(__file__).with_name('checks19.py'),ROOT/'research/stage11/identity.py']
    result={str(p.relative_to(ROOT)):sha(p) for p in files}
    for p,v in load(PREVIOUS/'protocol.json')['source_hashes'].items():
        assert sha(ROOT/p)==v,('previous source changed',p)
        result[p]=v
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks())
    metadata={}
    for raw in set(c['raw'] for c in CASES.values()):
        p=Path(raw);st=p.stat()
        with p.open('rb') as f:head=hashlib.sha256(f.read(65536)).hexdigest()
        metadata[str(p)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,first64k_sha256=head,full_sha_computed=False)
        assert metadata[str(p)]==load(PREVIOUS/'protocol.json')['raw_metadata'][str(p)]
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,
        cases=CASES,methods=METHODS,snapshots=SNAPS,raw_metadata=metadata,warmup_s=.1,
        design='Same current20ms counts /2, original neighbor features vs soft periodic augmentation vs same plus bounded layout; no additional stage17/18 outer publication gate.',
        layout='Stage11 similarity fit and orientation helper, Hungarian assignment. Model seeded only3consecutive actual convex nondegenerate four measurements; partial observations cannot refresh full template age300ms. No coast measurements.',
        selection='35s previously scored development/cross-recording/negative data; not blind. Parameters frozen before this run; no post-result tuning.',
        scope='Detection/current image positions only; no independent rotor/RPM/pose truth or future prediction claim. No G or access to yolov10-main.',
        prior_art=[dict(url='https://github.com/siavashk/pycpd',status='Rigid/affine point registration reviewed, not executed;4points use existing local stage11 instead.'),
            dict(url='https://arxiv.org/abs/0905.2635',status='CPD primary reference reviewed; no new registration algorithm claim.'),
            dict(url='https://github.com/radimspetlik/helixtrack',status='Geometry/phase tracker inspected previously; initialization assumptions not imported.')]))
    print('Stage19 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def run(case):
    protocol=verify();cfg=CASES[case];folder=OUT/case
    assert not (folder/'observations.jsonl').exists(),'Refuse overwrite/partial run';folder.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter();reader=open_raw(cfg['raw']);h,w=reader.get_size();spectral=inc.IncrementalSpectrum(w,h)
    models={m:Observer(w,h,*opts) for m,opts in METHODS.items()};start=max(0.,cfg['start']-.1);end=cfg['start']+cfg['duration']
    n=0;last_print=began;prefix=[];expected=[];profiles={m:[] for m in METHODS}
    with (folder/'observations.jsonl').open('x',encoding='utf-8') as stream:
        for t,e in raw_chunks(reader,start,end-start,20000):
            began_shared=time.perf_counter();spectral.update(e,t);warming=t-start<.1-1e-8
            spec=None if warming else spectral.step(None,t)
            prepared=None if warming else counts(e,w,h)
            if prepared is not None:prepared['on']/=2;prepared['off']/=2
            shared_ms=(time.perf_counter()-began_shared)*1000;row=dict(t=t,methods={},shared_ms=shared_ms,event_count=len(e))
            for m,model in models.items():
                if warming:q=dict(observation=dict(t=t,detected=False,rotors=[]),candidates=[],layout={})
                else:
                    tick=time.perf_counter();q=model.step(e,t,prepared,spec);q['geometry_ms']=(time.perf_counter()-tick)*1000
                    profiles[m].append(q['geometry_ms']+shared_ms)
                row['methods'][m]=q
            if case=='clear6' and t<=cfg['start']+.2+1e-8:
                prefix.append((t,e.copy(),warming));expected.append({m:q['observation'] for m,q in row['methods'].items()})
            if t>cfg['start']+1e-8:stream.write(json.dumps(row,allow_nan=False)+'\n');n+=1
            if any(abs(t-s)<1e-8 for s in SNAPS[case]):np.savez_compressed(folder/f'snapshot_{t:.2f}.npz',total=prepared['total'])
            if time.perf_counter()-last_print>15:print(json.dumps(dict(case=case,t=t)),flush=True);last_print=time.perf_counter()
    wall=time.perf_counter()-began;prefix_equal=0
    if prefix:
        ring=inc.IncrementalSpectrum(w,h);twins={m:Observer(w,h,*opts) for m,opts in METHODS.items()}
        for (t,e,warming),reference in zip(prefix,expected):
            ring.update(e,t)
            if warming:continue
            s=ring.step(None,t);p=counts(e,w,h);p['on']/=2;p['off']/=2
            for m,f in twins.items():assert f.step(e,t,p,s)['observation']==reference[m],(m,t,'prefix')
            prefix_equal+=1
    assert n==round(cfg['duration']/.02)
    save(folder/'completion.json',dict(case=case,frames=n,causal_prefix_exact=prefix_equal,research_wall_s=wall,
        timing_ms={m:dict(median=float(np.median(v)),p95=float(np.percentile(v,95))) for m,v in profiles.items()},
        sources_unchanged=sources()==protocol['source_hashes'],labels_in_inference=False,
        timing_scope='Per-arm shared ingest/spectrum/counts charged once+own geometry; research wall runs all3arms, not single pipeline real-time proof.'))
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
            for m in METHODS:
                q=rr[i]['methods'][m];o=q['observation'];d=dict(case=case,method=m,t=l['t'],state=l['state'],detected=o['detected'],layout=q.get('layout',{}))
                if l['state']=='present':
                    b=np.array(l['bbox']);diag=np.linalg.norm(b[2:]-b[:2]);target=(b[:2]+b[2:])/2
                    error=float(np.linalg.norm(np.array(o['center'])-target)) if o['detected'] else None
                    d.update(error_px=error,success=bool(o['detected'] and error<=diag*.25),wrong=bool(o['detected'] and error>diag*.5))
                details.append(d)
        for m in METHODS:
            dd=[d for d in details if d['case']==case and d['method']==m];pp=[d for d in dd if d['state']=='present'];aa=[d for d in dd if d['state']=='absent'];errors=[d['error_px'] for d in pp if d['detected']]
            table.append(dict(case=case,method=m,frames=len(rr),detected=sum(r['methods'][m]['observation']['detected'] for r in rr),
                present_labels=len(pp),present_detected=len(errors),conditional_error_px=float(np.mean(errors)) if errors else None,joint_success=sum(d['success'] for d in pp),wrong_position=sum(d['wrong'] for d in pp),
                absent_labels=len(aa),absent_detections=sum(d['detected'] for d in aa),
                template_frames=sum(r['methods'][m].get('layout',{}).get('template_available_after',False) for r in rr),
                matched_frames=sum(r['methods'][m].get('layout',{}).get('matched',False) for r in rr),
                negative_publications=sum(r['methods'][m]['observation']['detected'] for r in rr) if case=='negative5' else None))
        dd={m:{d['t']:d for d in details if d['case']==case and d['method']==m and d['state']=='present'} for m in METHODS}
        shared=[t for t in dd['neighbor20'] if all(dd[m][t]['detected'] for m in METHODS)]
        common.append(dict(case=case,n=len(shared),errors_px={m:float(np.mean([dd[m][t]['error_px'] for t in shared])) if shared else None for m in METHODS}))
    save(OUT/'analysis.json',dict(table=table,details=details,common=common));print('Analysis saved',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','all','run','analyze']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action=='all':
        for case in CASES:run(case)
        analyze()
    elif a.action=='run':run(a.case)
    else:globals()[a.action]()
