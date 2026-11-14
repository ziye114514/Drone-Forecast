"""Predeclared native5/10/20ms vs causal warped20ms current-position comparison."""
from pathlib import Path
import argparse,datetime,hashlib,json,time
import numpy as np
from windows21 import ROOT,METHODS,Observer
from checks21 import run as checks
from event_drone_forecast.raw_io import open_raw,raw_chunks

OUT=ROOT.parent/'output/stage21_native_windows_20261018';BASE=ROOT.parent/'output/stage19_soft_layout_20261018'
CASES={'clear6':dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=5.),'side6':dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=12.),
       'review4':dict(raw='E:/dataset/train/4/aligned/event.raw',start=74.,duration=10.),'negative5':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=8.)}
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
    ff=[Path(__file__),Path(__file__).with_name('windows21.py'),Path(__file__).with_name('checks21.py')]
    result={str(p.relative_to(ROOT)):sha(p) for p in ff}
    for p,v in load(BASE/'protocol.json')['source_hashes'].items():assert sha(ROOT/p)==v;result[p]=v
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks());metadata={}
    for raw in set(c['raw'] for c in CASES.values()):
        p=Path(raw);st=p.stat()
        with p.open('rb') as f:head=hashlib.sha256(f.read(65536)).hexdigest()
        metadata[str(p)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,first64k_sha256=head,full_sha_computed=False)
        assert metadata[str(p)]==load(BASE/'protocol.json')['raw_metadata'][str(p)]
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,methods=METHODS,
        snapshots=SNAPS,raw_metadata=metadata,cache_hashes={c:sha(BASE/c/'observations.jsonl') for c in CASES},
        design='Identical neighbor10 geometry/state and50Hz cutoffs; only current event integration5/10/20ms and causal translation warp differ. Counts normalized per10ms.100ms startup abstention aligns prior baseline.',
        warp='x_end=x_event+(t_end-t_event)*past_velocity; velocity from same branch previous measurements, used only<=160ms. Out-of-sensor warped points dropped. No future reference velocities.',
        prior_art=[dict(url='https://arxiv.org/abs/1804.01306',use='Primary contrast-maximization/motion-compensation reference; only past-velocity translation warp used, not optimization framework.'),
            dict(path='src/event_drone_forecast/rotor_sensing.py:periodic_measurement',use='Existing translational dewarping equation reused; existing geometry and stage9 protection executed.')],
        selection='Same35s development/cross-recording/negative data; fixed before new scoring, not blind. No post-result tuning.',
        scope='Current detection/position ablation, no RPM/pose/future-trajectory claim. No G or access to yolov10-main.'))
    print('Stage21 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def run(case):
    protocol=verify();cfg=CASES[case];folder=OUT/case;assert not (folder/'observations.jsonl').exists();folder.mkdir(parents=True,exist_ok=True)
    assert sha(BASE/case/'observations.jsonl')==protocol['cache_hashes'][case]
    cache={round(r['t'],8):r['methods']['neighbor20']['observation'] for r in map(json.loads,(BASE/case/'observations.jsonl').read_text().splitlines())}
    reader=open_raw(cfg['raw']);h,w=reader.get_size();models={m:Observer(w,h,m) for m in METHODS};start=max(0.,cfg['start']-.1);end=cfg['start']+cfg['duration']
    began=time.perf_counter();last_print=began;n=equal=0;prefix=[];expected=[]
    with (folder/'observations.jsonl').open('x',encoding='utf-8') as stream:
        for t,e in raw_chunks(reader,start,end-start,20000):
            warming=t-start<.1-1e-8;row=dict(t=t,methods={})
            for m,f in models.items():
                if warming:q=dict(observation=dict(t=t,detected=False,rotors=[]),candidates=[],audit={})
                else:q=f.step(e,t)
                row['methods'][m]=q
            if t>cfg['start']+1e-8:
                assert row['methods']['native20']['observation']==cache[round(t,8)],(case,t,'baseline changed')
                equal+=1;n+=1;stream.write(json.dumps(row,allow_nan=False)+'\n')
            if case=='clear6' and t<=cfg['start']+.2+1e-8:prefix.append((t,e.copy(),warming));expected.append({m:q['observation'] for m,q in row['methods'].items()})
            if any(abs(t-s)<1e-8 for s in SNAPS[case]):
                p=e['y'].astype(np.int32)*w+e['x'];np.savez_compressed(folder/f'snapshot_{t:.2f}.npz',total=np.bincount(p,minlength=w*h).reshape(h,w))
            if time.perf_counter()-last_print>15:print(json.dumps(dict(case=case,t=t)),flush=True);last_print=time.perf_counter()
    prefix_equal=0
    if prefix:
        twins={m:Observer(w,h,m) for m in METHODS}
        for (t,e,warming),reference in zip(prefix,expected):
            if warming:continue
            for m,f in twins.items():assert f.step(e,t)['observation']==reference[m],(m,t,'prefix')
            prefix_equal+=1
    assert n==round(cfg['duration']/.02)
    save(folder/'completion.json',dict(case=case,frames=n,native20_exact_vs_stage19=equal,causal_prefix_exact=prefix_equal,
        sources_unchanged=sources()==protocol['source_hashes'],labels_in_inference=False,research_wall_s=time.perf_counter()-began,
        scope='Four-arm quality experiment; wall is not single pipeline latency.'))
    print('Completed '+case,flush=True)


def labels(case):
    if case=='negative5':return []
    cfg=CASES[case];name='train4' if case=='review4' else 'train6'
    return [r for r in load(ROOT.parent/'output/raw_only_project/evaluation_data'/f'{name}.json')['labels'] if cfg['start']<r['t']<=cfg['start']+cfg['duration']]


def analyze():
    verify();table=[];details=[];common=[]
    for case in CASES:
        rr=list(map(json.loads,(OUT/case/'observations.jsonl').read_text().splitlines()));tt=np.array([r['t'] for r in rr])
        for l in labels(case):
            i=int(np.searchsorted(tt,l['t'],side='right')-1)
            if i<0 or l['t']-tt[i]>.025:continue
            for m in METHODS:
                o=rr[i]['methods'][m]['observation'];d=dict(case=case,method=m,t=l['t'],state=l['state'],detected=o['detected'])
                if l['state']=='present':
                    b=np.array(l['bbox']);diag=np.linalg.norm(b[2:]-b[:2]);center=(b[:2]+b[2:])/2
                    error=float(np.linalg.norm(np.array(o['center'])-center)) if o['detected'] else None
                    d.update(error_px=error,success=bool(o['detected'] and error<=diag*.25),wrong=bool(o['detected'] and error>diag*.5))
                details.append(d)
        for m in METHODS:
            dd=[d for d in details if d['case']==case and d['method']==m];pp=[d for d in dd if d['state']=='present'];aa=[d for d in dd if d['state']=='absent'];errors=[d['error_px'] for d in pp if d['detected']]
            table.append(dict(case=case,method=m,frames=len(rr),detected=sum(r['methods'][m]['observation']['detected'] for r in rr),present_labels=len(pp),present_detected=len(errors),
                conditional_error_px=float(np.mean(errors)) if errors else None,joint_success=sum(d['success'] for d in pp),wrong_position=sum(d['wrong'] for d in pp),
                absent_labels=len(aa),absent_detections=sum(d['detected'] for d in aa),negative_publications=sum(r['methods'][m]['observation']['detected'] for r in rr) if case=='negative5' else None))
        per={m:{d['t']:d for d in details if d['case']==case and d['method']==m and d['state']=='present'} for m in METHODS}
        common_times=[t for t in per['native20'] if all(per[m][t]['detected'] for m in METHODS)]
        common.append(dict(case=case,n=len(common_times),errors_px={m:float(np.mean([per[m][t]['error_px'] for t in common_times])) if common_times else None for m in METHODS}))
    save(OUT/'analysis.json',dict(table=table,details=details,common=common));print('Analysis saved',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','all','run','analyze']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action=='all':
        for case in CASES:run(case)
        analyze()
    elif a.action=='run':run(a.case)
    else:globals()[a.action]()
