"""Predefined RAW comparison; labels are read by analysis only."""
from pathlib import Path
import argparse,collections,datetime,hashlib,importlib.util,json,sys,time
import numpy as np
from frontend import Frontend,SparseSpectrum,PARAMS,ROOT
from checks import run as checks
from event_drone_forecast.raw_io import open_raw,raw_chunks

spec=importlib.util.spec_from_file_location('frozen16',ROOT/'research/stage16/blade_detector.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
OUT=ROOT.parent/'output/stage17_sparse_validated_roi_20261018'
RAW6='E:/dataset/train/6/media/event.raw'
CASES={'clear6':dict(raw=RAW6,start=22.,duration=5.,role='development'),
       'side6':dict(raw=RAW6,start=108.,duration=12.,role='development'),
       'review4':dict(raw='E:/dataset/train/4/aligned/event.raw',start=74.,duration=10.,role='fixed cross-recording check, previously visually inspected'),
       'negative5':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=8.,role='explicit no-UAV control')}
SNAPS={'clear6':[23.,24.,25.],'side6':[110.,112.,118.],'review4':[75.,78.,83.],'negative5':[2.,6.]}


def save(p,x):
    def scalar(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=scalar),encoding='utf-8')


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def sources():
    files=[Path(__file__),ROOT/'research/stage17/frontend.py',ROOT/'research/stage17/checks.py',
        ROOT/'research/stage16/blade_detector.py',ROOT/'research/stage9/recovery.py',ROOT/'research/stage8/polarity.py',
        *(ROOT/'src/event_drone_forecast').glob('*.py')]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def freeze():
    assert not (OUT/'protocol.json').exists()
    result=checks();save(OUT/'module_checks.json',result)
    raw={}
    for cfg in CASES.values():
        path=Path(cfg['raw']);st=path.stat()
        with path.open('rb') as f:head=hashlib.sha256(f.read(65536)).hexdigest()
        raw[str(path)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,first64k_sha256=head,full_sha_computed=False)
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        parameters=PARAMS,cases=CASES,snapshots=SNAPS,raw_metadata=raw,warmup_s=.1,
        methods={'legacy16':'Unmodified full-grid1ms spectral detector,10Hz.',
                 'fixed_spectrum':'250us sparse active-block spectrum,10Hz, periodic-region evidence only.',
                 'validated_global':'250us spectrum+existing geometry/state recovery+3-frame publication confirmation,50Hz full-frame.',
                 'validated_roi':'Same as validated_global,50Hz local ROI,full-frame at least every100ms or after160ms loss.'},
        geometry='Current20ms event counts divided by2 preserve10ms-equivalent units; old contour/ellipse/line/grouping reused. Periodic block support dilated3x3, native count gate remains. Four candidates are not validated four physical motors.',
        evaluation='Sparse previously archived image boxes, past-only nearest observation<=25ms. Mean conditional/current pixel error, coverage, joint success<=.25boxdiag, wrong-position>.5boxdiag, common samples; full known-negative false publications. Region presence is separate from position output.',
        rotor_reference='Existing3clear-frame AI approximate12centers with8/12/20px thresholds, supplemental only, not human GT.',
        selection='Case intervals and parameters fixed before RAW scoring. Train4 is a cross-recording check, not claimed untouched blind data. No post-result tuning.',
        scope='Candidate detection/observation experiment only, no mechanical RPM/pose truth, no trajectory improvement claim. No G or access to yolov10-main.',
        prior_art=[dict(url='https://github.com/radimspetlik/effqpdet',status='Official description reviewed; original C++ not built or run; cmake/compiler not found on current PATH.'),
                   dict(url='https://github.com/radimspetlik/helixtrack',status='Recording-initialized phase/geometry tracker reviewed, not run in this study.')],
        reuse='Executes local original geometry, group association and stage9 state protection; sparse-spectrum/ROI adapter is custom and not full author replication.'))
    print('Stage17 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'],'Frozen source changed'
    return p


def legacy_cache(case):
    folder={'clear6':'run_0_35','side6':'run_105_120'}.get(case)
    if folder is None:return {}
    path=ROOT.parent/'output/stage16_blade_detector_20261017'/folder/'ticks.jsonl'
    return {round(r['t'],8):r for r in map(json.loads,path.read_text().splitlines())}


def run(case):
    verify();cfg=CASES[case];folder=OUT/case;assert not (folder/'completion.json').exists()
    folder.mkdir(parents=True,exist_ok=True);reader=open_raw(cfg['raw']);h,w=reader.get_size()
    models={m:Frontend(w,h,m) for m in ['global','roi']};control=old.BurstDetector(w,h);fixed=SparseSpectrum(w,h)
    history=collections.deque();cache=legacy_cache(case);equal=0;published=0;profiles={m:[] for m in ['legacy16','fixed_spectrum','global','roi']}
    start=max(0.,cfg['start']-.1);end=cfg['start']+cfg['duration'];began=time.perf_counter();last_print=began;prefix=[];prefix_expected=[]
    with (folder/'observations.jsonl').open('w',encoding='utf-8') as stream:
        for t,e in raw_chunks(reader,start,end-start,PARAMS['step_us']):
            history.append((t,e))
            while history and history[0][0]<=t-.1+1e-8:history.popleft()
            recent=np.concatenate([x for _,x in history]);warming=t-start<.1-1e-8
            row=dict(t=t,methods={},raw_events_current=len(e),raw_events_history=len(recent))
            for m,f in models.items():
                if warming:
                    q=dict(observation=dict(t=t,detected=False,rotors=[]),search='warming',candidates=[],raw_candidates=[],periodic_present=False)
                else:q=f.step(recent,e,t);profiles[m].append(q['timing_ms']['total'])
                row['methods'][m]=q
            coarse=round(t*1e6)%100000==0 and not warming
            if coarse:
                tic=time.perf_counter();previous=control.step(recent);profiles['legacy16'].append((time.perf_counter()-tic)*1000)
                row['methods']['legacy16']=previous
                if round(t,8) in cache:
                    expected={k:cache[round(t,8)][k] for k in previous}
                    assert previous==expected,(case,t,'legacy cache differs');equal+=1
                tic=time.perf_counter();q=fixed.step(recent,t);profiles['fixed_spectrum'].append((time.perf_counter()-tic)*1000)
                row['methods']['fixed_spectrum']=dict(regions=q['regions'],active_blocks=q['active_blocks'],eligible_blocks=q['eligible_blocks'])
            if case=='clear6' and t<=cfg['start']+.2+1e-8:
                prefix.append((t,recent.copy(),e.copy(),warming))
                prefix_expected.append({m:q['observation'] for m,q in row['methods'].items() if m in models})
            if t>cfg['start']+1e-8:
                stream.write(json.dumps(row,allow_nan=False)+'\n');published+=1
            if any(abs(t-s)<1e-8 for s in SNAPS[case]):
                pixels=e['y'].astype(np.int32)*w+e['x'];total=np.bincount(pixels,minlength=w*h).reshape(h,w)
                on=np.bincount(pixels[e['p']>0],minlength=w*h).reshape(h,w)
                np.savez_compressed(folder/f'snapshot_{t:.2f}.npz',on=on,off=total-on)
            if time.perf_counter()-last_print>15:
                print(json.dumps(dict(case=case,t=t)),flush=True);last_print=time.perf_counter()
    prefix_count=0
    if prefix:
        twins={m:Frontend(w,h,m) for m in models}
        for (t,recent,e,warming),expected in zip(prefix,prefix_expected):
            if warming:continue
            for m,f in twins.items():assert f.step(recent,e,t)['observation']==expected[m],(t,m,'prefix differs')
            prefix_count+=1
    assert published==round(cfg['duration']/.02),(case,published)
    timing={m:dict(n=len(v),median=float(np.median(v)),p95=float(np.percentile(v,95))) for m,v in profiles.items() if v}
    save(folder/'completion.json',dict(case=case,published=published,legacy_cache_exact_times=equal,causal_prefix_times=prefix_count,
        timing_ms=timing,wall_s=time.perf_counter()-began,width=w,height=h,labels_in_inference=False,G_used=False,
        model_sources_unchanged=sources()==load(OUT/'protocol.json')['source_hashes'],
        speed_limit='Research loop runs two new frontends plus two controls and shared decoding/concatenation; not a single pipeline real-time benchmark.'))
    print('Completed '+case,flush=True)


def labels(case):
    cfg=CASES[case];lo=cfg['start'];hi=lo+cfg['duration']
    if case=='negative5':return []
    name='train4' if case=='review4' else 'train6'
    path=ROOT.parent/'output/raw_only_project/evaluation_data'/f'{name}.json'
    return [r for r in load(path)['labels'] if lo<r['t']<=hi]


def analyze():
    verify();table=[];details=[];common=[];presence=[]
    for case in CASES:
        rows=list(map(json.loads,(OUT/case/'observations.jsonl').read_text().splitlines()));times=np.array([r['t'] for r in rows])
        for label_row in labels(case):
            i=int(np.searchsorted(times,label_row['t'],side='right')-1)
            if i<0 or label_row['t']-times[i]>.025:continue
            for method in ['global','roi']:
                o=rows[i]['methods'][method]['observation'];d=dict(case=case,method=method,t=label_row['t'],observation_t=float(times[i]),state=label_row['state'],detected=o['detected'])
                if label_row['state']=='present':
                    box=np.array(label_row['bbox']);diag=np.linalg.norm(box[2:]-box[:2]);target=(box[:2]+box[2:])/2
                    error=float(np.linalg.norm(np.array(o['center'])-target)) if o['detected'] else None
                    d.update(error_px=error,success=o['detected'] and error<=diag*.25,wrong=o['detected'] and error>diag*.5)
                details.append(d)
        for m in ['global','roi']:
            dd=[d for d in details if d['case']==case and d['method']==m];pp=[d for d in dd if d['state']=='present'];aa=[d for d in dd if d['state']=='absent']
            errors=[d['error_px'] for d in pp if d['detected']]
            table.append(dict(case=case,method=m,frames=len(rows),detected=sum(r['methods'][m]['observation']['detected'] for r in rows),
                four_candidate_outputs=sum(len(r['methods'][m]['observation']['rotors'])==4 for r in rows),
                mean_validated_candidates=float(np.mean([len(r['methods'][m]['candidates']) for r in rows])),
                roi_frames=sum(r['methods'][m]['search']=='roi' for r in rows),present_labels=len(pp),present_detected=sum(d['detected'] for d in pp),
                conditional_error_px=float(np.mean(errors)) if errors else None,joint_success=sum(d['success'] for d in pp),wrong_position=sum(d['wrong'] for d in pp),
                absent_labels=len(aa),absent_detections=sum(d['detected'] for d in aa),
                known_negative_publications=sum(r['methods'][m]['observation']['detected'] for r in rows) if case=='negative5' else None))
        aa={d['t']:d for d in details if d['case']==case and d['method']=='global' and d['state']=='present'}
        bb={d['t']:d for d in details if d['case']==case and d['method']=='roi' and d['state']=='present'}
        cc=[t for t in aa if aa[t]['detected'] and bb[t]['detected']]
        common.append(dict(case=case,n=len(cc),global_error=float(np.mean([aa[t]['error_px'] for t in cc])) if cc else None,
            roi_error=float(np.mean([bb[t]['error_px'] for t in cc])) if cc else None))
        coarse=[r for r in rows if 'legacy16' in r['methods']]
        for m in ['legacy16','fixed_spectrum','global','roi']:
            presence.append(dict(case=case,method=m,coarse_frames=len(coarse),
                periodic_region_present=sum(bool(r['methods'][m]['regions']) if m in ['legacy16','fixed_spectrum'] else r['methods'][m]['periodic_present'] for r in coarse)))
    save(OUT/'analysis.json',dict(table=table,details=details,common=common,coarse_presence=presence,
        limits='Presence of periodic regions is not localisation or per-rotor accuracy. Sparse image boxes are not motor/hub/RPM truth.'))
    print(json.dumps(table),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','all','run','analyze']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action=='all':
        for case in CASES:run(case)
        analyze()
    elif a.action=='run':run(a.case)
    else:globals()[a.action]()
