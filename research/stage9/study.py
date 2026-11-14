"""Frozen development comparison for protected geometry and recovery."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage8'))
spec=importlib.util.spec_from_file_location('legacy8_study',ROOT/'research/stage8/study.py')
legacy8=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy8)
from recovery import RecoveryFrontend,PARAMS,tests
from polarity import PolarityFrontend,counts
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=ROOT.parent/'output/stage9_state_recovery_20261017'
OLD=legacy8.OUT;CASES=legacy8.CASES
METHODS=['strict10','neighbor10','guard_strict10','guard_neighbor10']


def save(p,x):
    def scalar(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=scalar),encoding='utf-8')


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def source_hashes():
    files=[Path(__file__),Path(__file__).with_name('recovery.py'),*(ROOT/'src/event_drone_forecast').glob('*.py')]
    files+=[ROOT/'research/stage8/polarity.py',ROOT/'research/stage8/study.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def freeze():
    legacy8.verify();assert not (OUT/'protocol.json').exists()
    save(OUT/'module_checks.json',tests())
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes=source_hashes(),parameters=PARAMS,cases=CASES,step_s=.01,publish_s=.02,
        arms=METHODS,side_ablation='protect_neighbor10 uses same reference and rollback, but no new-hypothesis recovery.',
        baseline_cache={case:dict(observations_sha256=sha(OLD/case/'observations.jsonl'),completion_sha256=sha(OLD/case/'completion.json')) for case in CASES},
        provenance='Raw paths, intervals and start/reset identical to stage8. Cached baselines reused; neighbor10 recomputed and exact-equality checked at every published time.',
        rules='Only four actually observed candidates may refresh reliable layout; partial inferred layout does not refresh age. No coasting is published as measurement. Confirmation uses consecutive past frames only. No parameter search.',
        evaluation='Previously inspected development clips. Sparse whole-UAV boxes: output coverage, center error, within-quarter-diagonal success and >half-diagonal wrong-position proxy; common samples. No manual rotor/RPM truth, no new blind test, no future trajectory accuracy claim.',
        exclusions='No G, no YOLO, no access or changes to yolov10-main; original source/results unchanged.',
        prior_work=['https://github.com/nwojke/deep_sort/blob/master/deep_sort/track.py','https://github.com/FoundationVision/ByteTrack/blob/main/yolox/tracker/byte_tracker.py','https://arxiv.org/abs/2110.06864'],
        reuse='Old frontend/grouping/constellation code is executed; local Stage2/4 confirmation/expiry principles reused. External trackers only reviewed, not executed.'))
    print('Frozen stage9',flush=True)


def verify():
    legacy8.verify();p=load(OUT/'protocol.json');assert p['source_hashes']==source_hashes()
    for case,hashes in p['baseline_cache'].items():
        assert sha(OLD/case/'observations.jsonl')==hashes['observations_sha256']
        assert sha(OLD/case/'completion.json')==hashes['completion_sha256']
    return p


def fronts(w,h,side=False):
    result={'guard_strict10':RecoveryFrontend(w,h,'strict10'),
            'guard_neighbor10':RecoveryFrontend(w,h,'neighbor10')}
    if side:result['protect_neighbor10']=RecoveryFrontend(w,h,'neighbor10',allow_recovery=False)
    return result


def run(case):
    protocol=verify();cfg=CASES[case];folder=OUT/case;assert not (folder/'completion.json').exists()
    folder.mkdir(parents=True,exist_ok=True)
    cached={round(r['t'],8):r for r in map(json.loads,(OLD/case/'observations.jsonl').read_text().splitlines())}
    reader=open_raw(cfg['raw']);h,w=reader.get_size();models=fronts(w,h,case=='side')
    control=PolarityFrontend(w,h,'neighbor10');profiles={m:[] for m in models};count=0;equal=0;prefix={}
    begin=max(0.,cfg['start']-.03);end=cfg['start']+cfg['duration'];start=time.perf_counter();last_print=start
    with (folder/'observations.jsonl').open('w',encoding='utf-8') as file:
        for t,e in raw_chunks(reader,begin,end-begin,10000):
            prepared=counts(e,w,h);result={}
            for m,f in models.items():
                tick=time.perf_counter();o,_=f.detect_prepared(e,t,prepared);profiles[m].append((time.perf_counter()-tick)*1000)
                result[m]=dict(observation=o,audit=f.audit,candidates=f.candidates)
            actual,_=control.detect_prepared(e,t,prepared)
            if case=='side' and t<=cfg['start']+.5+1e-8:prefix[round(t,8)]={m:q['observation'] for m,q in result.items()}
            key=round(t,8)
            if key in cached:
                old=cached[key];assert actual==old['methods']['neighbor10']['observation'],(case,t,'baseline mismatch')
                equal+=1
                for m in ['strict10','neighbor10']:result[m]=old['methods'][m]
                file.write(json.dumps(dict(t=t,methods=result),allow_nan=False)+'\n');count+=1
            if time.perf_counter()-last_print>15:
                print(json.dumps(dict(case=case,t=t)),flush=True);last_print=time.perf_counter()
    assert count==round(cfg['duration']/.02)
    prefix_equal=0
    if case=='side':
        short=open_raw(cfg['raw']);fresh=fronts(w,h,True)
        for t,e in raw_chunks(short,begin,cfg['start']+.5-begin,10000):
            p=counts(e,w,h)
            for m,f in fresh.items():
                o,_=f.detect_prepared(e,t,p);assert o==prefix[round(t,8)][m],(t,m,'prefix mismatch')
            prefix_equal+=1
    save(folder/'completion.json',dict(case=case,published=count,neighbor_baseline_exact=equal,prefix_equal_processed=prefix_equal,
        labels_in_inference=False,G_used=False,yolov10_main_accessed=False,wall_s=time.perf_counter()-start,
        feature_detection_ms={m:dict(median=float(np.median(v)),p95=float(np.percentile(v,95))) for m,v in profiles.items()}))
    print('Completed '+case,flush=True)


def analyze():
    verify();table=[];details=[];comparisons=[]
    for case in CASES:
        records=[json.loads(x) for x in (OUT/case/'observations.jsonl').read_text().splitlines()]
        methods=list(records[0]['methods']);times=np.array([r['t'] for r in records]);case_details=[]
        for label in legacy8.labels(case):
            k=int(np.searchsorted(times,label['t'],side='right')-1)
            if k<0 or label['t']-times[k]>.025:continue
            for m in methods:
                o=records[k]['methods'][m]['observation']
                q=dict(case=case,method=m,label_t=label['t'],observation_t=float(times[k]),state=label['state'],detected=o['detected'])
                if label['state']=='present':
                    b=np.array(label['bbox']);diag=float(np.linalg.norm(b[2:]-b[:2]));center=(b[:2]+b[2:])/2
                    error=float(np.linalg.norm(np.array(o['center'])-center)) if o['detected'] else None
                    q.update(bbox=b.tolist(),diagonal_px=diag,error_px=error,normalized_error=None if error is None else error/diag,
                        successful=o['detected'] and error<=diag*.25,wrong_position=o['detected'] and error>diag*.5)
                case_details.append(q)
        details+=case_details
        for m in methods:
            rr=[r for r in case_details if r['method']==m];present=[r for r in rr if r['state']=='present'];absent=[r for r in rr if r['state']=='absent']
            errs=[r['error_px'] for r in present if r['detected']]
            audits=[r['methods'][m].get('audit',{}) for r in records]
            table.append(dict(case=case,method=m,published=len(records),detected=sum(r['methods'][m]['observation']['detected'] for r in records),
                four_selected=sum(len(r['methods'][m]['observation']['rotors'])==4 for r in records),
                present=len(present),present_detected=sum(r['detected'] for r in present),
                conditional_error_px=float(np.mean(errs)) if errs else None,
                success_quarter_diagonal=sum(r['successful'] for r in present),wrong_half_diagonal=sum(r['wrong_position'] for r in present),
                absent=len(absent),absent_detections=sum(r['detected'] for r in absent),
                transfers=sum(bool(r.get('transferred')) for r in audits),state_rejections=sum(bool(r.get('state_rejection')) for r in audits)))
        pairs=[('strict10','guard_strict10'),('neighbor10','guard_neighbor10')]
        if case=='side':pairs+=[('neighbor10','protect_neighbor10'),('protect_neighbor10','guard_neighbor10')]
        for a,b in pairs:
            aa={r['label_t']:r for r in case_details if r['method']==a and r['state']=='present'}
            bb={r['label_t']:r for r in case_details if r['method']==b and r['state']=='present'}
            common=[t for t in aa if aa[t]['detected'] and bb[t]['detected']]
            comparisons.append(dict(case=case,a=a,b=b,common_labels=len(common),
                mean_a=float(np.mean([aa[t]['error_px'] for t in common])) if common else None,
                mean_b=float(np.mean([bb[t]['error_px'] for t in common])) if common else None,
                per_label=[dict(t=t,a_detected=aa[t]['detected'],b_detected=bb[t]['detected'],
                    a_error=aa[t]['error_px'],b_error=bb[t]['error_px']) for t in aa]))
    save(OUT/'analysis.json',dict(table=table,details=details,common=comparisons))
    print(json.dumps(table),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','all','run','analyze']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action=='all':
        for case in CASES:run(case)
        analyze()
    elif a.action=='run':run(a.case)
    else:globals()[a.action]()
