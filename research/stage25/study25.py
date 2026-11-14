"""Fixed controlled partial-observation experiment, all outcomes retained."""
from pathlib import Path
import argparse,datetime,hashlib,json,time
import numpy as np
from occlusion25 import ROOT,PARAMS,OracleMask,Learner,windows
from checks25 import run as checks
from event_drone_forecast.raw_io import open_raw,raw_chunks

OUT=ROOT.parent/'output/stage25_controlled_occlusion_20261018';RAW='E:/dataset/train/6/media/event.raw'
CONDITIONS={f'{pattern}_{duration:.2f}':dict(pattern=pattern,duration=duration) for pattern in PARAMS['patterns'] for duration in PARAMS['occlusion_durations']}
CONDITIONS['dropout_.60']=dict(pattern='dropout',duration=.60)


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    def scalar(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(type(v).__name__)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=scalar),encoding='utf-8')


def sources():
    files=[Path(__file__),Path(__file__).with_name('occlusion25.py'),Path(__file__).with_name('checks25.py')]
    result={str(p.relative_to(ROOT)):sha(p) for p in files}
    for p,v in load(ROOT.parent/'output/stage24_raw_pipeline_20261018/protocol.json')['source_hashes'].items():assert sha(ROOT/p)==v;result[p]=v
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks())
    p=Path(RAW);st=p.stat()
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,
        raw=RAW,raw_metadata=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns),conditions=CONDITIONS,
        design='Clear acquisition then oracle-generated spatial event erasure. Uncorrupted neighbor20 detector tracks4slot locations for corruption generator only; masked event batches alone reach independent learners.',
        models=['neighbor','bounded_layout'],reference='Uncorrupted detector whole-box center, NOT independent GT. Scores diagnose invariance under artificial erasure; not natural-occlusion accuracy or motor identity.',
        scope='Current position relative to unmasked counterpart, forecast publication timeout only; no true future-position/RPM/pose accuracy claim. No G or access to yolov10-main.',
        selection='Same clear development recording, fixed durations/patterns before outcome. No post-result tuning.'))
    print('Stage25 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def run():
    verify();assert not (OUT/'observations.jsonl').exists()
    reader=open_raw(RAW);h,w=reader.get_size();teacher=windows.Observer(w,h,'native20');mask=OracleMask()
    learners={name:{kind:Learner(w,h,kind) for kind in ['neighbor','bounded_layout']} for name in CONDITIONS}
    start=PARAMS['start'];begin=start-.1;end=start+PARAMS['duration'];n=0;began=time.perf_counter();last_print=began
    with (OUT/'observations.jsonl').open('x',encoding='utf-8') as stream:
        for t,e in raw_chunks(reader,begin,end-begin,20000):
            if t-begin<.1-1e-8:continue
            reference=teacher.step(e,t)['observation'];fresh=mask.update(reference);row=dict(t=t,reference=reference,mask_current_four_available=fresh,conditions={})
            for name,cfg in CONDITIONS.items():
                active=PARAMS['erase_start']-1e-8<=t<PARAMS['erase_start']+cfg['duration']-1e-8
                current=e;erased=False
                if active:
                    if cfg['pattern']=='dropout':current=e[:0];erased=True
                    else:current,erased=mask.erase(e,cfg['pattern'])
                row['conditions'][name]=dict(active=active,erasure_applied=erased,events_before=len(e),events_after=len(current),
                    methods={kind:l.step(current,t) for kind,l in learners[name].items()})
                if abs(t-23.48)<1e-8:
                    pix=current['y'].astype(np.int32)*w+current['x'];np.savez_compressed(OUT/f'snapshot_{name}.npz',total=np.bincount(pix,minlength=w*h).reshape(h,w))
            if t>start+1e-8:stream.write(json.dumps(row,allow_nan=False)+'\n');n+=1
            if time.perf_counter()-last_print>15:print(json.dumps(dict(t=t,frames=n)),flush=True);last_print=time.perf_counter()
    assert n==round(PARAMS['duration']/.02)
    save(OUT/'completion.json',dict(frames=n,sources_unchanged=sources()==load(OUT/'protocol.json')['source_hashes'],wall_s=time.perf_counter()-began,
        labels_in_inference=False,reference_never_fed_to_learners=True))
    print('Controlled occlusion complete',flush=True)


def analyze():
    verify();rows=list(map(json.loads,(OUT/'observations.jsonl').read_text().splitlines()));table=[]
    for name in CONDITIONS:
        active=[r for r in rows if r['conditions'][name]['active']]
        for method in ['neighbor','bounded_layout']:
            errors=[];detected=0;matched=0
            for r in active:
                q=r['conditions'][name]['methods'][method];o=q['observation'];detected+=o['detected'];matched+=q['layout'].get('matched',False)
                if o['detected'] and r['reference']['detected']:errors.append(float(np.linalg.norm(np.array(o['center'])-r['reference']['center'])))
            table.append(dict(condition=name,method=method,erasure_ticks=len(active),published=int(detected),conditional_reference_error_px=float(np.mean(errors)) if errors else None,
                within20px_reference=sum(x<=20 for x in errors),layout_matches=int(matched),forecast_valid_ticks=sum(r['conditions'][name]['methods'][method]['forecast_valid'] for r in active)))
    save(OUT/'analysis.json',dict(table=table,scope='Artificial erasure / unmasked detector reference, not independent real-world accuracy.'))
    print('Controlled occlusion analysis saved',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','analyze','all']);a=p.parse_args()
    if a.action=='all':run();analyze()
    else:globals()[a.action]()
