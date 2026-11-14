"""Isolate no-augmentation layout on natural RAW and sparse future endpoints."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,sys
import numpy as np
from structured26 import ROOT,PARAMS,Observer,layout

sys.path.append(str(ROOT/'research/stage22'))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

robust=module('robust23_layout26',ROOT/'research/stage23/robust23.py')
scorer=module('endpoint22_layout26',ROOT/'research/stage22/study22.py')
from event_drone_forecast.raw_io import open_raw,raw_chunks

OUT=ROOT.parent/'output/stage26_natural_layout_20261018';BASE=ROOT.parent/'output/stage21_native_windows_20261018';FORECAST_BASE=ROOT.parent/'output/stage23_robust_acceleration_20261018'
CASES=load_cases=None
CASES={'clear6':dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=5.),'side6':dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=12.),
    'review4':dict(raw='E:/dataset/train/4/aligned/event.raw',start=74.,duration=10.),'negative5':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=8.)}
METHODS=['neighbor20','layout20'];load=scorer.load;save=scorer.save;sha=scorer.sha


def sources():
    ff=[Path(__file__),Path(__file__).with_name('structured26.py'),ROOT/'run_structured_forecast.py']
    result={str(p.relative_to(ROOT)):sha(p) for p in ff}
    for p,v in load(ROOT.parent/'output/stage25_controlled_occlusion_20261018/protocol.json')['source_hashes'].items():assert sha(ROOT/p)==v;result[p]=v
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();prior=load(FORECAST_BASE/'protocol.json')
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,cases=CASES,
        methods=METHODS,raw_metadata=load(BASE/'protocol.json')['raw_metadata'],label_hashes=prior['label_hashes'],
        detection_cache_hashes={c:sha(BASE/c/'observations.jsonl') for c in CASES},forecast_cache_hashes={c:sha(FORECAST_BASE/c/'forecasts.jsonl') for c in CASES},
        design='Natural RAW comparison, no periodic augmentation for either branch. Same20ms current counts, existing neighbor features; layout difference only. Original neighbor observations and all6forecasts must exactly match stage21/23.',
        scope='Development replay after controlled-erasure diagnostic, not blind. Current positions and sparse archived-box future endpoints separately scored. No G or access to yolov10-main.'))
    print('Stage26 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def run():
    protocol=verify()
    for case,cfg in CASES.items():
        folder=OUT/case;assert not (folder/'observations.jsonl').exists();folder.mkdir(parents=True,exist_ok=True)
        assert sha(BASE/case/'observations.jsonl')==protocol['detection_cache_hashes'][case]
        assert sha(FORECAST_BASE/case/'forecasts.jsonl')==protocol['forecast_cache_hashes'][case]
        cached={round(r['t'],8):r['methods']['native20']['observation'] for r in map(json.loads,(BASE/case/'observations.jsonl').read_text().splitlines())}
        old_forecast={round(r['t'],8):r['frontends']['native20'] for r in map(json.loads,(FORECAST_BASE/case/'forecasts.jsonl').read_text().splitlines())}
        reader=open_raw(cfg['raw']);h,w=reader.get_size();fronts={m:Observer(w,h,m) for m in METHODS};models={m:robust.Predictor() for m in METHODS}
        start=max(0.,cfg['start']-.1);end=cfg['start']+cfg['duration'];n=0
        with (folder/'observations.jsonl').open('x',encoding='utf-8') as fo,(folder/'forecasts.jsonl').open('x',encoding='utf-8') as fp:
            for t,e in raw_chunks(reader,start,end-start,20000):
                warming=t-start<.1-1e-8;row=dict(t=t,methods={});forecast=dict(t=t,frontends={})
                for m,f in fronts.items():
                    q=dict(observation=dict(t=t,detected=False,rotors=[]),layout={},audit={}) if warming else f.step(e,t)
                    row['methods'][m]=q
                if t<=cfg['start']+1e-8:continue
                for m,p in models.items():
                    o=row['methods'][m]['observation'];forecast['frontends'][m]=p.step(t,o['center'] if o['detected'] else None)[0]
                assert row['methods']['neighbor20']['observation']==cached[round(t,8)],(case,t,'detection baseline differs')
                assert forecast['frontends']['neighbor20']==old_forecast[round(t,8)],(case,t,'forecast baseline differs')
                fo.write(json.dumps(row,allow_nan=False)+'\n');fp.write(json.dumps(forecast,allow_nan=False)+'\n');n+=1
                if n%200==0:print(json.dumps(dict(case=case,t=t)),flush=True)
        assert n==round(cfg['duration']/.02)
        save(folder/'completion.json',dict(frames=n,baseline_observations_exact=n,baseline_all_model_records_exact=n,sources_unchanged=sources()==protocol['source_hashes'],labels_in_inference=False))
        print('Completed '+case,flush=True)


def analyze():
    verify();table=[];details=[]
    for case,cfg in CASES.items():
        rr=list(map(json.loads,(OUT/case/'observations.jsonl').read_text().splitlines()));tt=np.array([r['t'] for r in rr])
        labels=[] if case=='negative5' else [l for l in load(scorer.label_path(case))['labels'] if cfg['start']<l['t']<=cfg['start']+cfg['duration']]
        for m in METHODS:
            pp=[];aa=[]
            for l in labels:
                i=int(np.searchsorted(tt,l['t'],side='right')-1)
                if i<0 or l['t']-tt[i]>.025:continue
                o=rr[i]['methods'][m]['observation'];d=dict(case=case,method=m,t=l['t'],state=l['state'],detected=o['detected'])
                if l['state']=='present':
                    b=np.array(l['bbox']);error=float(np.linalg.norm(np.array(o['center'])-(b[:2]+b[2:])/2)) if o['detected'] else None
                    d.update(error_px=error,success=bool(o['detected'] and error<=np.linalg.norm(b[2:]-b[:2])*.25));pp.append(d)
                else:aa.append(d)
                details.append(d)
            errors=[d['error_px'] for d in pp if d['detected']]
            table.append(dict(case=case,method=m,frames=len(rr),published=sum(r['methods'][m]['observation']['detected'] for r in rr),present_labels=len(pp),present_detected=len(errors),
                conditional_error_px=float(np.mean(errors)) if errors else None,joint_success=sum(d['success'] for d in pp),absent_labels=len(aa),absent_published=sum(d['detected'] for d in aa),
                layout_matches=sum(r['methods'][m]['layout'].get('matched',False) for r in rr)))
    save(OUT/'detection_analysis.json',dict(table=table,details=details))
    scorer.OUT=OUT;scorer.INPUT=OUT;scorer.FRONTENDS=METHODS;scorer.MODELS=robust.MODELS;scorer.verify=verify
    scorer.score();print('Natural layout and endpoint analyses complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','analyze','all']);a=p.parse_args()
    if a.action=='all':run();analyze()
    else:globals()[a.action]()
