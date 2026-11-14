"""Acquire causal activity from RAW before any future-label evaluation."""
from pathlib import Path
import argparse,datetime,json,sys,time
import numpy as np
from activity68 import *
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage64'))
import study64 as old
from compensation64 import PastVelocity
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=ROOT.parent/'output/stage68_event_force_input_20261021'
CASES={c:dict(old.CASES[c]) for c in old.MAIN}
load=old.load;save=old.save;sha=old.sha;rows=old.rows

def sources():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),Path(__file__).with_name('activity68.py'),Path(__file__).with_name('controls68.py'),ROOT/'research/stage66/periodicity66.py',ROOT/'research/stage64/compensation64.py']}

def freeze():
    old.verify();assert not OUT.exists()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=CASES,
        parameters=PARAMS,source_hashes=sources(),current_hashes={c:sha(old.prior.REF/f'off_{c}/observations_forecasts.jsonl') for c in CASES},
        raw_metadata=load(old.OUT/'protocol.json')['raw_metadata'],
        scope='Current cached NN ROI + earlier CV states; new RAW ON/OFF features. No labels. '
              'Experimental force proxies, no calibration or true RPM claim.',
        prior_art=['https://arxiv.org/abs/2604.18289','https://arxiv.org/abs/2511.13100','https://arxiv.org/abs/2603.01997'],
        reuse='Existing66 polarity FFT, perception temporal phase coherence,64 velocity; original RLS follows in69.'))
    print('68 protocol frozen',flush=True)

def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for c,h in p['current_hashes'].items():assert sha(old.prior.REF/f'off_{c}/observations_forecasts.jsonl')==h
    for n,m in p['raw_metadata'].items():
        s=Path(n).stat();assert m==dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    return p

def run(case):
    verify();sys.addaudithook(old.prior.previous.guard);cfg=CASES[case]
    folder=OUT/case;folder.mkdir(exist_ok=False)
    rr=rows(old.prior.REF/f'off_{case}/observations_forecasts.jsonl');past=PastVelocity();result=[];timing=[]
    reader=open_raw(Path(cfg['raw']));h,w=reader.get_size();activity=Activity(w,h);packets=[]
    for i,(t,e) in enumerate(raw_chunks(reader,cfg['start'],cfg['duration'],10000)):
        packets.append(e.copy())
        if (i+1)%2:continue
        r=rr[len(result)];assert abs(r['t']-t)<1e-8
        v,va=past.before(t-.02);body=r['body_observation']
        b=body['bbox'] if body.get('detected') and body.get('confidence',0)>=.45 else None
        st=time.perf_counter();q=activity.step(np.concatenate(packets),b,t,v);timing.append((time.perf_counter()-st)*1000);packets=[]
        result.append(dict(t=t,activity=q,velocity_audit=va));past.commit(r)
    assert len(result)==len(rr)
    with (folder/'activity.jsonl').open('x',encoding='utf-8') as f:
        for r in result:f.write(json.dumps(r,allow_nan=False)+'\n')
    summary=dict(case=case,parent=len(result),native10=2*len(result),labels_read=False,
        current_ROI=sum(r['activity']['valid'] for r in result),
        density_supported=sum(r['activity'].get('density',0)>0 for r in result),
        periodic_supported=sum(r['activity'].get('frequency_squared') is not None for r in result),
        median_feature_ms=float(np.median(timing)))
    save(folder/'completion.json',summary);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--case',choices=list(CASES));a=p.parse_args()
    if a.action=='run':run(a.case)
    else:freeze()
