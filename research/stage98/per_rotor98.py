"""Reuse original per-rotor geometry/phase/FFT sensing on official near194.

No new forecaster or mechanical-RPM truth. This checks whether the original
paper-inspired four-rotor measurement chain is observable on the new data.
"""
from pathlib import Path
from collections import deque,Counter
from dataclasses import asdict
import argparse,datetime,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage88')]
from inventory88 import save,load,sha,parse,unique_rows
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.rotor_sensing import PaperPerception,RotorSensing
from event_drone_forecast.raw_io import raw_chunks
OUT=ROOT.parent/'output/stage98_four_rotor_measurements_20261022'
PREV=ROOT.parent/'output/stage95_official_visible_interval_20261022'
DATA=ROOT.parent/'output/stage94_official_large_target_20261022'


def sources():return {str(p):sha(p) for p in [Path(__file__),ROOT/'src/event_drone_forecast/rotor_sensing.py',ROOT/'src/event_drone_forecast/perception.py',ROOT/'src/event_drone_forecast/paper_config.py',ROOT/'src/event_drone_forecast/geometry.py']}


def freeze():
    assert not OUT.exists();p=load(PREV/'protocol_feature95.json');cfg=PaperConfig()
    save(OUT/'protocol98.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        start_s=p['start_s'],duration_s=p['duration_s'],step_us=20000,parameters=asdict(cfg),
        reuse='Unmodified PaperPerception and RotorSensing, including periodic_measurement FFT quadrants/phase, stable IDs, harmonic priors and defaults.',
        scope='Same supplied past50ms ROI as95, same native194 interval;20ms caller means default update_steps cadence is in20ms ticks. Not claimed exact author pipeline.',
        questions='Four current ellipses? Four fresh relative frequencies/references? Why measurements reject? No mechanicalRPM truth or new future trajectory score.',
        no_threshold_search=True,default_promoted=False))
    print('98 original per-rotor module protocol frozen',flush=True)


def verify():
    p=load(OUT/'protocol98.json');assert p['source_hashes']==sources();return p


def run(prefix=False):
    p=verify();data=load(DATA/'download94.json');assert sha(data['path'])==data['native_sha256']
    from metavision_core.event_io.h5_io import HDF5EventsReader
    reader=HDF5EventsReader(data['path']);height,width=reader.get_size();cfg=PaperConfig();geom=PaperPerception(width,height,cfg);sensing=RotorSensing(cfg)
    labels,bad=parse((DATA/'input/coordinates.txt').read_text(encoding='utf-8-sig'));assert not bad;labels=unique_rows(labels)
    duration=1. if prefix else p['duration_s'];start=p['start_s']
    if prefix:labels=[r for r in labels if r['t']<=start+duration]
    folder=OUT/('prefix' if prefix else 'full');folder.mkdir(exist_ok=False)
    history=deque();index=0;latest=None;n=0;counts=Counter();reasons=Counter();cost=[];beg=time.perf_counter();last=beg
    with (folder/'measurements.jsonl').open('x',encoding='utf-8') as f:
        for t,e in raw_chunks(reader,start,duration,20000):
            st=time.perf_counter()
            while index<len(labels) and labels[index]['t']<=t+1e-9:latest=labels[index];index+=1
            bbox=None if latest is None or t-latest['t']>.05+1e-9 else latest['bbox']
            cropped=e[:0] if bbox is None else e[(e['x']>=bbox[0])&(e['x']<bbox[2])&(e['y']>=bbox[1])&(e['y']<bbox[3])]
            observation,_=geom.detect(cropped,t);history.append((t,e.copy()))
            while history and history[0][0]<t-cfg.frequency_window_us/1e6-.000001:history.popleft()
            value=sensing.step(observation,history,t);elapsed=(time.perf_counter()-st)*1000;cost.append(elapsed)
            visible=[r for r in value['rotors'] if r['observed_now']];fresh=[r for r in value['rotors'] if r['frequency_valid']]
            counts['four_visible']+=value['four_visible'];counts['four_ellipses']+=value['four_ellipses'];counts['four_frequencies']+=value['four_frequencies'];counts['any_fresh_frequency']+=bool(fresh)
            for r in visible:
                reason=r['frequency_diagnostics'].get('reason','accepted' if r['frequency_diagnostics'].get('valid') else 'unknown')
                reasons[reason]+=1
            f.write(json.dumps(dict(t=t,past_roi=bbox,observed_centers=[dict(cx=r['cx'],cy=r['cy'],major=r['major'],minor=r['minor'],mode=r['mode']) for r in observation.get('rotors',[])],sensing=value,module_ms=elapsed),allow_nan=False)+'\n');n+=1
            if time.perf_counter()-last>15:print('98 per-rotor',t,dict(counts),flush=True);last=time.perf_counter()
    if prefix:
        a=[json.loads(s) for s in (OUT/'full/measurements.jsonl').open(encoding='utf-8')][:n]
        b=[json.loads(s) for s in (folder/'measurements.jsonl').open(encoding='utf-8')]
        for x,y in zip(a,b):x.pop('module_ms');y.pop('module_ms');assert x==y
    save(folder/'completion98.json',dict(ticks=n,counts=dict(counts),visible_track_frequency_reasons=dict(reasons),
        module_median_ms=float(np.median(cost)),module_p95_ms=float(np.percentile(cost,95)),wall_s=time.perf_counter()-beg,
        actual_native_events=True,past_provided_ROI=True,prefix_exact=prefix,mechanical_RPM_truth=False,source_hash_unchanged=p['source_hashes']==sources()))
    print('98 complete',prefix,dict(counts),dict(reasons),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','prefix']);a=ap.parse_args()
    if a.action=='prefix':run(True)
    else:globals()[a.action]()
