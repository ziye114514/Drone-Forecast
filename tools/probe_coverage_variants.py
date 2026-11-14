"""Can side-view coverage be raised without waking the background?

The gate closed on 56% of side-view samples, so before inventing another
criterion the coverage itself has to improve. Four declared variants, each with
a stated reason; the empty-scene clip is the guard that must stay near zero.

  A  baseline                      33.3ms window, >=6 events per half window
  B  longer window                 66.7ms window (more events per half); the
                                   discovery/validation split stays33.3ms each
  C  longer window + sparser floor 66.7ms and >=4 events per half
  D  sparser density               33.3ms, density floor4 and contrast1.5x

No threshold is tuned against trajectory error; only coverage and the empty-scene
false-positive count are read here.
"""
from pathlib import Path
from collections import deque
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT)]
from live_decode import native_batches
import predictive_harmonics
from rotor_support_front import PARAMS,rotor_support

BASE=ROOT.parent/'output'
OUT=BASE/'rotor_support_20261101'
CLIPS=[('side','E:/dataset/train/6/media/event.raw',111.0),
    ('clear','E:/dataset/train/6/media/event.raw',26.5),
    ('negative','E:/dataset/train/5/media/event.raw',0.5)]
VARIANTS=[('A',dict(window_us=33333,min_half=6,density_floor=6,contrast=2.0)),
    ('B',dict(window_us=66667,min_half=6,density_floor=6,contrast=2.0)),
    ('C',dict(window_us=66667,min_half=4,density_floor=6,contrast=2.0)),
    ('D',dict(window_us=33333,min_half=6,density_floor=4,contrast=1.5))]


def events_at(raw,cutoff):
    queue=deque()
    for t,events in native_batches(Path(raw),cutoff-.35,.45,False,33333,reader_backend='rolling'):
        queue.append(events.copy())
        while queue and (not len(queue[0]) or queue[0]['t'][-1]<(cutoff-.30)*1e6):queue.popleft()
        if t>=cutoff-1e-9:break
    return np.concatenate(queue) if queue else np.zeros(0,dtype=[('x','u2'),('y','u2'),('p','i2'),('t','i8')])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache={name:events_at(raw,cutoff) for name,raw,cutoff in CLIPS}
    original=predictive_harmonics.PARAMS['min_events_per_half']
    result=dict(variants={},clips={name:cutoff for name,_,cutoff in CLIPS},
        guard='the empty-scene clip must stay near zero period pixels')
    for label,config in VARIANTS:
        params=dict(PARAMS,window_us=config['window_us'],
            min_events_per_pixel=config['density_floor'],local_multiplier=config['contrast'])
        predictive_harmonics.PARAMS['min_events_per_half']=config['min_half']
        entry=dict(config=config,clips={})
        try:
            for name,_,cutoff in CLIPS:
                started=time.perf_counter()
                frame=rotor_support(cache[name],cutoff,params=params)
                entry['clips'][name]=dict(counts=frame['counts'],
                    components=frame['component_count'],bbox=frame['bbox'],
                    elapsed_s=round(time.perf_counter()-started,2))
        finally:
            predictive_harmonics.PARAMS['min_events_per_half']=original
        result['variants'][label]=entry
        print(label,{name:(value['counts']['period_pixels'],value['counts']['density_pixels'])
            for name,value in entry['clips'].items()},flush=True)
    (OUT/'coverage_variants.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print('wrote',OUT/'coverage_variants.json')


if __name__=='__main__':main()
