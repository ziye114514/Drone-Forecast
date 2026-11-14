"""Why is side-view coverage low: not enough eligible pixels, or weaker validation?

Eligible pixel counts are similar on the side and clear clips, but a much smaller
share passes the temporal validation. This probe reports the validation statistic
(bound) for eligible pixels on three clips and the accepted share at three
declared thresholds, plus the empty-scene false-positive count at the same
thresholds. A threshold that buys side coverage by waking the background is not
a gain.
"""
from pathlib import Path
from collections import deque
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT)]
from live_decode import native_batches
from rotor_support_front import PARAMS,accumulate,density_mask
from predictive_harmonics import predictive_harmonics

BASE=ROOT.parent/'output'
OUT=BASE/'rotor_support_20261101'
CLIPS=[('side','E:/dataset/train/6/media/event.raw',111.0),
    ('clear','E:/dataset/train/6/media/event.raw',26.5),
    ('negative','E:/dataset/train/5/media/event.raw',0.5),
    ('dense_background','E:/dataset/train/3/aligned/event.raw',1.0)]
THRESHOLDS=[0.01,0.02,0.05]


def events_at(raw,cutoff):
    queue=deque()
    for t,events in native_batches(Path(raw),cutoff-.35,.45,False,33333,reader_backend='rolling'):
        queue.append(events.copy())
        while queue and (not len(queue[0]) or queue[0]['t'][-1]<(cutoff-.30)*1e6):queue.popleft()
        if t>=cutoff-1e-9:break
    return np.concatenate(queue) if queue else np.zeros(0,dtype=[('x','u2'),('y','u2'),('p','i2'),('t','i8')])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(clips={},thresholds=THRESHOLDS,
        note='bound is the predictive-harmonics validation statistic; lower is stronger')
    for name,raw,cutoff in CLIPS:
        events=events_at(raw,cutoff)
        on,off,meta=accumulate(events,cutoff)
        dense,_=density_mask(on+off)
        kept=events[dense[events['y'].astype(np.int64),events['x'].astype(np.int64)]]
        period=predictive_harmonics(kept,cutoff)
        bounds=period['candidate_bound']
        minimum=np.min(bounds,axis=1) if len(bounds) else np.zeros(0)
        accepted={}
        for threshold in THRESHOLDS:
            blocked=period['distribution'] if False else period['temporal_coverage']
            passed=(minimum<=threshold)
            accepted[str(threshold)]=dict(pixels=int(np.count_nonzero(passed)),
                share=None if not len(minimum) else float(np.mean(passed)))
        shuffled=kept.copy()
        rng=np.random.default_rng(0)
        shuffled['p']=rng.permutation(shuffled['p'])
        null=predictive_harmonics(shuffled,cutoff)
        null_min=np.min(null['candidate_bound'],axis=1) if len(null['candidate_bound']) else np.zeros(0)
        null_accepted={str(threshold):int(np.count_nonzero(null_min<=threshold))
            for threshold in THRESHOLDS}
        quartiles=None if not len(minimum) else dict(
            p10=float(np.percentile(minimum,10)),median=float(np.median(minimum)),
            p90=float(np.percentile(minimum,90)))
        result['clips'][name]=dict(cutoff=cutoff,events=int(meta['events']),
            density_pixels=int(dense.sum()),eligible=int(period['eligible_pixels']),
            accepted_at_default=int(period['accepted_pixels']),
            bounds=quartiles,accepted_by_threshold=accepted,
            polarity_shuffled_null=dict(eligible=int(null['eligible_pixels']),
                accepted=int(null['accepted_pixels']),accepted_by_threshold=null_accepted))
        print(name,'eligible',period['eligible_pixels'],'accepted',period['accepted_pixels'],
            'null',null['accepted_pixels'],
            'bounds',quartiles,'by threshold',{k:v['pixels'] for k,v in accepted.items()},flush=True)
    (OUT/'period_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print('wrote',OUT/'period_validation.json')


if __name__=='__main__':main()
