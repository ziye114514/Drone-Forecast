"""Cross-cutoff region tracks over the rotor-only support (no identity claims).

Each cutoff: rotor support -> self-motion accumulation -> connected regions ->
greedy association with age/misses. Outputs per-cutoff tracks with age and a
short frequency history, plus aggregate continuity (how many tracks survive,
how stable their frequency and size are). Tracks are geometric evidence only:
no rotor identity, no rotor count, no RPM.
"""
from pathlib import Path
from collections import deque
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT)]
from live_decode import native_batches
from rotor_support_front import PARAMS,RegionTracker,RotorSupportTracker,rotor_support

BASE=ROOT.parent/'output'
OUT=BASE/'rotor_support_20261101'
SEQUENCES=[dict(name='clear_26.2_27.0',raw='E:/dataset/train/6/media/event.raw',start=26.20,duration=.80,
        predictions='motion_consistency_30px_20261028/RAW_clear_release_candidate2s/predictions.jsonl'),
    dict(name='side_110.9_111.7',raw='E:/dataset/train/6/media/event.raw',start=110.90,duration=.80,
        predictions='motion_consistency_30px_20261028/RAW_side_release_candidate8s/predictions.jsonl')]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(sequences={},identity_tracking=False,rotor_count_claimed=False,
        mechanical_RPM_measured=False,
        note='tracks are geometric continuity only; ids are local to one run and do not name rotors')
    for spec in SEQUENCES:
        started=time.perf_counter()
        tracker=RotorSupportTracker(self_motion=True);regions=RegionTracker()
        queue=deque();end=spec['start']+spec['duration'];frames=[]
        for t,events in native_batches(Path(spec['raw']),spec['start']-.25,spec['duration']+.3,False,
                33333,reader_backend='rolling'):
            queue.append(events.copy())
            while queue and (not len(queue[0]) or queue[0]['t'][-1]<(t-.22)*1e6):queue.popleft()
            if t<spec['start']-1e-9:continue
            if t>end+1e-9:break
            frame=tracker.step(np.concatenate(queue),float(t))
            shift=frame['accumulation'].get('shift_px') or (0.,0.)
            update=regions.update(frame['accumulated_components'],float(t),shift_px=shift)
            update['source']=frame['accumulation'].get('source')
            update['shift_px']=shift
            update['support_pixels']=int(np.count_nonzero(frame['stable_support']))
            frames.append(update)
        final=list(regions.tracks.values())
        ages=[track['last_t']-track['first_t'] for track in final]
        long_tracks=[track for track in final if track['hits']>=5]
        def frequency_spread(track):
            values=[v for v in track['frequency_history'] if v]
            if len(values)<2:return None
            mean=float(np.mean(values))
            return None if mean<=0 else float(np.std(values)/mean)
        spreads=[s for s in (frequency_spread(t) for t in long_tracks) if s is not None]
        areas=[t['pixels'] for t in long_tracks]
        boxes=[t['bbox'] for t in long_tracks]
        result['sequences'][spec['name']]=dict(
            spec=spec,cutoffs=len(frames),
            tracks_total=len(final),tracks_hits_ge5=len(long_tracks),
            median_age_s=None if not ages else float(np.median(ages)),
            confirmed_counts=[frame['confirmed'] for frame in frames],
            total_counts=[frame['total'] for frame in frames],
            associations=[frame['associations'] for frame in frames],
            unmatched=[frame['unmatched_components'] for frame in frames],
            shift_sources=[frame['source'] for frame in frames],
            shifts=[[round(float(v),2) for v in frame['shift_px']] for frame in frames],
            frequency_relative_std_median=None if not spreads else float(np.median(spreads)),
            long_track_pixels=[int(v) for v in areas],
            long_track_boxes=boxes,
            frames=[dict(t=frame['t'],confirmed=frame['confirmed'],total=frame['total'],
                tracks=[{k:v for k,v in track.items() if k!='frequency_history'} for track in frame['tracks']])
                for frame in frames],
            elapsed_s=round(time.perf_counter()-started,2))
        entry=result['sequences'][spec['name']]
        print(spec['name'],'tracks',entry['tracks_total'],'>=5 hits',entry['tracks_hits_ge5'],
            'confirmed',entry['confirmed_counts'],'median age',entry['median_age_s'],flush=True)
    (OUT/'tracks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print('wrote',OUT/'tracks.json')


if __name__=='__main__':main()
