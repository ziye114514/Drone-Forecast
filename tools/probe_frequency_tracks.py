"""Can a causal temporal selector stabilise the per-region frequency?

The per-window template search picks among harmonics and often jumps between
values a factor-of-two apart. Each region therefore carries two separated
frequency hypotheses per eligible pixel. This probe asks a narrow question:

    using ONLY past assigned values, does choosing the hypothesis closest to the
    track's running median reduce the frequency spread compared with the value
    the window reported?

No future value is used, no rotor identity is claimed, and a smoothed value is
reported as smoothed rather than as a better measurement.
"""
from pathlib import Path
from collections import deque
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT)]
from live_decode import native_batches
from rotor_support_front import RegionTracker,RotorSupportTracker

BASE=ROOT.parent/'output'
OUT=BASE/'rotor_support_20261101'
CLIPS=[dict(name='clear_26.2_27.0',raw='E:/dataset/train/6/media/event.raw',start=26.20,duration=.80),
    dict(name='side_110.9_111.7',raw='E:/dataset/train/6/media/event.raw',start=110.90,duration=.80)]
MAX_FACTOR=1.6


def relative_std(values):
    values=[v for v in values if v]
    if len(values)<2:return None
    mean=float(np.mean(values))
    return None if mean<=0 else float(np.std(values)/mean)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(clips={},causal_selector=True,max_factor=MAX_FACTOR,
        note='the selector only sees values assigned at earlier cutoffs; smoothed values are not measurements')
    for spec in CLIPS:
        started=time.perf_counter();tracker=RotorSupportTracker(self_motion=True);regions=RegionTracker()
        queue=deque();end=spec['start']+spec['duration']
        history={};reported={};assigned={};smoothed={};switches=0;switched_frames=0;frames=0
        candidate_differs=0
        for t,events in native_batches(Path(spec['raw']),spec['start']-.25,spec['duration']+.3,False,
                33333,reader_backend='rolling'):
            queue.append(events.copy())
            while queue and (not len(queue[0]) or queue[0]['t'][-1]<(t-.22)*1e6):queue.popleft()
            if t<spec['start']-1e-9:continue
            if t>end+1e-9:break
            frames+=1
            frame=tracker.step(np.concatenate(queue),float(t))
            shift=frame['accumulation'].get('shift_px') or (0.,0.)
            update=regions.update(frame['accumulated_components'],float(t),shift_px=shift)
            for track in update['tracks']:
                reported_value=track['median_frequency_hz']
                if not reported_value:continue
                track_id=track['id']
                past=history.get(track_id,[])
                chosen=reported_value
                if len(past)>=2:
                    median=float(np.median(past))
                    options=[reported_value]
                    for value,count in (track.get('candidate_frequencies') or []):
                        options.append(float(value))
                    valid=[v for v in options if v>0 and abs(np.log(v/median))<=np.log(MAX_FACTOR)]
                    if valid:
                        chosen=min(valid,key=lambda v:abs(np.log(v/median)))
                        if abs(np.log(chosen/reported_value))>1e-9:switches+=1;switched_frames+=1
                if abs(np.log(chosen/reported_value))>1e-9:candidate_differs+=1
                past_reported=reported.get(track_id,[])
                smoothed_value=reported_value
                if len(past_reported)>=2:
                    smoothed_value=float(np.median(past_reported[-5:]))
                smoothed.setdefault(track_id,[]).append(smoothed_value)
                history.setdefault(track_id,[]).append(chosen)
                reported.setdefault(track_id,[]).append(reported_value)
                assigned.setdefault(track_id,[]).append(chosen)
        long_ids=[track_id for track_id,values in reported.items() if len(values)>=5]
        def spreads(table):
            values=[relative_std(table[i]) for i in long_ids]
            values=[v for v in values if v is not None]
            return dict(tracks=len(values),median=None if not values else float(np.median(values)))
        result['clips'][spec['name']]=dict(spec=spec,frames=frames,long_tracks=len(long_ids),
            reported=spreads(reported),assigned=spreads(assigned),
            switches=switches,switch_rate=None if not frames else switches/max(frames,1),
            smoothed_reported=spreads(smoothed),candidate_differs=candidate_differs,
            candidate_share=None if not frames else candidate_differs/max(frames,1),
            samples={str(track_id):dict(reported=reported[track_id][:8],assigned=assigned[track_id][:8],
                smoothed=smoothed.get(track_id,[])[:8],
                reported_spread=relative_std(reported[track_id]),assigned_spread=relative_std(assigned[track_id]))
                for track_id in long_ids[:5]},
            elapsed_s=round(time.perf_counter()-started,2))
        entry=result['clips'][spec['name']]
        print(spec['name'],'long tracks',entry['long_tracks'],'reported',entry['reported'],
            'assigned',entry['assigned'],'switches',entry['switches'],flush=True)
    (OUT/'frequency_tracks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print('wrote',OUT/'frequency_tracks.json')


if __name__=='__main__':main()
