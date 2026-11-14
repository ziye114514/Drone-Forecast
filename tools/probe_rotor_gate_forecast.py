"""Does the rotor-only front end help as a GATE on the existing thrust input?

The front end cannot yet give a disc scale, so it provides no new magnitude. What
it can give is an independent existence check: at this cutoff, is there a tracked
support region with a temporally consistent frequency?

  base   frozen rows, unchanged
  gated  same rows, but force.input_u is dropped wherever the rotor gate is closed

Everything is causal: the gate runs at 10Hz from the same past events, and the
labels are opened only for scoring. A closed gate never invents an input.
"""
from pathlib import Path
from collections import deque
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_decode import native_batches
from rotor_support_front import RegionTracker,RotorSupportTracker
from probe_same_window_motion import replay
from compare_final_pipeline import score

BASE=ROOT.parent/'output'
OUT=BASE/'rotor_support_20261101'
SPECS=[dict(name='side_110_118',raw='E:/dataset/train/6/media/event.raw',start=110.0,duration=8.0,
        predictions='motion_consistency_30px_20261028/RAW_side_release_candidate8s/predictions.jsonl',
        labels='dense_labels_side_110_1195'),
    dict(name='train2_40_46',raw='E:/dataset/train/2/aligned/event.raw',start=40.0,duration=6.0,
        predictions='thrust_rate_review_20261029/experiment/other_baseline.jsonl',
        labels='dense_labels_train2_40_46')]
STRIDE=3
GATE_WINDOW_S=.12
OPTIONS=dict(curved=True,evidence_gain=True,memory=True,endpoint=True,memory_gaps=True,weak_support=True)


def run_clip(SPEC):
    started=time.perf_counter()
    tracker=RotorSupportTracker(tau_s=.3,min_level=.35,self_motion=True)
    regions=RegionTracker();queue=deque();samples=[];history={}
    end=SPEC['start']+SPEC['duration'];index=0
    for t,events in native_batches(Path(SPEC['raw']),SPEC['start']-.3,SPEC['duration']+.4,False,
            33333,reader_backend='rolling'):
        queue.append(events.copy())
        while queue and (not len(queue[0]) or queue[0]['t'][-1]<(t-.26)*1e6):queue.popleft()
        if t<SPEC['start']-1e-9:continue
        if t>end+1e-9:break
        index+=1
        if index%STRIDE:continue
        frame=tracker.step(np.concatenate(queue),float(t))
        shift=frame['accumulation'].get('shift_px') or (0.,0.)
        update=regions.update(frame['accumulated_components'],float(t),shift_px=shift)
        qualified=[track for track in update['tracks'] if track['qualified']]
        frequencies=[]
        for track in qualified:
            track_id=track['id'];reported=track['median_frequency_hz']
            if not reported:continue
            past=history.get(track_id,[])
            chosen=reported
            if len(past)>=2:
                median=float(np.median(past))
                options=[reported]+[float(v) for v,c in (track.get('candidate_frequencies') or [])]
                valid=[v for v in options if v>0 and abs(np.log(v/median))<=np.log(1.6)]
                if valid:chosen=min(valid,key=lambda v:abs(np.log(v/median)))
            history.setdefault(track_id,[]).append(chosen)
            frequencies.append(chosen)
        samples.append(dict(t=float(t),qualified=len(qualified),
            frequencies=[round(v,1) for v in frequencies],
            gate=bool(qualified and frequencies)))
    rows=[json.loads(line) for line in (BASE/SPEC['predictions']).read_text('utf-8').splitlines()]
    times=np.array([sample['t'] for sample in samples])
    def gate_at(t):
        if not len(times):return None
        position=min(range(len(times)),key=lambda i:abs(times[i]-t))
        if abs(times[position]-t)>GATE_WINDOW_S:return None
        return samples[position]
    gated=[]
    for row in rows:
        sample=gate_at(row['t'])
        closed=bool(sample is not None and not sample['gate'])
        gated.append(dict(row,force=dict(row['force'],input_u=None if closed else row['force'].get('input_u')),
            rotor_gate=dict(t=None if sample is None else sample['t'],
                gate=None if sample is None else sample['gate'],closed=closed)))
    base_rows=[dict(row) for row in rows]
    base_values,_=replay(base_rows,False,**OPTIONS)
    gated_values,_=replay(gated,False,**OPTIONS)
    truth=sorted(json.loads((BASE/SPEC['labels']/'labels.json').read_text('utf-8'))['records'].values(),
        key=lambda item:item['t'])
    base_score=score(base_values,truth);gated_score=score(gated_values,truth)
    changed=sum(1 for before,after in zip(rows,gated)
        if (before['force'].get('input_u') is None)!=(after['force'].get('input_u') is None))
    result=dict(spec=SPEC,stride=STRIDE,sample_hz=1000/(STRIDE*33.333),samples=len(samples),
        gate_true=sum(1 for s in samples if s['gate']),gate_false=sum(1 for s in samples if not s['gate']),
        ticks_with_gate_sample=sum(1 for row in rows if gate_at(row['t']) is not None),
        ticks_closed=changed,
        base=base_score['physical_future_error_px'],gated=gated_score['physical_future_error_px'],
        base_area=base_score['box_center_error_px'],gated_area=gated_score['box_center_error_px'],
        samples_detail=[dict(t=s['t'],gate=s['gate'],qualified=s['qualified'],frequencies=s['frequencies'])
            for s in samples],
        elapsed_s=round(time.perf_counter()-started,1),
        note='the gate only removes input; it never adds one, and no label enters it')
    return result


def main():
    results={}
    for spec in SPECS:
        result=run_clip(spec)
        results[spec['name']]=result
        print(spec['name'],json.dumps(dict(samples=result['samples'],gate_true=result['gate_true'],
            gate_false=result['gate_false'],ticks_closed=result['ticks_closed'],
            base=result['base']['0.4']['mean'],gated=result['gated']['0.4']['mean'],
            elapsed_s=result['elapsed_s']),ensure_ascii=False),flush=True)
    payload=dict(clips=results,stride=STRIDE,gate_window_s=GATE_WINDOW_S,
        note='the gate only removes input; it never adds one, and no label enters it')
    (OUT/'rotor_gate_forecast.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print('wrote',OUT/'rotor_gate_forecast.json')


if __name__=='__main__':main()
