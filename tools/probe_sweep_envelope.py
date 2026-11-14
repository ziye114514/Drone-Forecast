"""Long-baseline sweep envelope: accumulate period support over 0.8s in object coordinates.

Single-window arcs cannot constrain a circle (sagitta 2.9px for a 30px chord on a
40px radius disc, i.e. at the pixel-band thickness). The blades sweep the whole
disc within a few revolutions, so a long, motion-compensated OR-accumulation
should reveal the disc envelope instead of an arc. This probe measures that
envelope and its size; the frozen conics stay diagnostic only.
"""
from pathlib import Path
from collections import deque
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
from live_decode import native_batches
from rotor_support_front import RotorSupportTracker

BASE=ROOT.parent/'output'
OUT=BASE/'rotor_support_20261101'
CLIPS=[dict(name='clear_26.2_27.0',raw='E:/dataset/train/6/media/event.raw',start=26.20,duration=.80,
        predictions='motion_consistency_30px_20261028/RAW_clear_release_candidate2s/predictions.jsonl'),
    dict(name='side_110.9_111.7',raw='E:/dataset/train/6/media/event.raw',start=110.90,duration=.80,
        predictions='motion_consistency_30px_20261028/RAW_side_release_candidate8s/predictions.jsonl')]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(clips={},frozen_reference_is_diagnostic_only=True,rotor_count_claimed=False,
        mechanical_RPM_measured=False,
        method='self-motion aligned OR accumulation of the period mask over0.8s (tau effectively infinite)')
    for spec in CLIPS:
        started=time.perf_counter()
        tracker=RotorSupportTracker(tau_s=1e6,min_level=.5,self_motion=True)
        queue=deque();end=spec['start']+spec['duration'];frames=[]
        for t,events in native_batches(Path(spec['raw']),spec['start']-.25,spec['duration']+.3,False,
                33333,reader_backend='rolling'):
            queue.append(events.copy())
            while queue and (not len(queue[0]) or queue[0]['t'][-1]<(t-.22)*1e6):queue.popleft()
            if t<spec['start']-1e-9:continue
            if t>end+1e-9:break
            frame=tracker.step(np.concatenate(queue),float(t))
            found=frame['accumulated_components']
            frames.append(dict(t=float(t),components=len(found),
                pixels=[item['pixels'] for item in found],
                boxes=[item['bbox'] for item in found],
                support_pixels=int(np.count_nonzero(frame['stable_support'])),
                shift=frame['accumulation'].get('shift_px')))
        last=frames[-1]
        rows=[json.loads(line) for line in (BASE/spec['predictions']).read_text('utf-8').splitlines()]
        row=min(rows,key=lambda item:abs(item['t']-last['t']))
        rotors=[r for r in row['geometry'].get('rotors',[]) if r.get('current_observation',True)]
        diagnostic=None
        if rotors:
            expected=sum(np.pi*(float(r['major'])/2)*(float(r['minor'])/2) for r in rotors)
            widths=[b[2]-b[0] for b in last['boxes']]
            heights=[b[3]-b[1] for b in last['boxes']]
            major=[float(r['major']) for r in rotors]
            minor=[float(r['minor']) for r in rotors]
            diagnostic=dict(t=last['t'],frozen_mode=row['structure'].get('mode'),
                frozen_rotors=len(rotors),frozen_major=sorted(round(v,1) for v in major),
                frozen_minor=sorted(round(v,1) for v in minor),
                expected_disc_area_px=float(expected),envelope_support_pixels=last['support_pixels'],
                envelope_components=last['components'],
                component_width_px=sorted(widths),component_height_px=sorted(heights),
                area_ratio=None if not expected else float(last['support_pixels']/expected),
                note='expected area sums the frozen conic areas; ratio near1 means no gross smear')
        result['clips'][spec['name']]=dict(spec=spec,cutoffs=len(frames),frames=frames,
            diagnostic=diagnostic,elapsed_s=round(time.perf_counter()-started,2))
        print(spec['name'],'cutoffs',len(frames),'components',[f['components'] for f in frames][-6:],
            'diag',diagnostic,flush=True)
    (OUT/'sweep_envelope.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print('wrote',OUT/'sweep_envelope.json')


if __name__=='__main__':main()
