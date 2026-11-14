"""Do support fragments lie on one disc? Circle-fit agglomeration, no fixed gap.

For each cutoff the stable support is labelled into fragments; fragments are
merged only while a single circle still explains their pixels (low rms/r AND
enough angular coverage, because a short arc constrains nothing). This replaces
the fixed-gap merging that the previous round rejected.

Frozen rotor conics are used only as a diagnostic reference, never as input.
"""
from pathlib import Path
from collections import deque
import json,sys,time
import numpy as np
from scipy.ndimage import label
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT)]
from live_decode import native_batches
from rotor_support_front import RotorSupportTracker,fit_disc_candidates

BASE=ROOT.parent/'output'
OUT=BASE/'rotor_support_20261101'
CLIPS=[dict(name='clear_26.4_26.95',raw='E:/dataset/train/6/media/event.raw',start=26.40,duration=.55,
        predictions='motion_consistency_30px_20261028/RAW_clear_release_candidate2s/predictions.jsonl'),
    dict(name='side_110.9_111.45',raw='E:/dataset/train/6/media/event.raw',start=110.90,duration=.55,
        predictions='motion_consistency_30px_20261028/RAW_side_release_candidate8s/predictions.jsonl')]


def fragment_points(stable_support):
    labels,count=label(np.asarray(stable_support,bool))
    points={}
    for index in range(1,count+1):
        ys,xs=np.nonzero(labels==index)
        if len(xs):points[index]=np.column_stack((xs,ys)).astype(float)
    return points


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(clips={},frozen_reference_is_diagnostic_only=True,rotor_count_claimed=False,
        blade_identity_claimed=False,
        method='agglomerative circle fit; merge accepted only when rms/r<=0.12 and coverage>=120deg')
    for spec in CLIPS:
        started=time.perf_counter()
        tracker=RotorSupportTracker(self_motion=True);frames=[]
        queue=deque();end=spec['start']+spec['duration']
        for t,events in native_batches(Path(spec['raw']),spec['start']-.25,spec['duration']+.3,False,
                33333,reader_backend='rolling'):
            queue.append(events.copy())
            while queue and (not len(queue[0]) or queue[0]['t'][-1]<(t-.22)*1e6):queue.popleft()
            if t<spec['start']-1e-9:continue
            if t>end+1e-9:break
            frame=tracker.step(np.concatenate(queue),float(t))
            candidates=fit_disc_candidates(fragment_points(frame['stable_support']))
            frames.append(dict(t=float(t),fragments=candidates['fragments'],
                disc_candidates=candidates['disc_candidates'],
                candidates=[dict(members=item['members'],accepted=item['accepted'],
                    cx=round(item['fit']['cx'],1),cy=round(item['fit']['cy'],1),
                    r=round(item['fit']['r'],1),coverage_deg=round(item['fit']['coverage_deg'],1),
                    rms_over_r=round(item['fit']['rms_over_r'],3),points=item['fit']['points'])
                    for item in candidates['candidates']],
                merges=len(candidates['merges']),support_pixels=int(np.count_nonzero(frame['stable_support']))))
        rows=[json.loads(line) for line in (BASE/spec['predictions']).read_text('utf-8').splitlines()]
        last=frames[-1];row=min(rows,key=lambda item:abs(item['t']-last['t']))
        rotors=[r for r in row['geometry'].get('rotors',[]) if r.get('current_observation',True)]
        diagnostic=None
        if rotors:
            distances=[]
            for candidate in last['candidates']:
                if not candidate['accepted']:continue
                distances.append(min(float(np.hypot(candidate['cx']-r['cx'],candidate['cy']-r['cy'])/
                    max(float(r.get('major',1))/2,1e-6)) for r in rotors))
            diagnostic=dict(t=last['t'],frozen_mode=row['structure'].get('mode'),frozen_rotors=len(rotors),
                accepted_candidates=len(distances),
                min_distance_semi_major=None if not distances else float(min(distances)),
                max_distance_semi_major=None if not distances else float(max(distances)),
                note='distance from each accepted disc candidate centre to the nearest frozen rotor centre')
        result['clips'][spec['name']]=dict(spec=spec,cutoffs=len(frames),frames=frames,
            diagnostic=diagnostic,elapsed_s=round(time.perf_counter()-started,2))
        counts=[frame['disc_candidates'] for frame in frames]
        print(spec['name'],'cutoffs',len(frames),'disc candidates',counts,
            'diagnostic',diagnostic,flush=True)
    (OUT/'disc_geometry.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print('wrote',OUT/'disc_geometry.json')


if __name__=='__main__':main()
