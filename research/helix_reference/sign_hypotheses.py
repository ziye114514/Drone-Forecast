"""Two physically possible initial rotation signs, SAME saved current events."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from reference_api import AuthorTracker,OUT

parent=json.loads((OUT/'real_probe.json').read_text(encoding='utf-8'))
frozen=json.loads((OUT/'weighted_ablation/results.json').read_text(encoding='utf-8'))
out=OUT/'sign_hypotheses';out.mkdir(exist_ok=False)
event_file=OUT/'weighted_ablation/rectified_events.npz'
data=np.load(event_file)
trackers={(i,sign):AuthorTracker(sign*parent['seeds'][str(i)]['frequency_hz']*60,2) for i in range(4) for sign in (1,-1)}
records=[];legacy_delta=0.
try:
    for k in range(20):
        frame={}
        for i in range(4):
            e=data[f'rotor{i}_tick{k}']
            for sign in (1,-1):
                value=trackers[i,sign].feed(e['x'],e['y'],e['t'],e['p'],solve_pose=k%2==1)
                frame[f'{i}:{sign}']=value
                if sign==1:legacy_delta=max(legacy_delta,abs(value['phase_hz']-frozen['records'][k]['results'][f'{i}:refine']['phase_hz']))
        records.append(dict(t=frozen['records'][k]['t'],results=frame))
finally:
    for tracker in trackers.values():tracker.close()
assert legacy_delta<1e-4
summary={}
for i in range(4):
    modes={}
    for sign in (1,-1):
        seq=[r['results'][f'{i}:{sign}'] for r in records]
        modes[str(sign)]=dict(phase_hz_median=float(np.median([r['phase_hz'] for r in seq])),
            weighted_R_median=float(np.median([r['weighted_phase_concentration'] for r in seq])),
            split_R_median=float(np.median([r['polarity_split_weighted_concentration'] for r in seq])),
            qualified_ticks=sum(r['weighted_phase_concentration']>=.5 and r['informative_ring_sectors']>=9 for r in seq),
            physical_image_signs=sorted(set(int(np.sign(r['angular_velocity_rad_s'])*r['rotation_sign']) for r in seq)))
    selected=[]
    for row in records:
        choices=[row['results'][f'{i}:{s}'] for s in (1,-1)]
        admissible=[r for r in choices if r['weighted_phase_concentration']>=.5 and r['informative_ring_sectors']>=9]
        selected.append(max(admissible,key=lambda r:r['weighted_phase_concentration']) if admissible else None)
    summary[str(i)]=dict(modes=modes,supported_with_either_sign=sum(r is not None for r in selected),
                        selected_phase_hz_median=float(np.median([r['phase_hz'] for r in selected if r is not None])) if any(r is not None for r in selected) else None)
(out/'results.json').write_text(json.dumps(dict(scope='Observation replay, two physical sign hypotheses, same event payload and original configuration. No real mechanical RPM truth.',
    original_positive_hypothesis_max_difference=legacy_delta,summary=summary,records=records),indent=2),encoding='utf-8')
print(json.dumps(summary),flush=True)
