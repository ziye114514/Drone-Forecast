"""Same cached native-derived coordinates: signed ABI must preserve original."""
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from reference_api import AuthorTracker,OUT
data=np.load(OUT/'weighted_ablation/rectified_events.npz')
old=json.loads((OUT/'sign_hypotheses/results.json').read_text(encoding='utf-8'))
dll=OUT/'adaptive_signed/build/Release/helix_reference.dll'
parent=json.loads((OUT/'real_probe.json').read_text(encoding='utf-8'))
worst=0.
for i in range(4):
    for sign in (1,-1):
        with AuthorTracker(sign*60*parent['seeds'][str(i)]['frequency_hz'],2,dll_path=dll,
                           canvas=(256,256,128,128,80),signed_coordinates=True) as tracker:
            for k in range(20):
                e=data[f'rotor{i}_tick{k}'];value=tracker.feed(e['x'],e['y'],e['t'],e['p'],solve_pose=k%2==1)
                ref=old['records'][k]['results'][f'{i}:{sign}']
                for key in ['phase_hz','weighted_phase_concentration','predictive_phase_concentration']:
                    worst=max(worst,abs(value[key]-ref[key]))
assert worst<1e-4,worst
(OUT/'adaptive_signed/signed_abi_control.json').write_text(json.dumps(dict(maximum_difference=worst,
    cached_native_payloads=80,hypotheses=8,author_algorithm_modified=False)),encoding='utf-8')
print('signed ABI control maximum difference:',worst,flush=True)
