"""Final result integrity: source versions, cutoff, provenance and complete video."""
from pathlib import Path
import hashlib
import json
import cv2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent/'output/stage146_threshold_branches_20261024'
checks = {}
for case, expected in [('clear4_final', 400), ('side4_final', 475), ('train2_4_final', 300), ('fred0_4_final', 150)]:
    directory = OUT/case
    protocol = json.loads((directory/'protocol.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == expected
    for source, digest in protocol['sources'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
    for row in rows:
        assert row['max_event_t'] is None or row['max_event_t'] < round(row['t']*1e6)
        geometry = row['rotor_branch']['geometry']
        assert geometry['observed_count'] == len(geometry['observed_rotors'])
        if geometry['state'] == 'three_completed':
            assert geometry['observed_count'] == 3 and len(geometry['inferred_ids']) == 1
            assert not geometry['inferred_center_is_measurement']
        if row['rotor_branch']['thrust_proxy']['valid']:
            assert row['rotor_branch']['thrust_proxy']['actual_force_N'] is None
        expert = row['rotor_branch']['expert']
        assert all(0 <= w <= .25 for w in expert['weights_by_horizon'].values())
    video = cv2.VideoCapture(str(directory/'predictions.mp4')); decoded = 0
    while True:
        ok, _ = video.read()
        if not ok: break
        decoded += 1
    video.release(); assert decoded == expected
    checks[case] = dict(records=expected, decoded_video_frames=decoded,
                        all_source_hashes_match=True, causal_event_cutoff=True,
                        inferred_is_not_observed=True, weights_bounded=True)
controls = json.loads((OUT/'controls.json').read_text()); assert all(controls.values())
panels = json.loads((OUT/'panels146.json').read_text())
assert all(p['exact_spatial_mask_reconstruction'] for p in panels.values())
checks['controls'] = controls
checks['all_display_masks_exact'] = True
(OUT/'checks146.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(checks, ensure_ascii=False), flush=True)
