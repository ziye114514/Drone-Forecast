"""Freeze a small RAW frame cache and audit positive-only localization denominators.

Annotations are used after predictions, never for selecting inference crops.
"""
from pathlib import Path
import collections
import hashlib
import json
import os
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from live_decode import native_batches

SOURCE = ROOT.parent / 'output/stability_audit_20261031'
CASES = [('FRED0', 'RAW_FRED0_recovery4s', 'FRED0_recovery_scores.json', 'candidate'),
         ('FRED7', 'RAW_FRED7_final4s', 'FRED7_final_scores.json', 'candidate'),
         ('FRED9', 'RAW_FRED9_budget4s', 'FRED9_budget_scores.json', 'candidate')]


def main(output):
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
    from metavision_sdk_core import BaseFrameGenerationAlgorithm
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    frames = []
    audits = {}
    for name, folder, score_file, arm in CASES:
        prior = SOURCE / folder
        protocol = json.loads((prior / 'protocol.json').read_text('utf-8'))
        rows = [json.loads(x) for x in (prior / 'predictions.jsonl').read_text('utf-8').splitlines()]
        score = json.loads((SOURCE / score_file).read_text('utf-8'))[arm]
        anchors = score['anchors']
        missing = [a for a in anchors if not a['detected']]
        lookup = {round(r['t'], 6): r for r in rows}
        audits[name] = dict(
            positive_annotated_ticks=len(anchors),
            current_output_ticks=score['current_outputs'],
            missing_on_positive_ticks=len(missing),
            IoU50_fraction_including_misses=score['IoU50_fraction_including_misses'],
            missing_tracking_reasons=dict(collections.Counter(
                lookup[round(a['t'], 6)]['tracking'].get('reason') for a in missing)),
            missed_positive_times=[a['t'] for a in missing],
            missing_is_not_background_fraction=True,
            annotation_interpolation_max_gap_s=.134,
            source_score=str(SOURCE / score_file))
        directory = out / 'frames' / name
        directory.mkdir(parents=True)
        noise = ActivityNoiseFilterAlgorithm(1280, 720, 10000)
        buffer = noise.get_empty_output_buffer()
        image = np.empty((720, 1280, 3), np.uint8)
        iterator = native_batches(protocol['raw'], protocol['actual_iterator_start'],
            protocol['actual_duration'], False, protocol['native_step_us'])
        for i, (t, events) in enumerate(iterator):
            if i >= len(rows):
                raise ValueError('Extra RAW frame')
            assert abs(t - rows[i]['t']) < 1e-7
            noise.process_events(events, buffer)
            clean = buffer.numpy()
            recent = clean[clean['t'] >= round(t * 1e6) - 20000]
            BaseFrameGenerationAlgorithm.generate_frame(recent, image)
            path = directory / f'{i:04d}.npz'
            np.savez_compressed(path, image=image, events=events, t=t)
            frames.append(dict(sequence=name, index=i, t=t, path=str(path)))
        iterator.close()
        print(json.dumps(dict(sequence=name, cached_frames=i+1, **audits[name]), ensure_ascii=False), flush=True)
    (out / 'positive_denominators.json').write_text(json.dumps(audits, indent=2), 'utf-8')
    (out / 'frames.json').write_text(json.dumps(frames, indent=2), 'utf-8')
    (out / 'protocol.json').write_text(json.dumps(dict(
        purpose='Reuse fixed full-frame model plus conditional single ROI zoom; EV-Tach-inspired separate pilot',
        model_sha256='ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4',
        no_annotation_crops=True, model_not_retrained=True,
        crop_policy='Existing live_roi_refine; recent own detector reference, no ground truth',
        metric='Positive localization IoU>=.5 and source .4s FDE; not all-scene AP',
        full_frame_reference_every_frame=True, offline_inference_is_not_live30Hz=True,
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2), 'utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
