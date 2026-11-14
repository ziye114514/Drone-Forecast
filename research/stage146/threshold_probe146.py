"""Observation-only threshold study with frozen *current* detector ROIs.

No forecasting scores or future labels are used to choose a mask. This is not
an end-to-end deployment run; the selected setting must then run from RAW.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'research/stage146'), str(ROOT/'research/stage129')]
from branches146 import ThresholdFrontend, BranchGeometry, render_panel
from boundary129 import bounded
from polarity import counts
from event_drone_forecast.raw_io import open_raw, raw_chunks
from metavision_sdk_core import BaseFrameGenerationAlgorithm

OUT = ROOT.parent/'output/stage146_threshold_branches_20261024'
CASES = dict(
    clear_merge=dict(raw='E:/dataset/train/6/media/event.raw', start=26.4, target=26.66,
                     cache=OUT/'clear4/predictions.jsonl'),
    clear=dict(raw='E:/dataset/train/6/media/event.raw', start=22.8, target=23.02,
               cache=OUT/'clear_fixed2_pilot/predictions.jsonl'),
    four_review_C=dict(raw='E:/dataset/train/6/media/event.raw', start=111.8, target=112.02,
                       cache=OUT/'side_fixed2/predictions.jsonl'),
    side=dict(raw='E:/dataset/train/6/media/event.raw', start=117.8, target=118.12,
              cache=OUT/'side_fixed2/predictions.jsonl'),
    fred0=dict(raw=str(OUT/'input/fred0/events.raw'), start=109.4, target=109.62,
               cache=OUT/'fred0_fixed2/predictions.jsonl'))
VARIANTS = dict(fixed2=(2., False), fixed4=(4., False), fixed8=(8., False), adaptive70=(2., True))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--case', choices=list(CASES))
    selected = parser.parse_args().case
    folder = OUT/'threshold_probe'; folder.mkdir(exist_ok=True)
    summary = {}
    for name, case in CASES.items():
        if selected and name != selected: continue
        rows = {round(r['t'], 6): r for r in
                (json.loads(line) for line in case['cache'].read_text().splitlines())}
        reader = open_raw(case['raw']); h, w = reader.get_size()
        fronts = {key: ThresholdFrontend(w, h, threshold, adaptive)
                  for key, (threshold, adaptive) in VARIANTS.items()}
        geometry = {key: BranchGeometry() for key in fronts}
        summary[name] = {key: dict(states={}, target=None) for key in fronts}
        for t, events in raw_chunks(reader, case['start'], .4, 20000):
            row = rows.get(round(t, 6))
            if row is None: continue
            prepared = counts(events, w, h)
            for key, front in fronts.items():
                front.set_hint(events, row['body'], t)
                observation, _ = front.detect_prepared(events, t, prepared)
                rotors, _ = bounded(observation.get('rotors', []), w, h)
                measured = geometry[key].step(t, row['body'], rotors)
                states = summary[name][key]['states']; states[measured['state']] = states.get(measured['state'], 0)+1
                if abs(t-case['target']) < 1e-7:
                    item = dict(geometry=measured, threshold=front.support_threshold, roi=front.current_roi)
                    summary[name][key]['target'] = item
                    image = np.zeros((h, w, 3), np.uint8)
                    BaseFrameGenerationAlgorithm.generate_frame(events, image)
                    class Display: mask=front.mask146
                    cv2.imwrite(str(folder/f'{name}_{key}.png'), render_panel(image, Display(), item, t))
        print(json.dumps(dict(case=name, results={key: dict(states=v['states'], target=None if v['target'] is None else
              dict(state=v['target']['geometry']['state'], threshold=v['target']['threshold'])) for key, v in summary[name].items()})), flush=True)
    destination = folder/(f'results_{selected}.json' if selected else 'results.json')
    destination.write_text(json.dumps(summary, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist(), indent=2), encoding='utf-8')


if __name__ == '__main__': main()
