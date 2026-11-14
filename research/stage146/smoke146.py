"""One observation-only real RAW check of the new local split recovery."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'research/stage146'), str(ROOT/'research/stage129')]
from branches146 import BranchObserver, render_panel
from polarity import counts
from event_drone_forecast.raw_io import open_raw, raw_chunks
from metavision_sdk_core import BaseFrameGenerationAlgorithm
import cv2
import numpy as np

out = ROOT.parent/'output/stage146_threshold_branches_20261024'
rows = {round(r['t'], 6): r for r in (json.loads(v) for v in (out/'clear4/predictions.jsonl').read_text().splitlines())}
reader = open_raw('E:/dataset/train/6/media/event.raw'); h, w = reader.get_size(); observer = BranchObserver(w, h)
for t, events in raw_chunks(reader, 26.4, .4, 20000):
    row = rows[round(t, 6)]; prepared = counts(events, w, h); prepared['on']/=2; prepared['off']/=2
    measured = observer.step(events, t, prepared, row['body'], row['coefficients']['optimized_geometry'], row['predictions']['optimized_geometry'])
    if abs(t-26.66) < 1e-7:
        assert measured['selected_mask'] == 'split_recovery', measured['split_recovery']
        assert measured['geometry']['state'] == 'four'
        image = np.zeros((h, w, 3), np.uint8); BaseFrameGenerationAlgorithm.generate_frame(events, image)
        cv2.imwrite(str(out/'merged_recovered_smoke.png'), render_panel(image, observer, measured, t))
        (out/'split_smoke.json').write_text(json.dumps(dict(t=t, state=measured['geometry']['state'],
            provenance=measured['split_recovery'], observer_only=True, new_NN=False), indent=2))
        print(json.dumps(dict(t=t, state=measured['geometry']['state'], split=measured['split_recovery'])), flush=True)
