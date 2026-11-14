"""Render full native 33ms packets, keeping the existing 20ms control cache."""
from pathlib import Path
import json
import sys
import numpy as np
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_core import BaseFrameGenerationAlgorithm


def main(output):
    out = Path(output)
    rows = json.loads((out / 'frames.json').read_text('utf-8'))
    seq = None
    for row in rows:
        if seq != row['sequence']:
            seq = row['sequence']
            noise = ActivityNoiseFilterAlgorithm(1280,720,10000)
            buffer = noise.get_empty_output_buffer()
            directory = out / 'frames33' / seq
            directory.mkdir(parents=True)
        packet = np.load(row['path'])
        noise.process_events(packet['events'], buffer)
        image = np.empty((720,1280,3),np.uint8)
        clean = buffer.numpy()
        BaseFrameGenerationAlgorithm.generate_frame(clean, image)
        np.savez_compressed(directory / f"{row['index']:04d}.npz", image=image, t=row['t'])
    print('Native33ms frames saved; no future events or reconstructed brightness')


if __name__ == '__main__':main(sys.argv[1])
