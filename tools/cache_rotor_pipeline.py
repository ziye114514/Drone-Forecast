"""Capture a few original native windows once; no labels, detector or filtering."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from live_decode import native_batches


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    specs=[dict(name='clear',raw='E:/dataset/train/6/media/event.raw',start=26.2,duration=1.2),
           dict(name='side',raw='E:/dataset/train/6/media/event.raw',start=110.,duration=2.),
           dict(name='empty',raw='E:/dataset/train/5/media/event.raw',start=.4,duration=.2)]
    for spec in specs:
        folder=out/spec['name'];folder.mkdir();frames=[];began=time.perf_counter()
        for i,(t,events) in enumerate(native_batches(Path(spec['raw']),spec['start'],spec['duration'],False,
                33333,reader_backend='rolling')):
            np.savez_compressed(folder/f'{i:04d}.npz',events=events,t=np.float64(t))
            frames.append(dict(index=i,t=t,events=len(events)))
        (folder/'manifest.json').write_text(json.dumps(dict(spec=spec,frames=frames,
            elapsed_s=time.perf_counter()-began,original_timestamps_and_polarity=True),indent=2),'utf-8')
        print(spec['name'],len(frames),'cached',flush=True)
    (out/'manifest.json').write_text(json.dumps(specs,indent=2),'utf-8')


if __name__=='__main__':main()
