"""Read-only RAW capture, preserving every native event in uncompressed packets."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_decode import native_batches


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--start',type=float,required=True)
    p.add_argument('--duration',type=float,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[];start=time.perf_counter()
    for i,(t,e) in enumerate(native_batches(a.raw,a.start,a.duration,False,33333,reader_backend='rolling')):
        np.save(out/f'{i:04d}.npy',e)
        records.append(dict(index=i,t=t,events=len(e),file=f'{i:04d}.npy'))
        if i%30==0:print(i,round(t,4),flush=True)
    result=dict(storage='npy',spec=dict(raw=a.raw,start=a.start,duration=a.duration),frames=records,
        original_timestamps_and_polarity=True,elapsed_s=time.perf_counter()-start,
        original_events_bytes=sum(np.load(out/r['file'],mmap_mode='r').nbytes for r in records))
    (out/'manifest.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps({k:v for k,v in result.items() if k!='frames'}))


if __name__=='__main__':main()
