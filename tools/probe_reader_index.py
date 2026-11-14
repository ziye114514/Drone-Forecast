"""One reader mode per process; canonical native fields and read costs only."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_decode import native_batches
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['default','disabled'],required=True)
p.add_argument('--output',type=Path,required=True);args=p.parse_args()
if args.output.exists():raise FileExistsError(args.output)
args.output.parent.mkdir(parents=True,exist_ok=True)
iterator=iter(native_batches('E:/dataset/train/6/media/event.raw',109.9,4.1,build_index=args.mode=='default'))
rows=[];costs=[]
while True:
    start=time.perf_counter()
    try:t,e=next(iterator)
    except StopIteration:break
    elapsed=(time.perf_counter()-start)*1000;costs.append(elapsed)
    digest=hashlib.sha256()
    for name,dtype in [('x','<u2'),('y','<u2'),('p','<i2'),('t','<i8')]:digest.update(e[name].astype(dtype).tobytes())
    rows.append(dict(t=t,n=len(e),sha256=digest.hexdigest(),read_ms=elapsed))
result=dict(mode=args.mode,rows=rows,native_batches=len(rows),native_events=sum(r['n'] for r in rows),
    first_seek_ms=costs[0],following_read_median_ms=float(np.median(costs[1:])),
    following_read_p95_ms=float(np.percentile(costs[1:],95)),following_read_max_ms=max(costs[1:]),
    timing_is_one_sequential_run_not_controlled_hardware_benchmark=True)
args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
