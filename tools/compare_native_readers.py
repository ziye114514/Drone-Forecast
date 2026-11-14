"""One SDK-reader change: current native batches/timestamps/fields must agree."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src')]
from event_drone_forecast.raw_io import open_raw,raw_chunks
from live_decode import native_batches

raw='E:/dataset/train/6/media/event.raw'
out=ROOT.parent/'output/live_decoder_comparison_20261025.json'
assert not out.exists()
def probe(it):
    rows=[];it=iter(it)
    while True:
        tick=time.perf_counter()
        try:t,e=next(it)
        except StopIteration:break
        elapsed=(time.perf_counter()-tick)*1000
        # Canonical fields exclude uninitialized struct padding.
        canonical=np.empty(len(e),dtype=[('x','<u2'),('y','<u2'),('p','<i2'),('t','<i8')])
        for n in canonical.dtype.names:canonical[n]=e[n]
        rows.append(dict(t=t,n=len(e),sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),read_ms=elapsed))
    return rows
with open_raw(raw) as reader:old=probe(raw_chunks(reader,109.9,4.1,20000))
new=probe(native_batches(raw,109.9,4.1))
assert len(old)==len(new)==205
diffs=[dict(i=i,old=a,new=b) for i,(a,b) in enumerate(zip(old,new)) if any(a[k]!=b[k] for k in ('t','n','sha256'))]
report=dict(raw=raw,window=[109.9,114.],batches=len(old),canonical_fields_exact=not diffs,differences=diffs,
            old_read_median_ms=float(np.median([r['read_ms'] for r in old])),new_read_median_ms=float(np.median([r['read_ms'] for r in new])),
            old_read_max_after_seek_ms=max(r['read_ms'] for r in old[1:]),new_read_max_after_seek_ms=max(r['read_ms'] for r in new[1:]),
            note='Sequential SDK reader comparison; cache/system load not controlled. Startup seek is separate.',old=old,new=new)
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('old','new','differences')}),flush=True)
assert not diffs
