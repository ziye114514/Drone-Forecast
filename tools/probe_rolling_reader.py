"""Compare native event FIELDS (excluding unspecified struct padding)."""
from pathlib import Path
import hashlib,json,sys,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_decode import native_batches


def sample(raw,start,n,backend):
    rows=[];before=time.perf_counter()
    for t,e in native_batches(raw,(round(start*1e6)//33333)*33333/1e6,n*33333/1e6,False,33333,reader_backend=backend):
        now=time.perf_counter();canonical=np.empty(len(e),dtype=[('x','<u2'),('y','<u2'),('p','<i2'),('t','<i8')])
        for key in canonical.dtype.names:canonical[key]=e[key]
        rows.append(dict(t=t,events=len(e),sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),wall_ms=(now-before)*1000))
        time.sleep(.01);before=time.perf_counter()
    return rows


def main():
    out=ROOT.parent/'output/fred_transfer_20261029/rolling_reader_diagnostic';out.mkdir(exist_ok=False)
    result={}
    for name,raw,start,n in [('other','E:/dataset/train/2/aligned/event.raw',40.,180),('clear','E:/dataset/train/6/media/event.raw',26.,60)]:
        variants={}
        for backend in ('iterator','rolling'):
            rows=sample(raw,start,n,backend);variants[backend]=rows
            (out/f'{name}_{backend}.json').write_text(json.dumps(rows,indent=2),'utf-8')
        a,b=variants.values();exact=len(a)==len(b)==n and all((x['t'],x['events'],x['sha256'])==(y['t'],y['events'],y['sha256']) for x,y in zip(a,b))
        result[name]=dict(native_ordered_fields_and_cutoffs_exact=exact,
            timing={k:dict(startup_first_batch_ms=v[0]['wall_ms'],maximum_steady_ms=max(r['wall_ms'] for r in v[1:]),
                total_ms=sum(r['wall_ms'] for r in v)) for k,v in variants.items()})
        if not exact:raise AssertionError((name,'Different native event fields'))
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
