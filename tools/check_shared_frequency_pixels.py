"""Current pooled/per-region frequency summaries must be exactly unchanged."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage79')]
from live_decode import native_batches
from live_pixel_axis import SDKFixedPixelAxis
from live_frequency_pixels import latest_pixels
from fast_structure_front import roi_bounds,NativeROI
from research.stage146.branches146 import rotor_rates
import native79


def pair(q,box,t,rotors,velocity,new):
    if not new:
        return rotor_rates(q,rotors,t,velocity,1280),native79.summarize(q,box,t,1280,720)
    allowed=bool(np.all((q['frequency']>=60.)&(q['frequency']<=1200.)))
    unique=latest_pixels(q,1280)
    return rotor_rates(unique,rotors,t,velocity,1280,unique_pixels=True),native79.summarize(unique if allowed else q,box,t,1280,720,unique_pixels=allowed)


def run(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    prot=json.loads((source/'protocol.json').read_text('utf-8'))
    rows=list(map(json.loads,(source/'predictions.jsonl').read_text('utf-8').splitlines()))
    periodic=SDKFixedPixelAxis(1280,720);selector=NativeROI();velocity=[0.,0.]
    times={n:[] for n in ['old','shared']};packets=0;before=0;after=0
    for row,(t,e) in zip(rows,native_batches(prot['raw'],prot['actual_iterator_start'],prot['actual_duration'],False,prot['native_step_us'])):
        assert abs(t-row['t'])<1e-8
        b=row['body'];selected=selector.select(e,roi_bounds(b['bbox'],padding=.35)) if b.get('detected') else e[:0]
        q=periodic.feed(selected,t);q=q[q['t']>=round(t*1e6)-20000]
        before+=len(q);after+=len(latest_pixels(q,1280));packets+=1
        for rep in range(3):
            values={}
            for n in (['old','shared'] if (packets+rep)%2==0 else ['shared','old']):
                a=time.perf_counter();values[n]=pair(q,b.get('bbox'),t,row['geometry'].get('rotors',[]),velocity,n=='shared')
                times[n].append((time.perf_counter()-a)*1000)
            assert values['old']==values['shared'],(t,values)
        if row['coefficient'] is not None:velocity=row['coefficient']['velocity']
    # Latest invalid samples must not hide an older valid pooled sample.
    q=np.zeros(2,dtype=[('x','u2'),('y','u2'),('t','i8'),('frequency','f4')])
    q['x']=40;q['y']=40;q['t']=[99000,99500];q['frequency']=[100.,1500.]
    assert pair(q,[10,10,100,100],.1,[],[0,0],False)==pair(q,[10,10,100,100],.1,[],[0,0],True)
    result=dict(native_packets=packets,calls_per_method=len(times['old']),frequency_records_before=before,
        latest_pixel_records=after,all_region_and_pooled_summaries_exact=True,
        invalid_latest_frequency_fallback_preserved=True,no_frequency_or_harmonic_estimation_changed=True,
        timing_scope='Combined pixel deduplication, rotor summaries and body summary only',
        timing_ms={n:dict(median=float(np.median(v)),mean=float(np.mean(v)),p95=float(np.percentile(v,95))) for n,v in times.items()})
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
