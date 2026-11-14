"""Compare a local AVX2 instantiation of the installed SDK with its binding."""
from pathlib import Path
import sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT.parent/'.cache/numba_cpu_064'),str(ROOT.parent/'.cache/sdk_activity_avx2')]
from llvmlite.binding import get_host_cpu_features
if not get_host_cpu_features().get('avx2'):raise RuntimeError('Host AVX2 required')
from _edf_sdk_activity import ActivityAdapter,compiled_AVX2
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_base import EventCD


def run(e,fast,partition=1):
    if fast:filter=ActivityAdapter(1280,720,10000);buffer=None
    else:filter=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=filter.get_empty_output_buffer()
    results=[];times=[]
    for chunk in np.array_split(e,partition):
        start=time.perf_counter()
        if fast:out=filter.process(chunk)
        else:filter.process_events(chunk,buffer);out=buffer.numpy()
        times.append((time.perf_counter()-start)*1000);results.append(out.copy())
    return np.concatenate(results),sum(times)


def same(a,b):
    assert len(a)==len(b),(len(a),len(b))
    assert all(np.array_equal(a[k],b[k]) for k in a.dtype.names)


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);cases=[]
    dtype=np.dtype(EventCD);control=np.zeros(100,dtype=dtype)
    control['x']=np.tile([0,1,0,1279,1278,1279,42,42,43,42],10)
    control['y']=np.tile([0,0,1,719,719,718,42,42,42,42],10)
    control['p']=np.arange(100)%2;control['t']=np.arange(100)*5000
    empty=np.empty(0,dtype=dtype)
    for e in [empty,control]:same(run(e,False)[0],run(e,True)[0])
    native=ROOT.parent/'output/predictive_harmonics_20261027/native_pilot'
    values=[(n,np.load(native/n/'native_window.npz')['events']) for n in ['clear','partial','side']]
    for name,e in values:
        reference,old=run(e,False);candidate,new=run(e,True);same(reference,candidate)
        same(reference,run(e,True,7)[0]);same(reference,run(e,False,7)[0])
        times={k:[] for k in ['SDK_binding','SDK_local_AVX2']}
        for repeat in range(5):
            for key,fast in ([('SDK_binding',False),('SDK_local_AVX2',True)] if repeat%2==0 else [('SDK_local_AVX2',True),('SDK_binding',False)]):
                result,ms=run(e,fast);same(reference,result);times[key].append(ms)
        cases.append(dict(name=name,source_events=len(e),kept=len(reference),all_fields_and_order_exact=True,
            partition_state_continuity_exact=True,median_filter_ms={k:float(np.median(v)) for k,v in times.items()}))
    result=dict(cases=cases,empty_and_corners_and_same_pixel_checks=True,host_AVX2=True,compiled_AVX2=compiled_AVX2,
        installed_SDK_algorithm_unmodified=True,component_only_not_complete_pipeline_FPS=True,
        vendor_headers_and_private_binary_not_for_public_release=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
