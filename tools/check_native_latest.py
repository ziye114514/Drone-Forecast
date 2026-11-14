"""Exact selected records and binary support under the optional native kernels."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from native_latest_frequency import LatestFrequency
from live_frequency_pixels import latest_pixels
from native_counts_numba import NumbaCounter
from native_render import occupancy_frame


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    from metavision_sdk_cv import FrequencyAlgorithm
    a=FrequencyAlgorithm(1280,720,7,60.,1200.,200,False)
    dtype=a.get_empty_output_buffer().numpy().dtype
    helper=LatestFrequency(1280,720,dtype);counter=NumbaCounter();rng=np.random.default_rng(271104)
    tests=[]
    for n,pixels in [(0,1000),(1000,1000),(20000,5000),(100000,20000)]:
        q=np.zeros(n,dtype);p=rng.integers(0,pixels,n);q['x']=p%1280;q['y']=p//1280;q['t']=np.arange(n);q['frequency']=rng.uniform(60,1200,n)
        original=latest_pixels(q,1280);new=helper(q)
        assert original.dtype==new.dtype
        # SDK frequency records contain padding bytes, which NumPy does not
        # initialize on structured gathers. Compare every measurement field.
        assert all(np.array_equal(original[key],new[key]) for key in dtype.names)
        assert np.all(helper.indices==-1)
        time_ms={k:[] for k in ['numpy','fused']}
        for repeat in range(5):
            for key,fn in ([('numpy',lambda:latest_pixels(q,1280)),('fused',lambda:helper(q))] if repeat%2==0 else [('fused',lambda:helper(q)),('numpy',lambda:latest_pixels(q,1280))]):
                start=time.perf_counter();fn();time_ms[key].append((time.perf_counter()-start)*1000)
        tests.append(dict(records=n,distinct_pixels=len(new),all_measurement_fields_and_order_exact=True,
            unused_padding_bytes_not_compared=True,median_component_ms={k:float(np.median(v)) for k,v in time_ms.items()}))
    for name in ['clear','partial','side']:
        e=np.load(ROOT.parent/'output/predictive_harmonics_20261027/native_pilot'/name/'native_window.npz')['events']
        for writable in [True,False]:
            e=e.copy();e.setflags(write=writable)
            assert np.array_equal(occupancy_frame(e),counter.occupancy(e))
    result=dict(latest_frequency=tests,three_native_occupancy_windows_exact=True,writable_readonly_signatures=True,
        startup=helper.provenance,component_only_not_full_FPS=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
