"""Exact ON/OFF counts on fixed native events; component timing only."""
from pathlib import Path
import sys,json,time,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from fast_structure_front import count_pair,roi_bounds,NativeROI,FastStructure
from check_final_compute import compare
from native_counts_numba import NumbaCounter


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2);cases=[];jit=NumbaCounter()
    source=ROOT.parent/'output/predictive_harmonics_20261027/native_pilot'
    dtype=np.dtype([('x','u2'),('y','u2'),('t','i8'),('frequency','f4')])
    for name,folder in [('clear','raw_release_clear'),('partial','raw_release_clear'),('side','raw_release_side')]:
        native=np.load(source/name/'native_window.npz');t=float(native['t']);e=native['events']
        rows=[json.loads(s) for s in (ROOT.parent/'output/three_continuation_20261026'/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        row=min(rows,key=lambda r:abs(r['t']-t));bounds=roi_bounds(row['body']['bbox']);selector=NativeROI()
        begin=time.perf_counter();jit(e,bounds,t,selector);first_real_ms=(time.perf_counter()-begin)*1000
        timing={'numpy':[],'cpp':[],'jit':[]};counts={}
        for repeat in range(5):
            order=[('numpy',False),('cpp',True),('jit',None)] if repeat%2==0 else [('jit',None),('cpp',True),('numpy',False)]
            for label,flag in order:
                begin=time.perf_counter();counts[label]=jit(e,bounds,t,selector) if label=='jit' else count_pair(e,bounds,t,selector,cpp_histogram=flag)
                timing[label].append((time.perf_counter()-begin)*1000)
        for polarity in range(2):
            for label in ['cpp','jit']:np.testing.assert_array_equal(counts['numpy'][polarity],counts[label][polarity])
        models=[FastStructure(optimized_compute=True),FastStructure(optimized_compute=True,native_histogram=True),FastStructure(optimized_compute=True,jit_counts=True)]
        values=[m.step(e,t,row['body'],np.zeros(0,dtype),row['coefficient']['velocity'],True) for m in models]
        delta=max(compare(values[0],v) for v in values[1:])
        cases.append(dict(name=name,events=len(e),roi=bounds.tolist(),all_count_pixels_exact=True,
            all_geometry_outputs_max_difference=delta,median_count_ms={k:float(np.median(v)) for k,v in timing.items()},
            first_real_call_compile_or_load_ms=first_real_ms))
    # Empty input and repeated same-pixel counts are also kept exactly.
    events=np.zeros(100000,dtype=[('x','u2'),('y','u2'),('p','i2'),('t','i8')]);events['x']=31;events['y']=12;events['p'][::2]=1;events['t']=999999
    for e in [events,events[:0]]:
        a=count_pair(e,[0,0,64,32],1.,cpp_histogram=False);b=count_pair(e,[0,0,64,32],1.,cpp_histogram=True)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        for x,y in zip(a,jit(e,[0,0,64,32],1.)):np.testing.assert_array_equal(x,y)
    rng=np.random.default_rng(271030);random=np.zeros(12000,dtype=events.dtype)
    random['x']=rng.integers(0,1280,len(random));random['y']=rng.integers(0,720,len(random));random['p']=rng.integers(0,2,len(random))
    random['t']=np.sort(rng.integers(970000,1000000,len(random)))
    for bounds in [[0,0,1280,720],[100,90,810,502],[31,12,32,13]]:
        a=count_pair(random,bounds,1.);b=jit(random,bounds,1.)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    result=dict(cases=cases,empty_and_100000_same_pixel_events_exact=True,random_ROI_and_time_cutoff_exact=True,
        jit_provenance=jit.provenance,component_only_not_full_pipeline_FPS=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
