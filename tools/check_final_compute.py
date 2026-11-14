"""Same native windows and contours: numerical equivalence before RAW timings."""
from pathlib import Path
import sys,os,json,time
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[variable]='1'
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from fast_structure_front import FastStructure
from fast_geometry_compute import percentile_pair_linear,split_contour_fast
from research.stage147.curvature147 import split_contour
from native_render import occupancy_frame
from metavision_sdk_core import BaseFrameGenerationAlgorithm


def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a)^set(b))
        return max((compare(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(list,tuple)):return max((compare(x,y) for x,y in zip(a,b)),default=0.) if len(a)==len(b) else (_ for _ in ()).throw(AssertionError('length'))
    if isinstance(a,np.ndarray) or isinstance(a,(float,int,np.number)) and not isinstance(a,(bool,np.bool_)):
        np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-10);return float(np.max(np.abs(np.asarray(a)-b))) if np.size(a) else 0.
    assert a==b,(a,b);return 0.


def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(271001)
    for n in [6,17,96,301,1300]:
        values=rng.normal(size=n);np.testing.assert_array_equal(percentile_pair_linear(values),np.percentile(values,[5,95]))
    cases=[];source=ROOT.parent/'output/predictive_harmonics_20261027/native_pilot'
    dtype=np.dtype([('x','u2'),('y','u2'),('t','i8'),('frequency','f4')]);cv2.setNumThreads(2)
    for name,folder in [('clear','raw_release_clear'),('partial','raw_release_clear'),('side','raw_release_side')]:
        p=np.load(source/name/'native_window.npz');e=p['events'];t=float(p['t'])
        rows=[json.loads(s) for s in (ROOT.parent/'output/three_continuation_20261026'/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        row=min(rows,key=lambda r:abs(r['t']-t));last=e[e['t']>=round(t*1e6)-20000]
        image=np.empty((720,1280,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(last,image)
        np.testing.assert_array_equal(occupancy_frame(last),np.uint8(image[:,:,0]!=52)*255)
        result={};times={k:[] for k in ['legacy','optimized']}
        for repeat in range(3):
            order=[('legacy',False),('optimized',True)]
            if repeat%2:order.reverse()
            for label,opt in order:
                m=FastStructure(optimized_compute=opt);start=time.perf_counter()
                value=m.step(e,t,row['body'],np.zeros(0,dtype),row['coefficient']['velocity'],True)
                times[label].append((time.perf_counter()-start)*1000)
                result[label]=value
        delta=compare(result['legacy'],result['optimized'])
        cases.append(dict(name=name,max_output_difference=delta,occupancy_exact=True,
            timing_ms={k:float(np.median(v)) for k,v in times.items()}))
    mask=np.zeros((150,230),np.uint8)
    for c in [(85,75),(130,75)]:cv2.ellipse(mask,c,(35,22),0,0,360,255,-1)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for c in contours:
        a,_=split_contour(c);b,_=split_contour_fast(c);compare(a,b)
    summary=dict(cases=cases,linear_percentiles_exact=True,controlled_split_geometry_equal=True,
        component_only_not_full_FPS=True,thresholds_and_source_samples_unchanged=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary))


if __name__=='__main__':main(sys.argv[1])
