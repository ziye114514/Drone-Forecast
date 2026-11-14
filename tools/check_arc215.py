"""One native-operator compatibility and timing test on65 real native arcs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,inspect,json,time,numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import batched_arc_fit as module
from native_arc_search import NativeArcSearch
from research.stage147.curvature147 import notches
from itertools import combinations
import curvature_visibility as contours_module
OUT=ROOT.parent/'output/arc215_20261108'

if __name__=='__main__':
    cv2.setNumThreads(2);engine=NativeArcSearch(OUT/'geometry_kernels.dll')
    def candidate(points,extent):
        module.configure_native(engine)
        try:return module.fit_arc_exact(points,extent)
        finally:module.configure_native(None)
    cases=[]
    base=ROOT.parent/'output/geometry170_20261106';reviews=json.loads((base/'final_run/reviewed.json').read_text('utf-8'))
    for record in reviews:
        raw=base/(record['label']+'.npz')
        if not raw.exists() or not record['arms']['legacy']['body'].get('detected'):continue
        with np.load(raw) as z:on=z['on'];off=z['off']
        x,y,X,Y=record['arms']['legacy']['geometry']['roi']
        for mask in contours_module.masks(on[y:Y,x:X],off[y:Y,x:X]):
            for c in contours_module.contours(mask)[:3]:
                p=c.reshape(-1,2);ids=sorted(a['index'] for a in notches(c));extent=max(np.ptp(p,axis=0))
                pairs=sorted(combinations(ids,2),key=lambda ij:-min(ij[1]-ij[0],len(p)-(ij[1]-ij[0])))[:12]
                for i,j in pairs:
                    for a,b in [(i,j),(j,i+len(p))]:
                        q=p[np.arange(a+3,b-2)%len(p)]
                        if len(q)>=22:cases.append((q,extent))
    different=[];maximum=0.;accepted=0
    for i,(points,extent) in enumerate(cases):
        a=module.fit_arc_exact(points,extent);b=candidate(points,extent)
        if (a is None)!=(b is None):different.append(i);continue
        if a is None:continue
        accepted+=1
        for key in ('parameters','rms','coverage','sigma','angle_sigma','condition','support_fraction'):
            maximum=max(maximum,float(np.max(abs(np.asarray(a[key])-np.asarray(b[key])))))
    timing={}
    for name,fn in [('reference_python5',module.fit_arc_exact),('native_SDK4_batch',candidate)]:
        start=time.perf_counter()
        for points,extent in cases:fn(points,extent)
        timing[name]=1000*(time.perf_counter()-start)
    result=dict(real_arcs=len(cases),matching_accepts=accepted,acceptance_differences=different,max_numeric_difference=maximum,timing_ms=timing,
        new_hypothesis_order_or_thresholds=False,Python_OpenCV=cv2.__version__,native_OpenCV='SDK4',full_RAW_FPS_measured=False)
    (OUT/'operator_check.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result),flush=True)
