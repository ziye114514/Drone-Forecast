"""Equivalence on native reviewed contours and controlled limits, then one timing."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from batched_arc_fit import fit_arc_exact,distances_batch
from fast_geometry_compute import fit_arc_fast
from research.stage147.curvature147 import notches
from itertools import combinations
import curvature_visibility as module
OUT=ROOT.parent/'output/geometry171_20261107';OUT.mkdir(exist_ok=True)

if __name__=='__main__':
    cv2.setNumThreads(2);cases=[]
    base=ROOT.parent/'output/geometry170_20261106';reviews=json.loads((base/'final_run/reviewed.json').read_text('utf-8'))
    for record in reviews:
        label=record['label'];raw=base/(label+'.npz')
        if not raw.exists():continue
        with np.load(raw) as z:on=z['on'];off=z['off']
        body=record['arms']['legacy']['body']
        if not body.get('detected'):continue
        x,y,X,Y=record['arms']['legacy']['geometry']['roi']
        for mask in module.masks(on[y:Y,x:X],off[y:Y,x:X]):
            for c in module.contours(mask)[:3]:
                p=c.reshape(-1,2);ids=sorted(a['index'] for a in notches(c));extent=max(np.ptp(p,axis=0))
                pairs=sorted(combinations(ids,2),key=lambda ij:-min(ij[1]-ij[0],len(p)-(ij[1]-ij[0])))[:12]
                for i,j in pairs:
                    for a,b in [(i,j),(j,i+len(p))]:
                        q=p[np.arange(a+3,b-2)%len(p)]
                        if len(q)>=22:cases.append((q,extent))
    rng=np.random.default_rng(171);maxdelta=0.;accepted=0
    for points,extent in cases:
        a=fit_arc_fast(points,extent);b=fit_arc_exact(points,extent)
        assert (a is None)==(b is None)
        if a is None:continue
        accepted+=1
        for key in a:
            np.testing.assert_allclose(a[key],b[key],rtol=0,atol=1e-12)
            maxdelta=max(maxdelta,float(np.max(abs(np.asarray(a[key])-np.asarray(b[key])))))
    for n in [0,6,21]:assert fit_arc_exact(np.zeros((n,2)),100) is None
    points=rng.normal(size=(73,2));parameters=np.array([[0.,1.,8.,4.,.3],[1.,2.,6.,3.,1.2]])
    from event_drone_forecast.geometry import _distance
    np.testing.assert_allclose(distances_batch(points,parameters),np.array([_distance(points,p) for p in parameters]),atol=0,rtol=0)
    times={}
    for name,fit in [('reference',fit_arc_fast),('batched',fit_arc_exact)]:
        ticks=[]
        for _ in range(3):
            start=time.perf_counter()
            for p,e in cases:fit(p,e)
            ticks.append(1000*(time.perf_counter()-start))
        times[name]=dict(median_ms=float(np.median(ticks)),trials_ms=ticks)
    result=dict(real_native_arcs=len(cases),accepted=accepted,max_numeric_delta=maxdelta,
        same_hypothesis_seed_order_thresholds=True,timing=times,whole_RAW_FPS_measured=False)
    (OUT/'arc_operator_check.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
