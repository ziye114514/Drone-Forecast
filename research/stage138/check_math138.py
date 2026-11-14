"""Changed mathematics and sensor-edge ROI behavior, before RAW deployment."""
from pathlib import Path
import json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage138')]
from objective138 import FixedObjective,fast137
from tiles138 import TileOps
OUT=ROOT.parent/'output/stage138_shared_objective_20261024'

def main():
    cv2.setNumThreads(1);masks=[]
    for center in [(0,0),(1,100),(200,0),(319,100),(100,239),(150,120)]:
        m=np.zeros((240,320),np.uint8);cv2.ellipse(m,center,(32,8),13,0,360,255,-1);masks.append(m)
    rng=np.random.default_rng(138)
    for _ in range(12):
        m=np.zeros((240,320),np.uint8)
        for _ in range(5):cv2.circle(m,tuple(rng.integers([0,0],[320,240])),int(rng.integers(1,12)),255,-1)
        masks.append(m)
    masks += [np.zeros((240,320),np.uint8),np.full((240,320),255,np.uint8)]
    for raw in masks:
        a=TileOps();opened=a.open_tile(raw);closed=a.close_tile(opened);ox,oy=a.mask_origin;h,w=opened.shape
        for tile,op in [(opened,cv2.MORPH_OPEN),(closed,cv2.MORPH_CLOSE)]:
            expected=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
            if op==cv2.MORPH_CLOSE:expected=cv2.morphologyEx(expected,op,np.ones((3,3),np.uint8))
            restored=np.zeros_like(raw);restored[oy:oy+h,ox:ox+w]=tile;np.testing.assert_array_equal(restored,expected)
            c,_=cv2.findContours(tile,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE,offset=(ox,oy));d,_=cv2.findContours(expected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);assert len(c)==len(d)
            for u,v in zip(c,d):np.testing.assert_array_equal(u,v)
    maximum=0.;timing=[];oldtiming=[];n=0
    source=ROOT.parent/'output/stage137_fast_geometry_20261024/natural6/predictions.jsonl'
    for line in source.open(encoding='utf-8'):
        r=json.loads(line)
        for name,d in r['models'].items():
            f=d.get('measurement',{}).get('fit') or {}
            if not d.get('current') or len(f.get('parameter_vector',[]))!=8:continue
            ell=[fast137.fixed117.geom.geometry58.from_rotor(z) for z in r['geometry'][name]['rotors']]
            points=[np.c_[(fast137.fixed117.geom.geometry58.perimeter(ell[i])-f['origin_px'])/f['scale_px'],np.ones(64)] for i in f['rotor_order']]
            template=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]);shape=(f['radius_in_template_units'],f['frame_aspect']);p=np.array(f['parameter_vector']);eval=FixedObjective(points,template,f['scale_px'],shape)
            J=eval.jac(p);old=fast137.fast_fixed_jac(p,points,template,f['scale_px'],shape);np.testing.assert_allclose(J,old,atol=2e-9,rtol=2e-11);maximum=max(maximum,float(np.max(abs(J-old))))
            direction=np.arange(1,9)/10;eps=1e-6;finite=(eval.fun(p+eps*direction)-eval.fun(p-eps*direction))/(2*eps);np.testing.assert_allclose(J@direction,finite,rtol=3e-5,atol=3e-4)
            def legacy_fun(q):
                A=np.linalg.inv(np.r_[q,1.].reshape(3,3));result=[]
                for xy,e in zip(template,points):
                    x,y=xy*[1.,shape[1]];C=np.array([[1.,0,-x],[0,1.,-y],[-x,-y,x*x+y*y-shape[0]**2]])
                    cp=e@(A.T@C@A);result.extend(f['scale_px']*np.sum(cp*e,axis=1)/np.maximum(2*np.linalg.norm(cp[:,:2],axis=1),1e-12))
                return np.array(result)
            for step in range(4):
                q=p+np.random.default_rng(n*4+step).normal(0,1e-5,8)
                tick=time.perf_counter();value=legacy_fun(q);fast137.fast_fixed_jac(q,points,template,f['scale_px'],shape);oldtiming.append((time.perf_counter()-tick)*1000)
                tick=time.perf_counter();actual=eval.fun(q);eval.jac(q);timing.append((time.perf_counter()-tick)*1000);np.testing.assert_allclose(actual,value,atol=2e-12,rtol=2e-11)
            n+=1
        if n>=24:break
    assert n>=24;OUT.mkdir(parents=True,exist_ok=True);result=dict(mask_cases=len(masks),sensor_edges_empty_full_and_sparse_exact=True,global_contour_order_and_coordinates_exact=True,actual_fit_inputs=n,jacobian_max_abs_difference=maximum,independent_directional_derivative_passed=True,benchmark_each_iterate_changes=True,old_fun_and_jac_median_ms=float(np.median(oldtiming)),new_fun_and_jac_median_ms=float(np.median(timing)))
    (OUT/'math_checks138.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
