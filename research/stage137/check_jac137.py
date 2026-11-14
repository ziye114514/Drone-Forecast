"""Check changed derivative on actual fit inputs, not the implementation itself."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage137')]
from fast137 import fast_fixed_jac,fixed117
OUT=ROOT.parent/'output/stage137_fast_geometry_20261024'

def main():
    old=fixed117.fixed_jac;tests=[]
    source=ROOT.parent/'output/stage135_optimized_forecast_20261024/raw/natural6_fast/predictions.jsonl'
    for line in source.open(encoding='utf-8'):
        r=json.loads(line)
        for name,d in r['models'].items():
            f=d.get('measurement',{}).get('fit') or {}
            if not d.get('current') or not d.get('fixed_structure_used') or len(f.get('parameter_vector',[]))!=8:continue
            rotors=r['geometry'][name]['rotors'];order=f['rotor_order'];ellipses=[fixed117.geom.geometry58.from_rotor(z) for z in rotors]
            points=[np.c_[(fixed117.geom.geometry58.perimeter(ellipses[i])-f['origin_px'])/f['scale_px'],np.ones(64)] for i in order]
            template=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
            tests.append((np.array(f['parameter_vector']),points,template,f['scale_px'],(f['radius_in_template_units'],f['frame_aspect'])))
        if len(tests)>=25:break
    assert len(tests)>=20;difference=0.;oldtimes=[];newtimes=[]
    for a in tests:
        before=old(*a);after=fast_fixed_jac(*a);np.testing.assert_allclose(after,before,rtol=2e-11,atol=2e-9);difference=max(difference,float(np.max(abs(after-before))))
        p=a[0];direction=np.arange(1,9)/10;eps=1e-6
        # Independent objective directional derivative; old closed form alone
        # could reproduce an earlier derivative bug.
        def residual(q):
            A=np.linalg.inv(np.r_[q,1.].reshape(3,3));radius,aspect=a[-1];values=[]
            for xy,e in zip(a[2],a[1]):
                x,y=xy*[1.,aspect];C=np.array([[1.,0,-x],[0,1.,-y],[-x,-y,x*x+y*y-radius*radius]])
                cp=e@(A.T@C@A);values.extend(a[3]*np.sum(cp*e,axis=1)/np.maximum(2*np.linalg.norm(cp[:,:2],axis=1),1e-12))
            return np.array(values)
        finite=(residual(p+eps*direction)-residual(p-eps*direction))/(2*eps)
        np.testing.assert_allclose(after@direction,finite,rtol=3e-5,atol=3e-4)
        for _ in range(6):
            tick=time.perf_counter();old(*a);oldtimes.append((time.perf_counter()-tick)*1000)
            tick=time.perf_counter();fast_fixed_jac(*a);newtimes.append((time.perf_counter()-tick)*1000)
    result=dict(actual_fits=len(tests),old_new_max_abs_difference=difference,independent_directional_derivative_passed=True,old_median_ms=float(np.median(oldtimes)),new_median_ms=float(np.median(newtimes)))
    (OUT/'jacobian_checks137.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
