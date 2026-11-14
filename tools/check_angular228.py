"""Angular-order controls; these are not a calibrated camera false-alarm rate."""
from pathlib import Path
import os,sys,json,numpy as np
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from angular_phase_candidates import ring_phase_candidates
from ellipse_visibility import ellipse_fov_clipped
OUT=ROOT.parent/'output/angular228_20261108'
if __name__=='__main__':
    OUT.mkdir(exist_ok=False);dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);negative=[]
    r=dict(cx=300.,cy=250.,major=80.,minor=50.,angle=0.)
    for seed in range(8):
        rng=np.random.default_rng(seed);e=np.zeros(3000,dtype);theta=rng.uniform(0,2*np.pi,len(e))
        e['x']=np.rint(300+40*np.cos(theta));e['y']=np.rint(250+25*np.sin(theta));e['p']=rng.integers(0,2,len(e));e['t']=rng.integers(0,33333,len(e));e.sort(order='t')
        result=ring_phase_candidates(e,r,.033333);assert not result['qualified'];negative.append(dict(seed=seed,**result))
    e=np.zeros(8*360,dtype);theta=np.tile(np.arange(360)*2*np.pi/360,8)
    e['x']=np.rint(300+40*np.cos(theta));e['y']=np.rint(250+25*np.sin(theta));e['p']=np.arange(len(e))%2;e['t']=np.repeat(np.arange(8)*1e6/240,360)
    flashing=ring_phase_candidates(e,r,.033333);assert not flashing['qualified']
    assert ellipse_fov_clipped(dict(cx=100,cy=10,major=80,minor=10,angle=90))
    assert not ellipse_fov_clipped(dict(cx=10,cy=100,major=80,minor=10,angle=90))
    for x,y in [(10,100),(100,10),(300,250)]:
        a=dict(cx=x,cy=y,major=80,minor=10,angle=73)
        b=dict(a,major=10,minor=80,angle=163)
        assert ellipse_fov_clipped(a)==ellipse_fov_clipped(b)
    (OUT/'checks.json').write_text(json.dumps(dict(nonperiodic_ring_negatives=negative,global_flashing_ring=flashing,
        projected_fov_rotated_and_axis_swap_invariants=True,camera_false_alarm_calibrated=False),indent=2),'utf-8')
    print('Angular negatives and projected-FOV invariants passed')
