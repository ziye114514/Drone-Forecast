"""Moving sparse single-rotor control, no four-point model or labels."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from anonymous_ring_phase import AnonymousRingPhase
DT=np.dtype({'names':['x','y','p','t'],'formats':['<u2','<u2','<i2','<i8'],'offsets':[0,2,4,8],'itemsize':16})

def frame(i,*,random=False):
    end=(i+1)*33333;begin=i*33333;e=np.zeros(1600,DT);e['t']=np.repeat(np.arange(800)*33332//799+begin,2)
    tt=e['t']/1e6;v=np.array([300.,20.]);theta=2*np.pi*250*tt+np.tile([0,np.pi],800)
    if random:theta=np.random.default_rng(i).uniform(0,2*np.pi,len(e))
    e['x']=np.rint(300+v[0]*tt+40*np.cos(theta));e['y']=np.rint(250+v[1]*tt+25*np.sin(theta));e['p']=np.arange(len(e))%2
    t=end*1e-6;g=dict(cx=300+v[0]*(t-.01),cy=250+v[1]*(t-.01),major=80.,minor=50.,angle=0.,ratio=.625,
        mode='ellipse',center_sigma_px=1.,axis_angle_sigma_deg=1.,current_observation=True,fov_clipped=False)
    row=dict(t=t,body=dict(detected=True,center=[300+v[0]*t,250+v[1]*t],center_provisional=False),geometry=dict(rotors=[g]),
        structure=dict(model_centers=None,mode='axis_only'),motion=dict(current_observation=True,dt_s=.033333,estimate=dict(dx=v[0]*.033333,dy=v[1]*.033333,score=1.)))
    return row,e

if __name__=='__main__':
    out=ROOT.parent/'output/controls337_20261109';out.mkdir(exist_ok=False);dll=ROOT.parent/'output/snapshot335_20261109/build/Release/helix_reference.dll'
    results={}
    for random in (False,True):
        m=AnonymousRingPhase(dll);records=[]
        try:
            for i in range(12):
                row,e=frame(i,random=random);r=m.step(row,e);records.append(r)
        finally:m.close()
        frequencies=[r['proxy_twofold_hz'] for frame_result in records for r in frame_result['meta']['records'] if r.get('current_force_admitted')]
        results['nonperiodic' if random else 'moving_single_ring']=dict(qualified_frames=sum(r['meta']['qualified_current_rings']>0 for r in records),
            frequencies=frequencies,phase_support_events=[r['meta']['current_support_events'] for r in records])
        (out/('random.jsonl' if random else 'moving.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),'utf8')
    # Freeze/late-suffix check using two identical causal prefixes.
    a=AnonymousRingPhase(dll);b=AnonymousRingPhase(dll)
    try:
        for i in range(5):
            row,e=frame(i);a.step(row,e);b.step(row,e)
        row,e=frame(5);changed=e.copy();late=changed['t']>=round(row['t']*1e6)-16667
        rng=np.random.default_rng(337);xy=np.c_[changed['x'][late],changed['y'][late]];rng.shuffle(xy);changed['x'][late]=xy[:,0];changed['y'][late]=xy[:,1]
        aa=a.step(row,e);bb=b.step(row,changed);left=[r.get('snapshot') for r in aa['meta']['records']];right=[r.get('snapshot') for r in bb['meta']['records']]
        assert left==right,(left,right);results['late_suffix_does_not_update_frozen_current_state']=True
    finally:a.close();b.close()
    (out/'results.json').write_text(json.dumps(results,indent=2),'utf8')
    assert results['moving_single_ring']['qualified_frames']>=5,results
    assert max(abs(np.asarray(results['moving_single_ring']['frequencies'])/500-1))<.03,results
    assert results['nonperiodic']['qualified_frames']==0,results
    print(results)
