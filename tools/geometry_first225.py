"""One small cold-start pilot, geometry before phase; no future or NN labels."""
from pathlib import Path
import os,sys,json,time,hashlib,numpy as np,cv2
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from angular_phase_candidates import ring_phase_candidates
from observed_ring_phase import ring_phase_prior
from curvature_visibility import masks,contours,recover
from fast_geometry_compute import fit_rotor_geometry_fast
from moving_event_proposal import activity_roi,propose
from native_geometry_kernels import load_backend
from batched_arc_fit import configure_native
from pure_visibility170 import window
from run_rotor_only import serial
OUT=ROOT.parent/'output/geometry_first225_20261108'

def counts(e):
    p=e['y'].astype(np.int64)*1280+e['x'];return [np.bincount(p[e['p']>0] if i==0 else p[e['p']<=0],minlength=720*1280).reshape(720,1280).astype(np.float32) for i in range(2)]

def hypotheses(early):
    on,off=counts(early);roi=activity_roi(on,off)
    if roi is None:return [],dict(reason='no_coherent_discovery_roi')
    x,y,X,Y=roi;a,b=on[y:Y,x:X],off[y:Y,x:X];dense,_=masks(a,b);rings=[]
    for c in contours(dense)[:6]:
        g=fit_rotor_geometry_fast(c,min(500.,1.5*max(X-x,Y-y)),allow_arc=False)
        if g is None or g['mode'] not in ('ellipse','arc'):continue
        if g['mode']!='ellipse' and not(g['visible_arc_fraction']>=.48 and g['center_sigma_px']<=3 and g['fit_condition']<=1500):continue
        rings.append(dict(cx=float(g['center'][0]+x),cy=float(g['center'][1]+y),major=g['major'],minor=g['minor'],
            ratio=g['ratio'],angle=g['angle'],mode=g['mode'],center_sigma_px=g['center_sigma_px'],
            axis_angle_sigma_deg=g['axis_angle_sigma_deg'],geometry_quality=g['geometry_quality']))
    rings,audit=recover(a,b,rings,np.array([x,y]))
    return rings,dict(roi=roi,curvature=audit,proposal_not_detection=True,geometry_uses_discovery_half_only=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=False);cv2.setNumThreads(2);configure_native(load_backend(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll'))
    files=['angular_phase_candidates.py','observed_ring_phase.py','tools/geometry_first225.py','curvature_visibility.py','moving_event_proposal.py']
    sha={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    for n in files:
        dst=OUT/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    records=[]
    for n,t,parent in [('11',11.8,'phase185_20261107'),('11',12.,'phase185_20261107'),
        ('12',11.8,'phase185_20261107'),('12',12.,'phase185_20261107'),('15',13.5,'motion181_20261107'),('18',13.5,'phase185_20261107')]:
        e=window(ROOT.parent/f'output/{parent}/cache_recording{n}',t);mid=round(t*1e6)-33333+33333//2;early=e[e['t']<mid]
        on,off=counts(early);warped,proposal=propose(e,t,on,off,active=True)
        entry=dict(recording=n,t=t,arms={})
        image=np.full((720,1280,3),245,np.uint8);image[e['y'],e['x']]=np.where(e['p'][:,None]>0,[240,80,30],[40,70,220])
        for arm in ['stationary','early_motion']:
            if arm=='early_motion' and warped is None:entry['arms'][arm]=dict(reason='no_early_motion_proposal',motion_proposal=proposal);continue
            current=early if arm=='stationary' else warped[warped['t']<mid]
            rings,audit=hypotheses(current);results=[]
            for r in rings:
                start=time.perf_counter()
                if arm=='stationary':
                    legacy=ring_phase_prior(e,r,t);candidate=ring_phase_candidates(e,r,t)
                else:legacy=None;candidate=ring_phase_candidates(e,r,t,proposal['velocity_px_s'])
                results.append(dict(ellipse=r,legacy_twofold=legacy,candidate=candidate,ms=1000*(time.perf_counter()-start)))
                if arm=='stationary':
                    cv2.ellipse(image,tuple(np.rint([r['cx'],r['cy']]).astype(int)),tuple(np.rint([r['major']/2,r['minor']/2]).astype(int)),r['angle'],0,360,(0,160,0) if candidate['qualified'] else (170,50,170),1)
            entry['arms'][arm]=dict(hypotheses=results,geometry_audit=audit,motion_proposal=proposal if arm=='early_motion' else None)
        cv2.imwrite(str(OUT/f'proposal_{n}_{t:.1f}.png'),image)
        records.append(entry)
        (OUT/'partial.json').write_text(json.dumps(records,default=serial,indent=2),'utf-8')
        print(n,t,{a:sum(x['candidate']['qualified'] for x in d.get('hypotheses',[])) for a,d in entry['arms'].items()},flush=True)
    # Threefold rotating blade fixture exposes the legacy hardcoded twofold assumption.
    from rotor_cuda_worker import DTYPE
    controls=[]
    for blades in [2,3]:
        base=np.arange(2000,dtype=np.int64)*33332//1999;e=np.zeros(2000*blades,DTYPE);e['t']=np.repeat(base,blades)
        angle=2*np.pi*250*e['t']/1e6+np.tile(np.arange(blades)*2*np.pi/blades,2000)
        e['x']=np.rint(300+40*np.cos(angle));e['y']=np.rint(250+25*np.sin(angle));e['p']=np.arange(len(e))%2
        r=dict(cx=300.,cy=250.,major=80.,minor=50.,ratio=.625,angle=0.)
        controls.append(dict(blades=blades,legacy=ring_phase_prior(e,r,.033333),candidate=ring_phase_candidates(e,r,.033333)))
    assert controls[1]['candidate']['qualified'] and not controls[1]['legacy']['qualified']
    configure_native(None)
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v for n,v in sha.items())
    (OUT/'results.json').write_text(json.dumps(dict(records=records,controls=controls,source_sha256=sha,
        NN_used=False,geometry_from_validation_events=False,RAW_FPS_measured=False,phase_frequency_not_mechanical_RPM=True),default=serial,indent=2),'utf-8')
