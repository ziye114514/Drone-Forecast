"""Four-point projective model study. Frozen development protocol, no GT input."""
from pathlib import Path
import argparse,collections,hashlib,itertools,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'research/stage4'))
from projective import *
from tracker import Skeleton
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.config import Config
from event_drone_forecast.motion import KinematicFilter

OUT=ROOT.parent/'output/stage5_four_point_projective_20261016'
RAW='E:/dataset/train/6/media/event.raw'
CASES={'clear':(20.,10.),'side':(108.,6.)}
ARMS=['original_cv','legacy_four','homography_four','joint_four','joint_density_four']

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def hashes():
    files=[*Path(__file__).parent.glob('*.py'),ROOT/'research/stage4/tracker.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def freeze():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
    save(OUT/'protocol.json',dict(source_hashes=hashes(),raw=RAW,cases=CASES,arms=ARMS,parameters=PARAMS,
        input='One RAW only, no human ROI, keypoints, flight logs or invented intrinsics.',
        four_points='Initialization and measurement updates require four currently observed ellipses. Missing points only coast <=.30s, never count as measured.',
        model='Normalized coplanar rectangle with common circle radius; joint H+normalized aspect/radius initialization, freeze shape after 5 qualified views.',
        geometry='Four image points alone always fit a homography; joint four-conic residual provides extra shape consistency, not proof of identity.',
        density='20ms leaky integration + ROI mean/std threshold + morphology + kmeans4 + hull ellipse proposal, inspired by EVPeriscope. Not original ROS code.',
        roi='Current/past original rotor proposals or past accepted model only. Forced k=4 never by itself validates four motors.',
        inference_step_s=.02,frontend_step_s=.01,forecast='Same CV parameters and .30s age limit in all arms; no new thrust integration.',
        evaluation='Same previously seen development clips; existing sparse human whole-target boxes posthoc, no interpolation. Matched endpoints reported.',
        synthetic='Known K, normalized geometry, full circles and known identities; independent analytic conic projection. IPPE both candidates kept.',
        no_independent_real_pose_truth=True,no_new_algorithm_or_sota_claim=True))

def verify():
    assert json.loads((OUT/'protocol.json').read_text())['source_hashes']==hashes()

def make_world(tilt,depth,radius,aspect):
    a=np.deg2rad(tilt);b=np.deg2rad(24.)
    Rz=np.array([[np.cos(b),-np.sin(b),0],[np.sin(b),np.cos(b),0],[0,0,1.]])
    Rx=np.array([[1.,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    R=Rz@Rx;K=np.array([[800.,0,640.],[0,800.,360.],[0,0,1.]])
    translation=np.array([depth*.20,depth*.08,depth]);H=K@np.c_[R[:,:2],translation]
    q=Q*[aspect,1.];inv=np.linalg.inv(H)
    rs=[conic_ellipse(inv.T@circle_conic(p,radius)@inv) for p in q]
    return H,K,R,q,rs

def ippe(q,image,K,normal):
    obj=np.ascontiguousarray(np.c_[q,np.zeros(4)],dtype=np.float64)
    result=cv2.solvePnPGeneric(obj,np.ascontiguousarray(image,dtype=np.float64),K,None,flags=cv2.SOLVEPNP_IPPE)
    errors=[]
    for rv in result[1]:
        n=cv2.Rodrigues(rv)[0][:,2]
        errors.append(float(np.degrees(np.arccos(np.clip(abs(n@normal),0,1)))))
    return dict(candidates=len(errors),lowest_reprojection_axis_error_deg=errors[0] if errors else None,
                oracle_best_axis_error_deg=min(errors) if errors else None)

def synthetic():
    verify();rows=[];start=time.perf_counter()
    grid=itertools.product([20,60,80,87],[5.,10.],[.35,.7],[.7,1.4],[0.,.6],range(3))
    for i,(tilt,depth,radius,aspect,noise,seed) in enumerate(grid):
        H,K,R,q,rs=make_world(tilt,depth,radius,aspect);truth=project(H,q);center=project(H,np.array([[0.,0.]]))[0]
        rng=np.random.default_rng(seed+500)
        measured=[]
        for r in rs:
            if noise:
                pts=ellipse_points(r,100)+rng.normal(0,noise,(100,2));(x,y),(d1,d2),ang=cv2.fitEllipseAMS(pts.astype(np.float32).reshape(-1,1,2))
                if d1<d2:d1,d2,ang=d2,d1,ang+90
                measured.append(dict(cx=float(x),cy=float(y),major=float(d1),minor=float(d2),angle=float(ang),mode='ellipse'))
            else:measured.append(r)
        p=np.array([[r['cx'],r['cy']] for r in measured]);ha=cv2.getPerspectiveTransform(q.astype(np.float32),p.astype(np.float32))
        fit=fit_joint(measured,max_nfev=100)
        row=dict(tilt=tilt,depth=depth,radius=radius,aspect=aspect,noise_px=noise,seed=seed,
            ellipse_center_bias_mean_px=float(np.mean(np.linalg.norm(p-truth,axis=1))),
            raw_mean_center_error_px=float(np.linalg.norm(p.mean(0)-center)),
            homography_center_error_px=float(np.linalg.norm(project(ha,np.array([[0.,0.]]))[0]-center)),
            joint_accepted=fit['accepted'],fit=fit)
        row['ippe_raw']=ippe(q,p,K,R[:,2]);row['ippe_true_centers']=ippe(q,truth,K,R[:,2])
        if 'motors' in fit:
            corrected=np.array(fit['motors'])
            row['joint_motor_error_px']=float(np.mean(np.linalg.norm(corrected-truth,axis=1)))
            row['joint_center_error_px']=float(np.linalg.norm(np.array(fit['center'])-center))
            row['ippe_corrected']=ippe(q,corrected,K,R[:,2])
        rows.append(row)
        if i%100==0:print('synthetic',i,round(time.perf_counter()-start,1),flush=True)
    save(OUT/'synthetic.json',dict(cases=len(rows),rows=rows,wall_s=time.perf_counter()-start,
          independent_analytic_truth=True,known_K_only_in_synthetic=True,known_motor_identities=True))

def raw(case):
    verify();folder=OUT/case
    if (folder/'summary.json').exists():raise RuntimeError('Existing run')
    folder.mkdir(exist_ok=True);start,duration=CASES[case]
    reader=open_raw(RAW);h,w=reader.get_size();front=PaperPerception(w,h,PaperConfig());density=np.zeros((h,w),np.float32)
    legacy=Skeleton(body=False,partial=False);models={n:ProjectiveFour(joint=n!='homography_four') for n in ARMS[2:]}
    detector=DensityFour();filters={n:KinematicFilter(Config(),2) for n in ARMS};seen={n:None for n in ARMS}
    stats={n:collections.Counter() for n in ARMS};images=[];times=[];events=0;frames=0;began=time.perf_counter();last=began
    with (folder/'trace.jsonl').open('w',encoding='utf-8') as f:
        for i,(t,e) in enumerate(raw_chunks(reader,start,duration+.4,10000)):
            o,total=front.detect(e,t);density*=np.exp(-.01/.02);density+=total;events+=len(e)
            if (i+1)%2:continue
            im=np.uint8(np.clip(np.log1p(density)/4,0,1)*255);images.append(cv2.resize(im,(640,360)));times.append(t)
            if t>start+duration+1e-8:continue
            legacy_o=legacy.step(t,o['rotors'],None,o.get('center'))
            previous=models['joint_density_four'];prior=None
            if previous.H is not None and previous.last_seen is not None and t-previous.last_seen<=.3:
                prior=project(previous.H,Q*[previous.shape[0],1.])
            rs,di=detector.detect(density,o['rotors'],prior)
            observations={'original_cv':dict(active=o['detected'],mode='measurement',center=o.get('center')),'legacy_four':legacy_o}
            for n,model in models.items():
                groups={'original':o['rotors']}
                if n=='joint_density_four':groups['density']=rs
                observations[n]=model.step(t,groups,o.get('center'))
            forecasts={}
            for n,m in observations.items():
                z=m.get('center') if m['active'] and m['mode']!='coast' else None
                if z is not None:seen[n]=t
                if seen[n] is not None and t-seen[n]>.5:filters[n]=KinematicFilter(Config(),2);seen[n]=None
                x=filters[n].update(t,z,confidence=.75)
                active=x is not None and seen[n] is not None and t-seen[n]<=.3+1e-8
                forecasts[n]=dict(active=bool(active),state=x.tolist() if x is not None else None,
                                  support_age_s=float(t-seen[n]) if seen[n] is not None else None)
                stats[n][m['mode']]+=1;stats[n]['active_forecasts']+=int(active)
            rec=dict(t=t,observation=o,density_rotors=rs,density_diagnostics=di,models=observations,forecasts=forecasts)
            f.write(json.dumps(rec,allow_nan=False)+'\n');frames+=1
            if time.perf_counter()-last>15:
                print(case,round(t,2),{n:dict(v) for n,v in stats.items()},flush=True);last=time.perf_counter()
    np.savez_compressed(folder/'event_frames.npz',images=np.array(images),times=times)
    summary=dict(frames=frames,events_including_future_display=events,wall_s=time.perf_counter()-began,stats={n:dict(v) for n,v in stats.items()},
                 four_current_observations_required=True,camera_matrix_supplied=False,old_frontend_unchanged=True)
    save(folder/'summary.json',summary);print(json.dumps(summary),flush=True)

def tests():
    H,K,R,q,rs=make_world(60,5,.55,1.)
    fit=fit_joint(rs,max_nfev=120)
    assert fit['accepted'];np.testing.assert_allclose(fit['motors'],project(H,q),atol=.001)
    assert np.mean(fit['center_correction_px'])>.01
    normal=ippe(q,project(H,q),K,R[:,2]);assert normal['candidates']==2 and normal['oracle_best_axis_error_deg']<.001
    assert ordered_four(rs[:3])[0] is None
    tr=ProjectiveFour()
    for i in range(5):r=tr.step(i*.02,{'test':rs})
    assert r['active'] and r['observed_count']==4
    r=tr.step(.1,{'test':rs[:3]});assert r['active'] and r['mode']=='coast' and r['observed_count']==0
    r=tr.step(.5,{'test':rs[:3]});assert not r['active']
    repeated=[dict(v) for v in rs];repeated[3]=repeated[2]
    assert ordered_four(repeated)[0] is None
    print('Independent projection, circle center correction, IPPE, strict-four, merged-point and expiry checks passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','synthetic','raw','tests']);p.add_argument('--case',choices=CASES);a=p.parse_args()
    if a.command=='freeze':freeze()
    elif a.command=='synthetic':synthetic()
    elif a.command=='tests':tests()
    else:raw(a.case)
