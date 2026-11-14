"""Frozen Stage3: single-arc reliability, joint plane normals and causal replay."""
from pathlib import Path
import argparse,datetime,hashlib,itertools,json,csv,sys,time
import cv2,numpy as np
from measurement import *
from event_drone_forecast.geometry import fit_rotor_geometry
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.paper_physics import ThrustAxisFilter,ProjectedThrustModel
OUT=ROOT.parent/'output/stage3_normal_confidence_20261016'
K=np.array([[800.,0,640],[0,800,360],[0,0,1.]])

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def csvsave(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def freeze():
    assert (OUT.parent/'stage2_module_validation_20261016/stage2_completion.json').exists()
    if (OUT/'protocol.json').exists():raise RuntimeError('Protocol exists')
    files=[Path(__file__),Path(__file__).with_name('measurement.py'),Path(__file__).with_name('test_measurement.py'),*(ROOT/'src/event_drone_forecast').glob('*.py')]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    for p in files:
        q=OUT/'source_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=hashes,
        single=dict(tilt_deg=[20,45,65,80,87],diameter_px=[12,36,120],off_axis=[False,True],
            arc_deg=[360,240,180,120,60],noise_px=[.25,.75],seeds=[30,31,32],occlusion_chord=[False,True],
            methods=['frozen_geometry','frozen_plus_audit','ams_plus_audit'],
            interpretation='Acceptance coverage and best-candidate angular error; best candidate is an oracle scoring bound, not actual disambiguation.'),
        audit=dict(max_points=160,leave_blocks=8,residual_sigma_floor_px=.5,correlation_inflation=4,
            max_normal_spread_deg=8,max_center_spread='max(2px,0.1diameter)',
            scope='Local sensitivity + leave-block-out diagnostics, NOT calibrated confidence intervals.'),
        multi=dict(tilt_deg=[20,45,65,80],diameter_px=[24,72,120],off_axis=[False,True],noise_px=[.25,.75],seeds=[50,51,52,53,54],
            conditions=['clean','one_arc120','two_arcs180','one_wrong_plane','missing_one','missing_two','all_arcs120'],
            methods=['frozen_up_prior','joint_all','joint_audit'],
            assumed_focal_scales=[1.,.8,1.2],focal_error_conditions=['clean'],
            shape='Four equal-radius circles on one plane with center offsets(+/-2R,+/-2R); layout not given to estimator.',
            accepted_rms_deg=5,minimum_consensus=3,alternative_axis_separation_deg=10,minimum_margin=.25,
            intrinsics='Known synthetic K, and explicit20% focal perturbation. Not a fabricated calibration of real RAW.'),
        temporal=dict(seeds=list(range(1000,1010)),dt_s=.02,steps=120,
            faults={'0.8-1.2s':'wrong-plane rotor','1.2-1.6s':'two rotors missing','1.6-1.8s':'all120degree arcs'},
            horizons_s=[.1,.2,.4],methods=['cv','frozen_axis','joint_all','joint_temporal_audit'],
            forecast_inputs='Current oracle image position/velocity and current force only, plus estimated past/current plane normal; no future observations.',
            scope='Controlled image-motion mechanism diagnostic with known effective gains, NOT end-to-end flight performance.'),
        no_new_algorithm_claimed=True,no_real_pose_truth_claimed=True))
    print('Stage3 protocol frozen',flush=True)

def verify():
    p=json.loads((OUT/'protocol.json').read_text())
    for rel,h in p['sources'].items():assert digest(ROOT/rel)==h,rel
    return p

def scene(tilt,azimuth,diameter,off=False,center_px=None):
    t,a=np.deg2rad([tilt,azimuth]);n=np.array([np.sin(t)*np.sin(a),-np.sin(t)*np.cos(a),np.cos(t)])
    u=unit(np.array([1.,0,0])-n*n[0]);v=np.cross(n,u);radius=.1;z=160/diameter
    center=np.array([.42*z,.18*z,z]) if off else np.array([0.,0.,z])
    if center_px is not None:center=np.r_[(np.asarray(center_px)-[640,360])*z/800,z]
    return dict(n=n,u=u,v=v,radius=radius,center=center)

def circle(world,offset=(0,0)):
    R=world['radius'];c=world['center']+R*(offset[0]*world['u']+offset[1]*world['v'])
    H=K@np.column_stack((R*world['u'],R*world['v'],c));inv=np.linalg.inv(H)
    C=inv.T@np.diag([1.,1.,-1.])@inv;A=C[:2,:2];center=-np.linalg.solve(A,C[:2,2]);scale=-(center@A@center+2*C[:2,2]@center+C[2,2])
    values,vectors=np.linalg.eigh(A/scale);axes=1/np.sqrt(values);angle=np.arctan2(vectors[1,0],vectors[0,0])%np.pi
    return H,np.r_[center,axes,angle]

def observed_points(world,rng,offset=(0,0),arc=360,noise=.5,chord=False):
    H,truth=circle(world,offset);phase=np.linspace(.3,.3+np.deg2rad(arc),140,endpoint=False)
    uv=np.column_stack((np.cos(phase),np.sin(phase),np.ones(len(phase))));p=uv@H.T;points=p[:,:2]/p[:,2,None]
    if chord and arc<360:points=np.vstack((points,np.linspace(points[-1],points[0],40)[1:-1]))
    points+=rng.normal(0,noise,points.shape);return points,truth

def old_geometry(points):
    m=fit_rotor_geometry(points,640,None,True)
    p=None if m is None or m['axis_only'] else np.r_[m['center'],m['major']/2,m['minor']/2,np.deg2rad(m['angle'])]
    return m,p

def single():
    cfg=verify()['single'];rows=[];case=0;start=time.perf_counter()
    for tilt,diameter,off,arc,noise,seed,chord in itertools.product(cfg['tilt_deg'],cfg['diameter_px'],cfg['off_axis'],cfg['arc_deg'],cfg['noise_px'],cfg['seeds'],cfg['occlusion_chord']):
        if arc==360 and chord:continue
        w=scene(tilt,40,diameter,off);points,truth=observed_points(w,np.random.default_rng(seed),arc=arc,noise=noise,chord=chord)
        m,p=old_geometry(points);a=fit_ams(points)
        audit_old=audit_ellipse(points,p,K);audit_ams=audit_ellipse(points,a,K)
        for method,ellipse,accepted,quality,audit in [
            ('frozen_geometry',p,p is not None,m['geometry_quality'] if m else 0.,None),
            ('frozen_plus_audit',p,p is not None and audit_old['accepted'],m['geometry_quality'] if m else 0.,audit_old),
            ('ams_plus_audit',a,a is not None and audit_ams['accepted'],None,audit_ams)]:
            r=dict(case=case,tilt_deg=tilt,diameter_px=diameter,off_axis=off,arc_deg=arc,noise_px=noise,seed=seed,chord=chord,
                method=method,accepted=bool(accepted),old_quality=quality)
            if audit:r.update(normal_spread_deg=audit['normal_spread_deg'],center_spread_px=audit['center_spread_px'],audit_reason=audit['reason'])
            if ellipse is not None:
                r['best_candidate_error_deg']=min(axis_error(n,w['n']) for n in candidates(ellipse,K))
                r['center_error_px']=float(np.linalg.norm(ellipse[:2]-truth[:2]));r['wrong_over10deg']=r['best_candidate_error_deg']>10
            rows.append(r)
        case+=1
        if case%200==0:print(json.dumps(dict(phase='single',case=case,wall_s=round(time.perf_counter()-start,1))),flush=True)
    csvsave(OUT/'single_cases.csv',rows);save(OUT/'single_completion.json',dict(cases=case,rows=len(rows),wall_s=time.perf_counter()-start))

def quad_observations(world,rng,condition,noise=.5):
    ellipses=[];points=[]
    for j,offset in enumerate([(-2,-2),(2,-2),(-2,2),(2,2)]):
        arc=120 if condition=='all_arcs120' or condition=='one_arc120' and j==3 else 180 if condition=='two_arcs180' and j>=2 else 360
        if condition=='missing_one' and j==3 or condition=='missing_two' and j>=2:
            ellipses.append(None);points.append(None);continue
        observed_world=world
        if condition=='one_wrong_plane' and j==3:
            observed_world=scene(70,80,80);observed_world['center']=world['center'].copy()
        p,gt=observed_points(observed_world,rng,offset,arc,noise,False);ellipses.append(fit_ams(p));points.append(p)
    return ellipses,points

def rotor_record(p,audit=None):
    return dict(cx=p[0],cy=p[1],major=2*p[2],minor=2*p[3],angle=float(np.rad2deg(p[4])),ratio=float(p[3]/p[2]),
        axis_only=False,geometry_quality=.8,activity=1000.,**({'measurement_audit':audit} if audit else {}))

def multi():
    cfg=verify()['multi'];rows=[];case=0;start=time.perf_counter()
    for tilt,diameter,off,noise,seed,condition in itertools.product(cfg['tilt_deg'],cfg['diameter_px'],cfg['off_axis'],cfg['noise_px'],cfg['seeds'],cfg['conditions']):
        w=scene(tilt,40,diameter,off);ellipses,points=quad_observations(w,np.random.default_rng(seed),condition,noise)
        scales=cfg['assumed_focal_scales'] if condition in cfg['focal_error_conditions'] else [1.]
        for scale in scales:
            assumed=K.copy();assumed[:2,:2]*=scale
            diagnostics=[audit_ellipse(q,p,assumed) if p is not None else dict(accepted=False) for p,q in zip(ellipses,points)]
            prior=ThrustAxisFilter(PaperConfig(camera_matrix=assumed.tolist()));normal=prior.step(0,[rotor_record(p) for p in ellipses if p is not None],0.,True)
            methods={'frozen_up_prior':dict(accepted=normal is not None,normal=None if normal is None else normal.tolist()),
                'joint_all':joint_normal(ellipses,assumed),
                'joint_audit':joint_normal(ellipses,assumed,diagnostics,True)}
            for method,result in methods.items():
                r=dict(case=case,tilt_deg=tilt,diameter_px=diameter,off_axis=off,noise_px=noise,seed=seed,condition=condition,focal_scale=scale,
                    method=method,accepted=result['accepted'],reason=result.get('reason'),inliers=result.get('inliers'),margin=result.get('margin'))
                if result['normal'] is not None:r.update(error_deg=axis_error(result['normal'],w['n']),wrong_over10deg=axis_error(result['normal'],w['n'])>10)
                rows.append(r)
            case+=1
        if case%200==0:print(json.dumps(dict(phase='multi',case=case,wall_s=round(time.perf_counter()-start,1))),flush=True)
    csvsave(OUT/'multi_cases.csv',rows);save(OUT/'multi_completion.json',dict(cases=case,rows=len(rows),wall_s=time.perf_counter()-start))

def temporal_trial(seed,stop=120):
    rng=np.random.default_rng(seed);dt=.02;steps=120;times=np.arange(steps)*dt
    normals=[];force=[];position=[np.array([820.,400.])];velocity=[np.array([0.,-8.])]
    for t in times:
        w=scene(45+12*np.sin(3*t),20+15*np.sin(2*t),80,center_px=position[-1]);normals.append(w['n'])
        s=1+.08*np.sin(4*t);force.append(s);acc=200*s*w['n'][:2]+[0,140]
        position.append(position[-1]+velocity[-1]*dt+.5*acc*dt**2);velocity.append(velocity[-1]+acc*dt)
    old=ThrustAxisFilter(PaperConfig(camera_matrix=K.tolist()));new=ConsensusAxis(PaperConfig(camera_matrix=K.tolist()))
    model=ProjectedThrustModel(PaperConfig(),False);model.parameters=np.array([200.,140.,0.,0.]);records=[]
    for k,t in enumerate(times[:stop]):
        w=scene(45+12*np.sin(3*t),20+15*np.sin(2*t),80,center_px=position[k])
        condition='one_wrong_plane' if .8<=t<1.2 else 'missing_two' if 1.2<=t<1.6 else 'all_arcs120' if 1.6<=t<1.8 else 'clean'
        ellipses,points=quad_observations(w,rng,condition,.5)
        diagnostics=[audit_ellipse(q,p,K) if p is not None else dict(accepted=False) for p,q in zip(ellipses,points)]
        raw=[rotor_record(p) for p in ellipses if p is not None]
        nold=old.step(t,raw,0,True);nnew=new.step(t,[rotor_record(p,d) for p,d in zip(ellipses,diagnostics) if p is not None],0,True)
        plain=joint_normal(ellipses,K);nplain=None if not plain['accepted'] else np.array(plain['normal'])
        estimated=[None,nold,nplain,nnew];omegas=[np.zeros(3),old.omega,np.zeros(3),new.omega];pred=[]
        for normal,omega in zip(estimated,omegas):
            if normal is None:pred.append([position[k]+velocity[k]*h for h in (.1,.2,.4)])
            else:
                rt,rp=model.rollout(position[k],velocity[k],normal,omega,force[k]);pred.append([np.array([np.interp(h,rt,rp[:,j]) for j in range(2)]) for h in (.1,.2,.4)])
        records.append(dict(t=t,condition=condition,prediction=pred,
            normal_error_deg=[None,axis_error(nold,w['n']) if nold is not None else None,axis_error(nplain,w['n']) if nplain is not None else None,axis_error(nnew,w['n']) if nnew is not None else None],
            new_measurement_accepted=new.diagnostics['accepted'],new_reason=new.diagnostics['reason']))
    return dict(times=times[:stop],prediction=np.array([r['prediction'] for r in records]),normal_error_deg=np.array([[np.nan if x is None else x for x in r['normal_error_deg']] for r in records]),
        new_measurement_accepted=np.array([r['new_measurement_accepted'] for r in records]),true_position=np.array(position),true_velocity=np.array(velocity)),records

def temporal():
    cfg=verify()['temporal'];folder=OUT/'temporal';folder.mkdir(exist_ok=True);start=time.perf_counter()
    for seed in cfg['seeds']:
        arrays,records=temporal_trial(seed);np.savez_compressed(folder/f'seed_{seed}.npz',**arrays)
        with (folder/f'seed_{seed}.jsonl').open('w',encoding='utf-8') as f:
            for r in records:
                r['prediction']=np.asarray(r['prediction']).tolist();f.write(json.dumps(r,allow_nan=False)+'\n')
        print(json.dumps(dict(phase='temporal',seed=seed,wall_s=round(time.perf_counter()-start,1))),flush=True)
    prefix,_=temporal_trial(cfg['seeds'][0],60);full=np.load(folder/f'seed_{cfg["seeds"][0]}.npz')
    np.testing.assert_array_equal(prefix['prediction'],full['prediction'][:60])
    save(folder/'completion.json',dict(seeds=len(cfg['seeds']),frames_per_seed=120,causal_prefix_frames=60,exact_prefix=True,wall_s=time.perf_counter()-start))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','single','multi','temporal']);a=parser.parse_args();globals()[a.action]()
