"""RAW-only head-tail-free disk/body-axis prototype with explicit limitations."""
from pathlib import Path
import argparse,hashlib,itertools,json,sys,time,collections
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'research/stage4'))
from tracker import Skeleton
from silhouette import BodyObserver,AxisForecast,body_measurement,angle_distance,PARAMS
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.motion import KinematicFilter
from event_drone_forecast.config import Config
OUT=ROOT.parent/'output/stage6_body_axis_v2_20261017';RAW='E:/dataset/train/6/media/event.raw'
CASES={'pre_absent':(0.,2.),'clear':(20.,10.),'side_exit':(108.,8.)}

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def hashes():
    files=[*Path(__file__).parent.glob('*.py'),ROOT/'research/stage4/tracker.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def freeze():
    if (OUT/'protocol.json').exists():raise RuntimeError('Already frozen')
    save(OUT/'protocol.json',dict(source_hashes=hashes(),cases=CASES,raw=RAW,parameters=PARAMS,
        chronological_segments=True,memory='Four-point model acquired in earlier segment retained; dynamic position/velocity reset across gaps. Assumes same single UAV in one RAW.',
        inputs='Only RAW. No human boxes/keypoints, CAD, RGB/depth, flight logs or invented intrinsics.',
        prototype='Independent reduced body event-support/PCA/LSD observer inspired by contour tracking; not a full original-paper reproduction.',
        four_point_initialization='Frozen stage4 four-only Skeleton, five consecutive qualified frames.',
        silhouette='Event support is not a complete optical silhouette. Near-side-view weak/orthographic assumption is explicit.',
        modes='Four-point initialization -> body position observation -> side-body axis observation without requiring rotor count -> coast/expiry.',
        sign='Undirected normal axis; arrow selected by prior and continuity, not independently observed.',
        force='Amplitude from past image motion, not RPM or true thrust. Image gravity sensitivity 0/200/500 px/s2 all reported, no GT selection.',
        methods='Original rotor CV; body CV; same-body-position CA; axis constrained accelerations; causal past-only fused.',
        dewarp='Additional ablation warps previous accumulator with past body velocity, clipped +/-15px per10ms.',
        raw_step_s=.01,publish_step_s=.02,evaluation='Existing sparse whole-UAV boxes posthoc; no interpolation. Matched endpoint and absent-label publication.',
        development_data_previously_seen=True,independent_pose_truth=False,default_upgrade=False,
        synthetic='Known ROI and independent rasterized disk/cylinder-like side mask with legs/clutter. Module test only.'))

def verify():assert json.loads((OUT/'protocol.json').read_text())['source_hashes']==hashes()

def tests():
    im=np.zeros((200,300),np.float32);cv2.ellipse(im,(150,100),(85,10),20,0,360,12,-1)
    m=body_measurement(im,[40,40,260,160]);assert m['valid'];assert np.degrees(angle_distance(m['angle_rad'],np.deg2rad(20)))<2
    obs=dict(rotors=[],detected=False);b=BodyObserver();r=b.step(0.,im,obs,dict(active=False));assert not r['active']
    c=AxisForecast();s=[150,100,10,0];r=c.step(0,s,dict(axis_valid=True,normal_axis_xy=[0.,-1.]),[150,100]);assert r['axis_applied']
    assert not c.step(.02,s,dict(axis_valid=False),None)['axis_applied']
    # Equivalent head-tail reversal leaves the unoriented plane axis unchanged.
    assert angle_distance(.1,.1+np.pi)<1e-12
    print('Raster side-axis, no-model rejection, axis-freshness and head-tail checks passed',flush=True)

def synthetic():
    verify();rows=[]
    # Rasterization does not use the inverse PCA/LSD estimator. Known ROI is
    # supplied only here; missing axis labels on real data are not fabricated.
    for angle,ratio,thick,legs,seed in itertools.product([-35.,0.,30.],[.02,.10,.25,.45],[0.,.12],[False,True],range(4)):
        rng=np.random.default_rng(seed+1700);im=np.zeros((220,320),np.float32)
        radius=90;vertical=max(2,int(radius*(ratio+thick)));a=np.deg2rad(angle)
        cv2.ellipse(im,(160,100),(radius,vertical),angle,0,360,10,-1)
        # Nonplanar structures have deliberately different orientation/support.
        if legs:
            for x in [-45,45]:
                p=np.array([x,15.]);q=np.array([x,50.]);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                p=R@p+[160,100];q=R@q+[160,100];cv2.line(im,tuple(np.int32(p)),tuple(np.int32(q)),8,5)
        yy=rng.integers(25,190,120);xx=rng.integers(45,275,120);im[yy,xx]+=rng.uniform(1,8,120)
        m=body_measurement(im,[35,20,285,200]);err=np.degrees(angle_distance(m['angle_rad'],a)) if 'angle_rad' in m else None
        rows.append(dict(angle_deg=angle,plane_ratio=ratio,body_thickness=thick,legs=legs,seed=seed,
            accepted=m['valid'],axis_error_deg=float(err) if err is not None else None,
            apparent_ratio=m.get('ratio'),reason=m['reason']))
    save(OUT/'synthetic.json',dict(cases=len(rows),rows=rows,known_roi=True,true_pose_only_in_synthetic=True,
          caveat='Ideal intensity/event-support mask, not a validated hardware event simulator.'))

def raw():
    verify();memory={n:BodyObserver() for n in ['body','dewarped']};four=Skeleton(body=False,partial=False)
    for case,(start,duration) in CASES.items():
        folder=OUT/case
        if (folder/'summary.json').exists():raise RuntimeError('Existing run')
        folder.mkdir(exist_ok=True);reader=open_raw(RAW);h,w=reader.get_size();front=PaperPerception(w,h,PaperConfig())
        density={n:np.zeros((h,w),np.float32) for n in memory};filters={n:KinematicFilter(Config(),2) for n in ['original',*memory]}
        seen={n:None for n in filters};predictors={n:AxisForecast() for n in memory};stats={n:collections.Counter() for n in filters}
        images=[];times=[];events=0;count=0;began=time.perf_counter();last=began
        with (folder/'trace.jsonl').open('w',encoding='utf-8') as f:
            for i,(t,e) in enumerate(raw_chunks(reader,start,duration+.4,10000)):
                o,total=front.detect(e,t);events+=len(e)
                for name in memory:
                    if name=='dewarped' and memory[name].last_seen is not None and t-memory[name].last_seen<.1:
                        shift=np.clip(memory[name].velocity*.01,-15,15)
                        density[name]=cv2.warpAffine(density[name],np.c_[np.eye(2),shift].astype(np.float32),(w,h),flags=cv2.INTER_LINEAR)
                    density[name]*=np.exp(-.01/.02);density[name]+=total
                if (i+1)%2:continue
                image=np.uint8(np.clip(np.log1p(density['body'])/4,0,1)*255);images.append(cv2.resize(image,(640,360)));times.append(t)
                if t>start+duration+1e-8:continue
                fm=four.step(t,o['rotors'],None,o.get('center'));bodies={n:m.step(t,density[n],o,fm) for n,m in memory.items()}
                measured={'original':o['center'] if o['detected'] else None}
                measured.update({n:b.get('center') if b['active'] and b.get('position_measured') else None for n,b in bodies.items()})
                forecasts={}
                for name,kf in filters.items():
                    z=measured[name]
                    if z is not None:seen[name]=t
                    if seen[name] is not None and t-seen[name]>.5:filters[name]=kf=KinematicFilter(Config(),2);seen[name]=None
                    state=kf.update(t,z,confidence=.75);active=state is not None and seen[name] is not None and t-seen[name]<=.3+1e-8
                    if name=='original':
                        forecasts['original_cv']=dict(active=bool(active),state=state.tolist() if state is not None else None,acceleration=[0.,0.])
                    else:
                        b=bodies[name];stats[name][b['mode']]+=1
                        stats[name]['axis_updates_without_four']+=int(b.get('axis_measured',False) and not b.get('four_current_observations',False))
                        if state is not None:
                            pred=predictors[name].step(t,state,b,z)
                            for method,a in pred['accelerations'].items():
                                forecasts[name+'_'+method]=dict(active=bool(active),state=state.tolist(),acceleration=a,
                                    axis_applied=pred['axis_applied'],force_magnitude_measured=False)
                            b['forecast_diagnostics']=dict(weights=pred['weights'],thrust_amplitude_source=pred['thrust_amplitude_source'])
                        else:
                            for method in ['cv','ca_history','axis_g0','axis_g200','axis_g500','causal_fused']:
                                forecasts[name+'_'+method]=dict(active=False,state=None,acceleration=[0.,0.])
                    stats[name]['active']+=int(active)
                rec=dict(t=t,observation=o,four_model=fm,bodies=bodies,forecasts=forecasts)
                f.write(json.dumps(rec,allow_nan=False)+'\n');count+=1
                if time.perf_counter()-last>15:
                    print(case,round(t,2),{n:dict(v) for n,v in stats.items()},flush=True);last=time.perf_counter()
        np.savez_compressed(folder/'event_frames.npz',images=np.array(images),times=times)
        save(folder/'summary.json',dict(frames=count,events_including_future_display=events,wall_s=time.perf_counter()-began,
            stats={n:dict(v) for n,v in stats.items()},model_memory={n:m.memory for n,m in memory.items()},
            same_raw_single_target_assumption=True,calibrated_pose=False,source_hashes=hashes()))
        print('complete',case,dict(frames=count,stats={n:dict(v) for n,v in stats.items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['tests','freeze','synthetic','raw']);a=p.parse_args()
    globals()[a.command]()
