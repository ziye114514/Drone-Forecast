"""Small, frozen-protocol skeleton feasibility study; experimental, not release."""
from pathlib import Path
import argparse, collections, hashlib, json, sys, time, itertools
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT.parent/'output/stage4_skeleton_pilot_20261016'
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))
from tracker import Skeleton, PARAMS, initialization, fit_affine, flatness, Q, X
from event_drone_forecast.raw_io import open_raw, raw_chunks
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.config import Config
from event_drone_forecast.motion import KinematicFilter

RAW='E:/dataset/train/6/media/event.raw'
CASES={'clear':(20.,10.),'side':(108.,6.)}
ARMS={'original_cv':None,'four_only':(False,False),
      'skeleton_partial':(False,True),'skeleton_body':(True,True)}

def save(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*Path(__file__).parent.glob('*.py'),*(ROOT/'src/event_drone_forecast').glob('*.py')]}

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'protocol.json'
    if p.exists():raise RuntimeError('Protocol already frozen')
    save(p,dict(scope='Feasibility pilot, not full reproductions of cited methods or 6-DoF pose.',
         inference_inputs=[RAW],cases=CASES,params=PARAMS,arms=list(ARMS),source_hashes=hashes(),
         baseline='Shared frozen PaperPerception 10 ms, all four CV filters sampled at 20 ms with identical noise.',
         model='Affine image projection of normalized symmetric four-motor square; no metric dimensions or invented K.',
         initialization='Five consecutive qualifying four-candidate frames; no labels or manual initialization.',
         evaluation='Existing sparse human whole-UAV boxes post-hoc only. No interpolation. Common endpoints also reported.',
         synthetic='Point-level module test with known motor identities, not sensor or detector validation.',
         horizons_s=[.1,.2,.4],coast_s=.3,
         physics='Thrust/RPM dynamics unchanged and not integrated: test measurement bottleneck first.',
         raw_camera_pose_truth=False,development_clips_previously_seen=True,
         whole_body='Event-density LK support outside observed disks; no learned semantic UAV classifier.',
         max_claim='Observed support, tracking and image forecasts only; inferred points never counted as truth.'))

def verify():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert p['source_hashes']==hashes(),'Frozen source changed: new protocol required'

def run_raw(case):
    verify(); folder=OUT/case
    if (folder/'summary.json').exists():raise RuntimeError('Existing run')
    folder.mkdir(exist_ok=True);start,duration=CASES[case]
    reader=open_raw(RAW);h,w=reader.get_size();frontend=PaperPerception(w,h,PaperConfig())
    trackers={n:Skeleton(body=b,partial=p) for n,args in ARMS.items() if args for b,p in [args]}
    filters={n:KinematicFilter(Config(),2) for n in ARMS}
    seen={n:None for n in ARMS};stats={n:collections.Counter() for n in ARMS}
    records=[];images=[];image_times=[];last=time.perf_counter();began=last;event_count=0
    accumulated=np.zeros((h,w),np.float32)
    with (folder/'trace.jsonl').open('w',encoding='utf-8') as out:
        for i,(t,e) in enumerate(raw_chunks(reader,start,duration+.4,10000)):
            event_count+=len(e)
            # Original candidate processing unchanged. No GT or future images.
            o,total=frontend.detect(e,t)
            accumulated*=np.exp(-.01/.02);accumulated+=total
            if (i+1)%2:continue
            image=np.uint8(np.clip(np.log1p(accumulated)/4,0,1)*255)
            image=cv2.GaussianBlur(image,(3,3),.6)
            images.append(cv2.resize(image,(640,360)));image_times.append(t)
            if t>start+duration+1e-7:continue # extra images are post-hoc future display only
            models={'original_cv':dict(t=t,active=o['detected'],mode='original_measurement',
                                      center=o.get('center'),since_support_s=0.)}
            for n,tr in trackers.items():
                models[n]=tr.step(t,o['rotors'],image if tr.body else None,o.get('center'))
            states={}
            for n,m in models.items():
                measured=m['active'] and m['mode']!='coast'
                z=m.get('center') if measured else None
                if measured:seen[n]=t
                if seen[n] is not None and t-seen[n]>.5:
                    filters[n]=KinematicFilter(Config(),2);seen[n]=None
                state=filters[n].update(t,z,confidence=.75)
                active=state is not None and seen[n] is not None and t-seen[n]<=.3+1e-9
                states[n]=dict(active=bool(active),state=state.tolist() if state is not None else None,
                               support_age_s=float(t-seen[n]) if seen[n] is not None else None)
                stats[n]['active']+=int(active);stats[n][m['mode']]+=1
                stats[n]['measured']+=int(measured)
            rec=dict(t=t,observation=o,models=models,forecasts=states)
            records.append(rec);out.write(json.dumps(rec,allow_nan=False)+'\n')
            if time.perf_counter()-last>15:
                print(case,round(t,2),{n:dict(s) for n,s in stats.items()},flush=True);last=time.perf_counter()
    np.savez_compressed(folder/'event_frames.npz',images=np.array(images),times=image_times)
    summary=dict(frames=len(records),events=event_count,wall_s=time.perf_counter()-began,
                 stats={n:dict(s) for n,s in stats.items()},
                 initializations={n:tr.init_count for n,tr in trackers.items()},
                 native_raw_decoded=True,fresh_frontend_cold_start=True,camera_matrix_supplied=False)
    save(folder/'summary.json',summary);print(json.dumps(summary),flush=True)

def synthetic():
    verify(); rows=[]
    # Known correspondences isolate the claimed three-point geometry advantage.
    # Perspective cases deliberately violate the affine model.
    for seed,tilt,projection,missing in itertools.product(range(20),[20,60,80,87,89],['affine','perspective'],[1,2]):
        rng=np.random.default_rng(seed+3000)
        a=np.deg2rad(tilt)
        P=np.c_[Q[:,0],Q[:,1]*np.cos(a),Q[:,1]*np.sin(a)]
        if projection=='affine':truth=P[:,:2]*65+[640,360]
        else:truth=P[:,:2]/(7+P[:,2,None])*455+[640,360]
        ids=np.arange(4-missing)
        noisy=truth[ids]+rng.normal(0,.75,(len(ids),2))
        if len(ids)>=3:estimated=X@fit_affine(ids,noisy).T
        else:
            # Previous correctly initialized shape is given in this diagnostic.
            estimated=truth+np.mean(noisy-truth[ids],axis=0)
        hidden=np.arange(4-missing,4)
        rows.append(dict(seed=seed,tilt_deg=tilt,projection=projection,missing=missing,
             observed_flatness=flatness(noisy),hidden_error_px=float(np.mean(np.linalg.norm(estimated[hidden]-truth[hidden],axis=1))),
             current_visible_points=len(ids),known_correspondences=True,previous_true_shape_given=missing==2))
    # 1 px noise relative to plane thickness; information loss at grazing view.
    groups=[]
    for tilt,proj,missing in itertools.product([20,60,80,87,89],['affine','perspective'],[1,2]):
        r=[v for v in rows if (v['tilt_deg'],v['projection'],v['missing'])==(tilt,proj,missing)]
        groups.append(dict(tilt_deg=tilt,projection=proj,missing=missing,
          mean_hidden_error_px=float(np.mean([v['hidden_error_px'] for v in r])),
          projected_thickness_px=float(130*np.cos(np.deg2rad(tilt))),
          plane_signal_to_1px_noise=float(65*np.cos(np.deg2rad(tilt)))))
    save(OUT/'synthetic.json',dict(cases=len(rows),rows=rows,aggregate=groups,
         scope='Known motor identities; two-point cases supplied previous true shape. No validation of detector, true pose or flight.'))

def tests():
    # Independent geometry fixtures and expiration/correspondence checks.
    A=np.array([[50,12,300],[8,20,200.]])
    p=X@A.T
    np.testing.assert_allclose(X@fit_affine(np.array([0,1,2]),p[:3]).T,p,atol=1e-10)
    for ids in itertools.combinations(range(4),3):
        np.testing.assert_allclose(fit_affine(np.array(ids),p[list(ids)]),A,atol=1e-10)
    assert flatness(np.c_[np.arange(4),np.zeros(4)])<1e-10
    rs=[dict(cx=x,cy=y,major=15.,minor=8.,angle=0.) for x,y in p]
    assert initialization(rs)[0] is not None
    tr=Skeleton()
    for j in range(5):r=tr.step(j*.02,rs)
    assert r['active']
    r=tr.step(.1,rs[:3]);assert r['matched_count']==3
    assert len(r['observed_slots'])==3 and len(r['points'])==4
    r=tr.step(.5,[]);assert not r['active']
    # Causal prefix invariance, including missing observations and recovery.
    def replay(n):
        tr=Skeleton();out=[]
        for j in range(n):
            moved=[dict(r,cx=r['cx']+j) for r in rs]
            out.append(tr.step(j*.02,moved[:3] if j>=8 else moved))
        return out
    assert replay(12)==replay(20)[:12]
    print('Geometry, identity, expiry and prefix checks passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','synthetic','raw','tests']);p.add_argument('--case',choices=CASES);a=p.parse_args()
    if a.command=='freeze':freeze()
    elif a.command=='synthetic':synthetic()
    elif a.command=='tests':tests()
    else:run_raw(a.case)
