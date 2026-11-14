"""Author pose from the preceding slice selects the NEXT raw event region.

The original rectified coordinate system is unchanged. Signed coordinates remove
the adapter's arbitrary256-pixel clipping. Author process_event accepts int;
heatmap visualization is disabled, so no image index is accessed out of bounds.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT)]
from reference_api import AuthorTracker,OUT
from sign_bank import SignBank
from live_decode import native_batches
from roi_bounds import prefilter,circle_bounds
from structure_prior import image_ellipse,missing_ellipse

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--event-budget',type=int,default=0,help='0 retains all events; otherwise fixed budget per rotor per20ms')
parser.add_argument('--fixed-directions',action='store_true',help='The signed omega bank already models both directions; disable author one-off event-count direction switch')
parser.add_argument('--fast-roi',action='store_true',help='Exact projected-circle bounding rectangle before coordinate warping')
parser.add_argument('--parallel',action='store_true',help='Four independent rotor banks in separate threads')
parser.add_argument('--structure-relock',action='store_true',help='Three current phase-supported conics guide fresh initialization of the fourth; requires new raw phase support')
parser.add_argument('--pixel-period-check',action='store_true',help='Require CURRENT independent SDK pixel-period evidence, with explicit half/double harmonic ambiguity')
args=parser.parse_args()
if args.event_budget<0:raise ValueError('event budget must be nonnegative')
parent=json.loads((OUT/'real_probe.json').read_text(encoding='utf-8'))
t0=parent['protocol']['anchor_t']
variant='fixed_direction' if args.fixed_directions else 'adaptive_signed'
name='probe' if not args.event_budget else f'budget_{args.event_budget}'
if args.fast_roi:name+='_roi'
if args.parallel:name+='_parallel'
if args.structure_relock:name+='_guided'
if args.pixel_period_check:name+='_period'
out=OUT/variant/name;out.mkdir(exist_ok=False)
dll=OUT/variant/'build/Release/helix_reference.dll'
frozen=Path(parent['protocol']['current_geometry_source'])
anchor=next(json.loads(s) for s in frozen.read_text(encoding='utf-8').splitlines() if abs(json.loads(s)['t']-t0)<1e-8)
rotors=anchor['geometry']['count']['rotors'];velocity=np.asarray(parent['protocol']['velocity_only_at_anchor'])
protocol=dict(scope='Observation-only real RAW moving-region author-core probe; no new NN/forecast run',
    raw=parent['protocol']['raw'],anchor_t=t0,duration_s=1.2,previous_exposed_s=.4,
    adaptive_region='previous-slice phase-supported author conic; freeze last supported region if none',
    fixed_coordinate_origin=[128,128],adapter_uses_signed_coordinates=True,unit_radius=80,
    initial_velocity_only=True,no_future_boxes_or_labels=True,assumed_B=2,
    thresholds_unchanged_from_causal_extension=True,event_budget_per_rotor_20ms=args.event_budget,
    subsampling='Uniform without replacement fixed RNG seed per rotor and tick, then chronological sort' if args.event_budget else None,
    GN_interval_s=.1 if args.event_budget else .04,initial_direction_bank_is_fixed=args.fixed_directions,
    exact_ROI_prefilter=args.fast_roi,independent_rotor_threads=4 if args.parallel else 1,
    three_conic_prior_reinitialization=args.structure_relock,guidance_gap_s=.1,guidance_cooldown_s=.3,
    current_pixel_period_check=args.pixel_period_check,period_seed_warmup_s=.12 if args.pixel_period_check else 0.,
    DLL_sha256=hashlib.sha256(dll.read_bytes()).hexdigest())
snap=out/'source_snapshot';snap.mkdir()
for name in ['reference_api.py','sign_bank.py','adaptive_probe.py','helix_bridge.cpp','roi_bounds.py','structure_prior.py']:
    shutil.copy2(Path(__file__).parent/name,snap/name)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
def factory(rpm,blades):return AuthorTracker(rpm,blades,dll_path=dll,canvas=(256,256,128,128,80),signed_coordinates=True,fixed_direction=args.fixed_directions)
banks={i:SignBank(parent['seeds'][str(i)]['frequency_hz'],2,factory) for i in range(4)}
regions={i:[np.array([[80.,0,128],[0,80,128],[0,0,1]])] for i in banks}
centers=np.asarray([[r['cx'],r['cy']] for r in rotors]);ordered=np.argsort(np.arctan2(*(centers-centers.mean(axis=0))[:,::-1].T))
opposite={int(ordered[k]):int(ordered[(k+2)%4]) for k in range(4)}
last_seen={i:t0 for i in banks};last_guided={i:t0 for i in banks}
last_frequency={i:parent['seeds'][str(i)]['frequency_hz'] for i in banks}
records=[];wall=time.perf_counter()
period_engine=None
if args.pixel_period_check:
    sys.path.insert(0,str(ROOT/'research/stage79'))
    import native79
    period_engine=native79.engine(1280,720,'strict7');period_output=period_engine.get_empty_output_buffer()
current_q=None
pool=ThreadPoolExecutor(max_workers=4) if args.parallel else None
def rotor_step(i,r,t,e,xy,k):
    a=np.deg2rad(r['angle']);rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    before=time.perf_counter()
    box=prefilter(xy,regions[i],r,rotation) if args.fast_roi else np.ones(len(e),bool)
    local_xy=xy[box];local_e=e[box];u=local_xy-[r['cx'],r['cy']]
    normalized=(u@rotation)/np.array([r['major']/2,r['minor']/2])
    patch=np.rint(128+80*normalized)
    hp=np.c_[patch,np.ones(len(local_e))];keep=np.zeros(len(local_e),bool)
    for H in regions[i]:
        unit=hp@np.linalg.inv(H).T;positive=unit[:,2]>1e-6
        uv=unit[:,:2]/np.maximum(unit[:,2,None],1e-6)
        rad=np.linalg.norm(uv,axis=1)
        keep|=positive&(rad>=.2)&(rad<=1.5)
    point=patch[keep];ev=local_e[keep];region_count=len(ev)
    if args.event_budget and len(ev)>args.event_budget:
        rng=np.random.default_rng(20261025+1000*i+k)
        chosen=np.sort(rng.choice(len(ev),args.event_budget,replace=False));point=point[chosen];ev=ev[chosen]
    packed=np.empty(len(ev),dtype=[('x','<i4'),('y','<i4'),('p','<i2'),('t','<i8')])
    packed['x']=point[:,0];packed['y']=point[:,1];packed['p']=ev['p'];packed['t']=ev['t']
    preprocess_ms=(time.perf_counter()-before)*1000;before=time.perf_counter()
    solve=round((t-t0)*50)%(5 if args.event_budget else 2)==0
    period=None
    if args.pixel_period_check:
        M=rotation@np.diag([r['major']/2,r['minor']/2])/80
        W=np.eye(3);W[:2,:2]=M
        W[:2,2]=np.array([r['cx'],r['cy']])+velocity*(t-t0)-M@np.array([128.,128.])
        bounds=[circle_bounds(W@H,1.) for H in regions[i]]
        if all(b is not None for b in bounds):
            lower=np.min([b[:,0] for b in bounds],axis=0)-4
            upper=np.max([b[:,1] for b in bounds],axis=0)+4
            period=native79.summarize(current_q,np.r_[lower,upper],t,1280,720)
        else:period=dict(qualified=False,frequency_hz=None,reason='unbounded_conic')
    value=banks[i].feed(t,packed,solve_pose=solve,current_period=period)
    value['core_ms']=(time.perf_counter()-before)*1000;value['preprocess_ms']=preprocess_ms
    supported=[np.asarray(v['H']) for v in value['modes'].values()
               if v['weighted_phase_concentration']>=.5 and v['informative_ring_sectors']>=9]
    if supported:regions[i]=supported
    if value['selected']:
        H=np.asarray(value['selected']['H']);center=H[:2,2]/H[2,2]
        center=np.array([r['cx'],r['cy']])+velocity*(t-t0)+rotation@((center-128)*np.array([r['major']/2,r['minor']/2])/80)
        value['conditional_axis_image_xy']=center.tolist()
    value['region_events_before_budget']=region_count;value['sampled_events']=len(ev)
    value['rectification_rotor']=dict(r)
    return value
try:
    warm=.12 if args.pixel_period_check else 0.
    for t,e in native_batches(protocol['raw'],round(t0-warm,2),round(1.2+warm,2)):
        if period_engine is not None:current_q=native79.checked_process(period_engine,e,period_output,1280,720)
        if t<=t0+1e-8:continue
        age=e['t']/1e6-t0;xy=np.c_[e['x'],e['y']]-age[:,None]*velocity;frame={};begin=time.perf_counter()
        if pool:
            jobs=[pool.submit(rotor_step,i,r,t,e,xy,len(records)) for i,r in enumerate(rotors)]
            frame={str(i):job.result() for i,job in enumerate(jobs)}
        else:
            frame={str(i):rotor_step(i,r,t,e,xy,len(records)) for i,r in enumerate(rotors)}
        for i in banks:
            if frame[str(i)]['valid']:
                last_seen[i]=t;last_frequency[i]=frame[str(i)]['selected']['phase_hz']
        guidance=None
        supported=[i for i in banks if frame[str(i)]['valid']]
        if args.structure_relock and len(supported)==3:
            missing=next(i for i in banks if i not in supported)
            if t-last_seen[missing]>=.1-1e-8 and t-last_guided[missing]>=.3-1e-8:
                conics={i:image_ellipse(frame[str(i)]['selected']['H'],rotors[i],velocity,t-t0) for i in supported}
                prior=missing_ellipse(conics,missing,opposite) if all(v is not None for v in conics.values()) else None
                if prior:
                    guidance=dict(t=t,missing=missing,ellipse=prior,classified_as_observation=False,
                                  valid_after_guidance=False,frequency_seed_from_past_only=last_frequency[missing])
                    banks[missing].close();banks[missing]=SignBank(last_frequency[missing],2,factory)
                    prior=dict(prior);prior['cx']-=velocity[0]*(t-t0);prior['cy']-=velocity[1]*(t-t0)
                    rotors[missing]=prior
                    regions[missing]=[np.array([[80.,0,128],[0,80,128],[0,0,1]])]
                    last_guided[missing]=t
        records.append(dict(t=t,rotors=frame,guidance=guidance,native_events=len(e),tracking_ms=(time.perf_counter()-begin)*1000))
finally:
    if pool:pool.shutdown(wait=True,cancel_futures=True)
    for bank in banks.values():bank.close()
assert len(records)==60,len(records)
summary={}
for period,lo,hi in [('exposed',0,.4),('new_extension',.4,1.2)]:
    rows=[r for r in records if t0+lo+1e-8<r['t']<=t0+hi+1e-8]
    summary[period]=dict(ticks=len(rows),all_four_supported=sum(all(v['valid'] for v in r['rotors'].values()) for r in rows),rotors={})
    for i in range(4):
        series=[r['rotors'][str(i)] for r in rows];values=[r['selected'] for r in series if r['valid']]
        summary[period]['rotors'][str(i)]=dict(valid_ticks=len(values),
            median_phase_hz=float(np.median([v['phase_hz'] for v in values])) if values else None,
            median_weighted_R=float(np.median([v['weighted_phase_concentration'] for v in values])) if values else None,
            ambiguity_ticks=sum(r['reason']=='conflicting_phase_frequency_hypotheses' for r in series))
result=dict(protocol=protocol,summary=summary,records=records,total_wall_s=time.perf_counter()-wall,
    tracking_ms_median=float(np.median([r['tracking_ms'] for r in records])),
    frequency_or_force_ground_truth=False,flight_prediction_improvement_proven=False)
(out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
