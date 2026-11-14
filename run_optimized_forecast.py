"""RAW-only optimized forecast, optional guarded rotor physics and arc splits.

One RAW reader, a shared event-count pair, unchanged local CUDA detector.
Validated144 baseline remains independent of experimental146/147 branches.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,subprocess,sys,time
from multiprocessing import shared_memory
import cv2,numpy as np
from forecast_runtime import resolve as resolve_runtime
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage143'),str(ROOT/'research/stage141'),str(ROOT/'research/stage140'),str(ROOT/'research/stage138'),str(ROOT/'research/stage137'),str(ROOT/'research/stage136'),str(ROOT/'research/stage135'),str(ROOT/'research/stage129'),str(ROOT/'research/stage124')]
from predictor135 import Predictor,forecast
from visible140 import Predictor as VisiblePredictor
from publication135 import Publication
from fast137 import setup as fast_geometry_setup
from objective138 import install as install_shared_objective
from tiles138 import fronts as tile_fronts
from fast_geometry136 import baseline_for
from boundary129 import bounded
from occupancy124 import SupportFrontend,MergeRecovery
from worker122 import PartialModel,geom
from polarity import counts
from event_drone_forecast.raw_io import open_raw,raw_chunks
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_core import BaseFrameGenerationAlgorithm

def js(x):return json.dumps(x,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
def fingerprints():
    paths=[Path(__file__).resolve(),ROOT/'forecast_runtime.py',ROOT/'research/stage45/graph45.py',ROOT/'research/stage141/native_axis141.py',ROOT/'research/stage79/native79.py',ROOT/'research/stage140/visible140.py',ROOT/'research/stage139/dense139.py',ROOT/'research/stage138/objective138.py',ROOT/'research/stage138/tiles138.py',ROOT/'research/stage137/fast137.py',ROOT/'research/stage136/fast_geometry136.py',ROOT/'research/stage135/predictor135.py',ROOT/'research/stage135/publication135.py',ROOT/'research/stage132/fast132.py',ROOT/'research/stage130/motion130.py',ROOT/'research/stage23/robust23.py',ROOT/'research/stage129/boundary129.py',ROOT/'research/stage124/occupancy124.py',ROOT/'research/stage47/worker47.py']
    paths += [p for d in [8,9,55,58,61,104,107,110,113,117,119,122] for p in (ROOT/f'research/stage{d}').glob('*.py')]
    paths += list((ROOT/'src/event_drone_forecast').glob('*.py'))
    paths += list((ROOT/'research/stage146').glob('branches146.py'))
    paths += list((ROOT/'research/stage147').glob('curvature147.py'))
    paths += [ROOT/'research/stage148/temporal148.py',ROOT/'research/stage148/residual148.py',ROOT/'research/stage148/guided148.py']
    paths += list((ROOT/'research/stage153').glob('lm153.py'))
    paths += list((ROOT/'research/stage154').glob('turn154.py'))
    paths += list((ROOT/'research/stage156').glob('parallel156.py'))
    paths += [ROOT/'research/stage149'/name for name in ['inertia149.py','dynamics149.py','observer149.py']]
    paths += [ROOT/name for name in ['run_five_circle_v3.py','structure_v3.py','five_circle_model.py','lumped_disturbance.py']]
    paths += [ROOT/name for name in ['live_structure_bridge.py','live_scale_rollout.py','live_local_flow.py','live_flow_acceleration.py']]
    paths += [ROOT/'structure_activity.py']
    paths += [ROOT/'whole_body_axis.py']
    paths += [ROOT/'research/stage6/silhouette.py']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1] if len(args)>1 else None;write=isinstance(mode,str) and any(v in mode for v in 'wax+')
        if not write and p.is_relative_to(ROOT.parent/'output') and p.suffix.lower() in ['.json','.jsonl','.png','.mp4','.npz']:
            raise RuntimeError('No output cache or truth input in optimized RAW entry: '+str(p))
def receive(p,diagnostics=None):
    # Native runtime libraries can write banners/warnings to stdout despite
    # Python's redirect_stdout. Keep these separate from the JSON protocol.
    for _ in range(64):
        line=p.stdout.readline()
        if not line:raise RuntimeError('GPU worker stopped; inspect gpu_worker.log')
        try:
            message=json.loads(line)
            if isinstance(message,dict):return message
        except json.JSONDecodeError:pass
        if diagnostics is not None:
            diagnostics.write('NON_JSON_GPU_STDOUT '+repr(line)+'\n');diagnostics.flush()
    raise RuntimeError('GPU stdout contains no protocol message; inspect gpu_worker.log')
DAMPING_ALPHA=0.9
def run(raw,output,start,duration,preview,verify_frames,geometry_backend='shared',motion_backend='visible',periodic_axis=False,robust_velocity=False,damped_forecast=True,rotor_branches=False,rotor_threshold=4.,rotor_adaptive=False,rotor_curvature=False,rotor_temporal=False,rotor_inertia=False,preview_window_ms=20.,pose_solver='trf',curved_rollout=False,skip_unobserved_geometry=False,parallel_geometry=False,gpu_python=None,detector_weights=None,detector_sha256=None,cache_dir=None,event_observer=None,branch_transform=None):
    rotor_temporal = rotor_temporal or rotor_inertia
    if not 0 < preview_window_ms <= 20: raise ValueError('Preview window must be in (0,20] ms')
    if pose_solver not in ('trf','lm') or (pose_solver=='lm' and geometry_backend!='shared'):raise ValueError('LM requires shared geometry')
    if parallel_geometry and geometry_backend!='shared':raise ValueError('Parallel geometry requires shared geometry')
    rotor_curvature = rotor_curvature or rotor_temporal
    rotor_branches = rotor_branches or rotor_curvature
    raw=Path(raw).resolve();out=Path(output).resolve()
    for p in [Path('D:/yolov10-main'),Path('E:/dataset/software/yolov10-main')]:
        if raw.is_relative_to(p) or out.is_relative_to(p):raise ValueError('Protected project')
    if out.exists():raise FileExistsError(out)
    if start<0 or duration<=0 or round(start*1e6)%20000 or round(duration*1e6)%20000:raise ValueError('Positive duration/start must align to20ms')
    if not raw.is_file() or raw.suffix.lower()!='.raw':raise ValueError('Expected RAW')
    runtime=resolve_runtime(gpu_python,detector_weights,detector_sha256,cache_dir)
    out.mkdir(parents=True,exist_ok=False);sys.addaudithook(guard);hashes=fingerprints();began=time.perf_counter();reader=open_raw(raw);h,w=reader.get_size();assert (h,w)==(720,1280)
    if geometry_backend=='shared':install_shared_objective();fronts=tile_fronts(w,h)
    else:fronts=fast_geometry_setup(w,h) if geometry_backend=='fast' else dict(count=MergeRecovery(w,h,'merge_safe'),occupancy=SupportFrontend(w,h,'occupancy_roi'))
    geometry_pool=None
    if parallel_geometry:
        sys.path.insert(0,str(ROOT/'research/stage156'))
        from parallel156 import fronts as isolated_fronts,Pool as GeometryPool
        fronts=isolated_fronts(w,h)
    if pose_solver=='lm':
        sys.path.insert(0,str(ROOT/'research/stage153'))
        from lm153 import install as install_lm
        install_lm()
    curved_forecast=None
    if curved_rollout:
        sys.path.insert(0,str(ROOT/'research/stage154'))
        from turn154 import forecast as curved_forecast
    models={name:PartialModel('affine3') for name in fronts};predictor=VisiblePredictor() if motion_backend=='visible' else Predictor()
    branch_observer=None;branch_counts={};branch_cost=[]
    if rotor_branches:
        sys.path.insert(0,str(ROOT/'research/stage146'))
        from branches146 import BranchObserver,render_panel,PARAMS as BRANCH_PARAMS
        if rotor_curvature:
            sys.path.insert(0,str(ROOT/'research/stage147'))
            from curvature147 import BranchObserver,PARAMS as CURVATURE_PARAMS
            BRANCH_PARAMS=dict(BRANCH_PARAMS,curvature147=CURVATURE_PARAMS)
        if rotor_temporal:
            sys.path.insert(0,str(ROOT/'research/stage148'))
            from temporal148 import BranchObserver,PARAMS as TEMPORAL_PARAMS
            from residual148 import PARAMS as RESIDUAL_PARAMS
            from guided148 import PARAMS as GUIDED_PARAMS
            BRANCH_PARAMS=dict(BRANCH_PARAMS,temporal148=TEMPORAL_PARAMS,residual148=RESIDUAL_PARAMS,guided148=GUIDED_PARAMS)
        if rotor_inertia:
            sys.path.insert(0,str(ROOT/'research/stage149'))
            from observer149 import BranchObserver,PARAMS as INERTIA_PARAMS
            BRANCH_PARAMS=dict(BRANCH_PARAMS,inertia149=INERTIA_PARAMS)
        branch_observer=BranchObserver(w,h,rotor_threshold,rotor_adaptive)
        if preview:(out/'rotor_branch_frames').mkdir()
    periodic_observer=None
    if periodic_axis:
        from native_axis141 import NativeAxis
        periodic_observer=NativeAxis(w,h)
    velocity_guard=None;velocity_guarded=0
    if robust_velocity:
        from variants143 import RobustVelocityModel
        velocity_guard=RobustVelocityModel(window=.30,disagreement=150.)
    noise=ActivityNoiseFilterAlgorithm(w,h,10000);buffer=noise.get_empty_output_buffer();memory=None;image=None;gpu=None;video=None;log=None;n=0;publisher=Publication();cost=[];geometry_cost=[];nn_cost=[];render_cost=[];frontend_cost=[];model_cost=[];geometry_skipped_ticks=0
    try:
        if parallel_geometry:geometry_pool=GeometryPool()
        memory=shared_memory.SharedMemory(create=True,size=h*w*3);image=np.ndarray((h,w,3),np.uint8,buffer=memory.buf);log=(out/'gpu_worker.log').open('x',encoding='utf-8')
        gpu=subprocess.Popen([runtime['gpu_python'],str(ROOT/'research/stage47/worker47.py'),'--shared-memory',memory.name,'--precision','32','--weights',runtime['detector_weights'],'--expected-sha256',runtime['detector_sha256'],'--cache-dir',runtime['cache_dir']],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        ready=receive(gpu,log);assert ready['ready']
        (out/'protocol.json').write_text(js(dict(stage=156 if parallel_geometry else (154 if curved_rollout else (153 if pose_solver=='lm' else (149 if rotor_inertia else (148 if rotor_temporal else (147 if rotor_curvature else (146 if rotor_branches else 144)))))),pose_solver=pose_solver,curved_rollout=curved_rollout,skip_unobserved_geometry=skip_unobserved_geometry,parallel_geometry=parallel_geometry,component_times_are_arm_sum=parallel_geometry,curved_policy='unmodeled_fallback_only',rotor_inertia=rotor_inertia,preview_window_ms=preview_window_ms,detector_window_ms=20.,rotor_temporal=rotor_temporal,rotor_curvature=rotor_curvature,rotor_branches=rotor_branches,rotor_threshold=rotor_threshold,rotor_adaptive=rotor_adaptive,rotor_branch_parameters=BRANCH_PARAMS if rotor_branches else None,predictor_stage=140 if motion_backend=='visible' else 135,periodic_axis_observation_only=periodic_axis,robust_velocity_guard=robust_velocity,damped_forecast=damped_forecast,damping_alpha=DAMPING_ALPHA,motion_backend=motion_backend,geometry_backend=geometry_backend,counts_overlap_GPU=geometry_backend=='shared',entry='run_optimized_forecast.py',raw=str(raw),start=start,duration=duration,raw_size=raw.stat().st_size,raw_mtime_ns=raw.stat().st_mtime_ns,
            input='RAW and fixed pre-existing detector weights only; no old output/cache/label input.',sources=hashes,runtime=runtime,gpu_worker=ready,geometry='Current20ms native normalized counts; count and occupancy share one prepared ON/OFF pair; existing FOV guard and independent affine3 models.',
            forecasts=['optimized_geometry','matched_body','body_cv']+(['curved_rollout'] if curved_rollout else []),research_controls_removed=True,default_promoted=False,
            optional_observation_extension=event_observer.protocol() if event_observer is not None else None,
            structural_extension=branch_transform.protocol() if branch_transform is not None else None)),'utf-8')
        if preview:video=cv2.VideoWriter(str(out/'predictions.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50,(w+400,h));assert video.isOpened()
        begin=max(0.,start-.1)
        with (out/'predictions.jsonl').open('x',encoding='utf-8') as stream:
            for t,e in raw_chunks(reader,begin,duration+start-begin,20000):
                tick=time.perf_counter();noise.process_events(e,buffer);clean=buffer.numpy().copy()
                geometry_ready=t-begin>=.1-1e-8
                if t<=start+1e-8:
                    if branch_observer is not None:branch_observer.warmup(e)
                    if periodic_observer is not None:periodic_observer.feed(e,t)
                    if geometry_ready:
                        prepared=counts(e,w,h);prepared['on']/=2;prepared['off']/=2
                        fronts['count'].detect_prepared(e,t,prepared);fronts['occupancy'].set_hint(e,None,t);fronts['occupancy'].detect_prepared(e,t,prepared)
                    continue
                if geometry_backend!='shared':prepared=counts(e,w,h);prepared['on']/=2;prepared['off']/=2
                BaseFrameGenerationAlgorithm.generate_frame(clean,image);digest=hashlib.sha256(image.tobytes()).hexdigest() if verify_frames else None
                gpu.stdin.write(json.dumps(dict(id=n,verify_frame=verify_frames))+'\n');gpu.stdin.flush()
                if geometry_backend=='shared':prepared=counts(e,w,h);prepared['on']/=2;prepared['off']/=2
                periodic_points=np.empty((0,2));periodic_measurement=None
                if periodic_observer is not None:periodic_q=periodic_observer.feed(e,t)
                nn=receive(gpu,log);assert nn['id']==n and (not verify_frames or nn['frame_sha256']==digest)
                if nn['boxes']:
                    b=max(nn['boxes'],key=lambda b:b['confidence']);box=b['bbox'];body=dict(t=t,detected=True,center=[(box[0]+box[2])/2,(box[1]+box[3])/2],bbox=box,rotors=[],confidence=b['confidence'],source='body')
                else:body=dict(t=t,detected=False,center=None,rotors=[],source='body')
                gtick=time.perf_counter();fronts['occupancy'].set_hint(e,body,t);observations={};current={};boundary={};diagnostics={};fcost=0.;mcost=0.
                use_geometry=geometry_ready and (body['detected'] or not skip_unobserved_geometry)
                if geometry_pool is not None:
                    joined=geometry_pool.step(fronts,models,e,t,prepared,body,w,h,use_geometry)
                    for name,value in joined.items():
                        observations[name]=value['observation'];boundary[name]=value['boundary'];current[name]=value['rotors'];diagnostics[name]=value['diagnostic']
                        fcost+=value['frontend_ms'];mcost+=value['model_ms']
                else:
                    for name,front in fronts.items():
                        ftick=time.perf_counter()
                        obs=front.detect_prepared(e,t,prepared)[0] if use_geometry else dict(t=t,detected=False,rotors=[],reason='geometry_not_scheduled_without_current_body' if geometry_ready else 'warmup')
                        fcost+=(time.perf_counter()-ftick)*1000
                        mtick=time.perf_counter();rotors,audit=bounded(obs.get('rotors',[]),w,h);baseline=baseline_for(models[name],t,rotors,body);_,d=models[name].step(t,body,rotors,dict(valid=False),baseline);mcost+=(time.perf_counter()-mtick)*1000
                        observations[name]=obs;boundary[name]=audit;current[name]=rotors;diagnostics[name]=d
                if skip_unobserved_geometry and not body['detected']:geometry_skipped_ticks+=1
                gcost=(time.perf_counter()-gtick)*1000;ptick=time.perf_counter();result=predictor.step(t,body,diagnostics);pcost=(time.perf_counter()-ptick)*1000
                valid=publisher.step(t,body['detected'])
                coefficients=dict(result['coefficients']);cv=coefficients['matched_body']
                if cv is not None:cv=dict(cv,acceleration=[0.,0.])
                coefficients['body_cv']=cv
                if velocity_guard is not None:
                    guarded=velocity_guard.step(dict(t=t,body=body,coefficients=coefficients,predictor=result),{})
                    if guarded.get('active') and guarded.get('coefficient') is not None:
                        coefficients['optimized_geometry']=guarded['coefficient'];velocity_guarded+=1
                predictions={name:{str(dt):forecast(c,dt) for dt in [.1,.2,.4]} for name,c in coefficients.items() if c is not None}
                if damped_forecast and 'optimized_geometry' in predictions:
                    origin_=np.asarray(coefficients['optimized_geometry']['position'],float)
                    for dt in (.1,.2,.4):
                        shrink=max(0.5,1.0-DAMPING_ALPHA*dt*dt)
                        point=np.asarray(predictions['optimized_geometry'][str(dt)],float)
                        predictions['optimized_geometry'][str(dt)]=(origin_+shrink*(point-origin_)).tolist()
                if curved_forecast is not None and coefficients.get('optimized_geometry') is not None:
                    predictions['curved_rollout']=(dict(predictions['optimized_geometry']) if result['structure_supported'] else
                        {str(h):curved_forecast(coefficients['optimized_geometry'],h) for h in (.1,.2,.4)})
                if periodic_observer is not None:periodic_measurement,periodic_points=periodic_observer.measure(periodic_q,body,t,(cv or {}).get('velocity',[0.,0.]))
                branch_measurement=None
                if branch_observer is not None:
                    btick=time.perf_counter()
                    branch_measurement=branch_observer.step(e,t,prepared,body,coefficients.get('optimized_geometry'),predictions.get('optimized_geometry',{}))
                    if branch_transform is not None:
                        branch_measurement=branch_transform.step(t,body,coefficients.get('optimized_geometry'),predictions.get('optimized_geometry',{}),branch_measurement,image=image,prepared=prepared)
                    branch_cost.append((time.perf_counter()-btick)*1000)
                    state=branch_measurement['geometry']['state'];branch_counts[state]=branch_counts.get(state,0)+1
                    expert=branch_measurement['expert']
                    if expert['physical'] is not None:predictions['rotor_physics']=expert['physical']
                    if expert['fused']:predictions['rotor_fused']=expert['fused']
                    predictions.update(branch_measurement.get('extra_predictions',{}))
                auxiliary_observation=None
                if event_observer is not None:
                    readonly_events=e.view();readonly_events.flags.writeable=False
                    auxiliary_observation=event_observer.step(readonly_events,t,copy.deepcopy(body),
                        copy.deepcopy(current),copy.deepcopy(diagnostics),copy.deepcopy(coefficients))
                elapsed=(time.perf_counter()-tick)*1000;cost.append(elapsed);geometry_cost.append(gcost);nn_cost.append(nn['inference_ms']);frontend_cost.append(fcost);model_cost.append(mcost);render_tick=time.perf_counter()
                record=dict(t=t,body=body,models=diagnostics,geometry=observations,boundary=boundary,tiling={name:getattr(front,'tile_diagnostic',None) for name,front in fronts.items()},periodic_axis=periodic_measurement,predictor=result,coefficients=coefficients,predictions=predictions,valid=valid,
                    rotor_branch=branch_measurement,preferred_forecast='curved_rollout' if curved_rollout else 'optimized_geometry',event_count=len(e),max_event_t=int(e['t'].max()) if len(e) else None,frame_sha256=digest,processing_ms=elapsed,geometry_ms=gcost,frontend_ms=fcost,model_ms=mcost,predictor_ms=pcost,NN_ms=nn['inference_ms'])
                if event_observer is not None:record['auxiliary_observation']=auxiliary_observation
                stream.write(js(record)+'\n')
                if video is not None:
                    display_image=image
                    if preview_window_ms<20:
                        display_image=np.zeros_like(image)
                        display_events=clean[clean['t']>=round(t*1e6-preview_window_ms*1000)]
                        BaseFrameGenerationAlgorithm.generate_frame(display_events,display_image)
                    if n%50==0:
                        (out/'native_frames').mkdir(exist_ok=True)
                        cv2.imwrite(str(out/'native_frames'/f'{n:06d}.png'),display_image)
                    if branch_measurement is not None and n%10==0:
                        cv2.imwrite(str(out/'rotor_branch_frames'/f'{n:06d}.png'),render_panel(display_image,branch_observer,branch_measurement,t))
                    im=np.full((h,w+400,3),248,np.uint8);im[:,:w]=display_image
                    if len(periodic_points):
                        points=np.clip(np.rint(periodic_points).astype(int),[0,0],[w-1,h-1]);im[points[:,1],points[:,0]]=(0,220,245)
                    if periodic_measurement and periodic_measurement['axis_valid']:
                        line=periodic_measurement['line'];center=np.asarray(line['center']);u=np.array([np.cos(np.deg2rad(line['angle'])),np.sin(np.deg2rad(line['angle']))]);ends=np.rint([center-line['major']*.5*u,center+line['major']*.5*u]).astype(int);cv2.line(im,tuple(ends[0]),tuple(ends[1]),(220,200,0),2)
                        for sign in [-1,1]:cv2.arrowedLine(im,tuple(np.rint(center).astype(int)),tuple(np.rint(center+sign*35*np.asarray(line['normal'])).astype(int)),(220,200,0),2,tipLength=.2)
                    for name,col in [('body_cv',(130,130,130)),('optimized_geometry',(40,170,20))]:
                        if valid and name in predictions:
                            points=np.rint([coefficients[name]['position']]+[predictions[name][str(dt)] for dt in [.1,.2,.4]]).astype(np.int32);cv2.polylines(im,[points],False,col,2);cv2.circle(im,tuple(points[-1]),4,col,2)
                    if curved_rollout and valid and 'curved_rollout' in predictions:
                        points=np.rint([coefficients['optimized_geometry']['position']]+[predictions['curved_rollout'][str(dt)] for dt in (.1,.2,.4)]).astype(np.int32)
                        cv2.polylines(im,[points],False,(180,30,210),2);cv2.circle(im,tuple(points[-1]),5,(180,30,210),2)
                    if rotor_temporal and valid and 'rotor_fused' in predictions and coefficients.get('optimized_geometry'):
                        points=np.rint([coefficients['optimized_geometry']['position']]+[predictions['rotor_fused'][str(dt)] for dt in [.1,.2,.4]]).astype(np.int32)
                        cv2.polylines(im,[points],False,(0,130,245),2);cv2.circle(im,tuple(points[-1]),5,(0,130,245),2)
                        for rotor in branch_measurement['geometry']['observed_rotors']:
                            center=tuple(np.rint([rotor['cx'],rotor['cy']]).astype(int))
                            cv2.circle(im,center,3,(240,180,0),1)
                            cv2.putText(im,'v'+str(rotor.get('visual_track_id','?')),(center[0]+4,center[1]-4),0,.4,(240,180,0),1)
                    if rotor_inertia and branch_measurement:
                        for coast in branch_measurement['inertia149']['coasting_tracks']:
                            center=tuple(np.rint(coast['center']).astype(int))
                            cv2.circle(im,center,7,(150,150,150),1)
                            cv2.putText(im,'coast v'+str(coast['visual_track_id']),(center[0]+8,center[1]),0,.35,(150,150,150),1)
                    lines=[f'RAW t={t:.2f}s','Green: optimized0.4s forecast','Gray: body CV0.4s',f'Rotor model: {result["structure_supported"]}',f'Dense motion: {result.get("dense_motion_fallback",{}).get("active",False)}',f'Geometric motion: {result["current_geometry_velocity_source"]}',f'Process: {elapsed:.1f}ms',f'Preview {preview_window_ms:g}ms; detector20ms','50fps playback, not realtime']
                    if periodic_observer is not None:lines += [f'Periodic axis: {periodic_measurement["axis_valid"]}','Cyan: unsigned observation only']
                    if curved_rollout:lines += ['Magenta: curved154; green: baseline144']
                    if rotor_temporal:lines += ['Orange: fused'+('149' if rotor_inertia else '148')+'0.4s; cyan: visual IDs',str(branch_measurement['expert']['selections_by_horizon'])]
                    for j,line in enumerate(lines):cv2.putText(im,line,(w+10,25+j*25),0,.45,(20,40,50),1)
                    if body.get('bbox'):
                        box=np.asarray(body['bbox']);lo=np.maximum(np.floor(box[:2]-15).astype(int),0);hi=np.minimum(np.ceil(box[2:]+15).astype(int),[w,h]);crop=im[lo[1]:hi[1],lo[0]:hi[0]]
                        if crop.size:
                            scale=min(375/crop.shape[1],300/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);im[340:340+crop.shape[0],w+10:w+10+crop.shape[1]]=crop
                    video.write(im)
                    if n==120:cv2.imwrite(str(out/'preview.png'),im)
                render_cost.append((time.perf_counter()-render_tick)*1000);n+=1
                if n%100==0:print(js(dict(frames=n,t=t,median_processing_ms=float(np.median(cost)))),flush=True)
        assert n==round(duration*50)
        completion=dict(frames=n,rotor_inertia=rotor_inertia,preview_window_ms=preview_window_ms,pose_solver=pose_solver,curved_rollout=curved_rollout,skip_unobserved_geometry=skip_unobserved_geometry,parallel_geometry=parallel_geometry,component_times_are_arm_sum=parallel_geometry,curved_policy='unmodeled_fallback_only',geometry_skipped_ticks=geometry_skipped_ticks,raw_only=True,labels_read=False,output_caches_read=False,sources_unchanged=hashes==fingerprints(),processing_median_ms=float(np.median(cost)),geometry_median_ms=float(np.median(geometry_cost)),gpu_median_ms=float(np.median(nn_cost)),serialization_preview_median_ms=float(np.median(render_cost)),wall_including_cold_start_s=time.perf_counter()-began,
            only_three_forecasts=not rotor_branches and not curved_rollout,rotor_temporal=rotor_temporal,rotor_curvature=rotor_curvature,rotor_branches=rotor_branches,rotor_branch_states=branch_counts,rotor_branch_median_ms=float(np.median(branch_cost)) if branch_cost else None,old_research_controls_not_executed=True,geometry_backend=geometry_backend,motion_backend=motion_backend,periodic_axis_observation_only=periodic_axis,robust_velocity_guard=robust_velocity,damped_forecast=damped_forecast,damping_alpha=DAMPING_ALPHA,velocity_guarded_ticks=velocity_guarded,frontend_median_ms=float(np.median(frontend_cost)),model_median_ms=float(np.median(model_cost)))
        (out/'completion.json').write_text(js(completion),'utf-8');print(js(completion),flush=True)
    finally:
        if geometry_pool is not None:geometry_pool.close()
        if video is not None:video.release()
        if gpu is not None:
            if gpu.poll() is None:
                try:gpu.stdin.write(json.dumps(dict(stop=True))+'\n');gpu.stdin.flush();gpu.wait(timeout=15)
                except (BrokenPipeError,OSError,subprocess.TimeoutExpired):gpu.terminate();gpu.wait(timeout=10)
            for s in [gpu.stdin,gpu.stdout]:
                if s is not None:s.close()
        image=None
        if memory is not None:memory.close();memory.unlink()
        if log is not None:log.close()
        if event_observer is not None:event_observer.close()
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.)
    p.add_argument('--duration',type=float,default=2.)
    p.add_argument('--preview',action='store_true')
    p.add_argument('--verify-frames',action='store_true')
    p.add_argument('--gpu-python',help='CUDA worker interpreter; defaults to private config/env/current Python.')
    p.add_argument('--detector-weights',help='Compatible fixed-shape ONNX detector.')
    p.add_argument('--detector-sha256',help='Expected weight digest; default is the reference detector digest.')
    p.add_argument('--cache-dir',help='Runtime-generated CUDA/Ultralytics settings directory.')
    p.add_argument('--parallel-geometry',action='store_true')
    p.add_argument('--pose-solver',choices=['trf','lm'],default='trf')
    p.add_argument('--curved-rollout',action='store_true',help='Curved unmodeled fallback; structured geometry keeps baseline.')
    p.add_argument('--skip-unobserved-geometry',action='store_true')
    p.add_argument('--geometry-backend',choices=['shared','fast','legacy'],default='shared')
    p.add_argument('--motion-backend',choices=['visible','structure'],default='visible')
    p.add_argument('--periodic-axis',action='store_true')
    p.add_argument('--rotor-inertia',action='store_true',help='Experimental; implies --rotor-temporal.')
    p.add_argument('--preview-window-ms',type=float,default=20.,help='Display-only; inference retains 20 ms.')
    p.add_argument('--rotor-temporal',action='store_true',help='Experimental; implies --rotor-curvature.')
    p.add_argument('--rotor-curvature',action='store_true',help='Experimental; implies --rotor-branches.')
    p.add_argument('--rotor-branches',action='store_true',help='Experimental four/three/two-component geometry and thrust proxy.')
    p.add_argument('--rotor-threshold',type=float,default=4.)
    p.add_argument('--rotor-adaptive',action='store_true')
    p.add_argument('--robust-velocity',action='store_true',help='Rejected velocity experiment; keep off.')
    p.add_argument('--no-damped-forecast',dest='damped_forecast',action='store_false')
    p.set_defaults(damped_forecast=True)
    a=p.parse_args()
    run(**vars(a))
