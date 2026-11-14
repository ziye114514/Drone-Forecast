"""Low-latency paced RAW replay; position branch, latest frame, 5Hz publication.

SDK preprocessing consumes every native20ms batch in one producer. CUDA
detection is throttled independently. Publication time is mapped into source
time so forecasts compensate measured age. This is replay, not live-camera IO.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from multiprocessing import shared_memory
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from forecast_runtime import resolve
from live_stream import LatestReplay, RateGate, MinimumIntervalGate
from live_process import ProcessReplay
from live_motion import LiveMotion, predictions
from live_publication import TimedPublisher
from event_drone_forecast.raw_io import open_raw, raw_chunks
from metavision_sdk_core import BaseFrameGenerationAlgorithm


def js(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())


def receive(process, log):
    for _ in range(64):
        line = process.stdout.readline()
        if not line: raise RuntimeError('CUDA worker stopped; see gpu_worker.log')
        try:
            message = json.loads(line)
            if isinstance(message, dict): return message
        except json.JSONDecodeError: pass
        log.write('NON_JSON_GPU_STDOUT ' + repr(line) + '\n')
        log.flush()
    raise RuntimeError('No CUDA protocol message')


def run(raw, output, start, duration, preview=False, verify_frames=False,
        tracker_hz=25., publish_hz=5., preview_window_ms=5., periodic_axis=False, producer='process',body_association='none',noise_model='legacy',publication_clock='independent',
        gpu_python=None, detector_weights=None, detector_sha256=None, cache_dir=None,frame_transport='events',startup_sync='none',scale_rollout='none',reader_index='default',periodic_sampling='full',periodic_packet='full',center_ensemble_weights=None,center_ensemble_sha256=None,recipe_id=None,local_flow='none',flow_preparation='serial',flow_acceleration='none'):
    if not 0 < publish_hz <= tracker_hz <= 50: raise ValueError('Require 0 < publish Hz <= tracker Hz <= 50')
    if producer not in ('process','thread'):raise ValueError('Unknown RAW producer')
    if body_association not in ('none','bytetrack'):raise ValueError('Unknown body association')
    if publication_clock not in ('tracking','independent'):raise ValueError('Unknown publication clock')
    if frame_transport not in ('events','frames') or (producer!='process' and frame_transport!='events'):
        raise ValueError('Shared frame transport requires process producer')
    if startup_sync not in ('none','reader-ready') or (startup_sync!='none' and producer!='process'):
        raise ValueError('Reader-ready startup synchronization requires process producer')
    if scale_rollout not in ('none','compatible'):raise ValueError('Unknown scale rollout mode')
    if local_flow not in ('none','klt'):raise ValueError('Unknown local flow mode')
    if flow_preparation not in ('serial','gpu-overlap') or (flow_preparation!='serial' and local_flow=='none'):
        raise ValueError('GPU-overlap preparation requires local flow')
    if flow_acceleration not in ('none','velocity-only','position-consistent') or (flow_acceleration!='none' and local_flow=='none'):
        raise ValueError('Flow acceleration requires local flow')
    if reader_index not in ('default','disabled') or (reader_index!='default' and producer!='process'):
        raise ValueError('Explicit RAW index configuration requires process producer')
    if periodic_sampling not in ('full','fixed-quarter') or (periodic_sampling!='full' and not periodic_axis):
        raise ValueError('Fixed pixel sampling requires the optional periodic axis')
    if periodic_packet not in ('full','latest-pixel') or (periodic_packet!='full' and (not periodic_axis or producer!='process')):
        raise ValueError('Current periodic packet compression requires process and periodic axis')
    if any(not np.isfinite(v) for v in (start, duration, tracker_hz, publish_hz, preview_window_ms)):
        raise ValueError('Non-finite configuration')
    if start < 0 or duration <= 0 or any(abs(v * 50 - round(v * 50)) > 1e-7 for v in (start, duration)):
        raise ValueError('Start/duration must align to20ms')
    if not 0 < preview_window_ms <= 20: raise ValueError('Preview window must be in (0,20]ms')
    raw, out = Path(raw).resolve(), Path(output).resolve()
    for protected in (Path('D:/yolov10-main'), Path('E:/dataset/software/yolov10-main')):
        if raw.is_relative_to(protected) or out.is_relative_to(protected): raise ValueError('Protected project')
    if out.exists(): raise FileExistsError(out)
    runtime = resolve(gpu_python, detector_weights, detector_sha256, cache_dir)
    secondary_runtime=None
    if (center_ensemble_weights is None)!=(center_ensemble_sha256 is None):raise ValueError('Secondary model path and SHA256 are both required')
    if center_ensemble_weights is not None:
        secondary_path=Path(center_ensemble_weights).resolve()
        if any(secondary_path.is_relative_to(p) for p in (Path('D:/yolov10-main'),Path('E:/dataset/software/yolov10-main'))):raise ValueError('Protected secondary model path')
        secondary_runtime=resolve(runtime['gpu_python'],secondary_path,center_ensemble_sha256,runtime['cache_dir'])
    reader = open_raw(raw) if producer=='thread' else None
    h, w = reader.get_size() if reader is not None else (720,1280)
    if (h, w) != (720, 1280): raise ValueError('Detector requires1280x720 RAW')
    out.mkdir(parents=True, exist_ok=False)
    # Protect inference from historical prediction/ground-truth input.
    def guard(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve()
        mode = args[1] if len(args) > 1 else None
        writing = isinstance(mode, str) and any(c in mode for c in 'wax+')
        if not writing and p.is_relative_to(ROOT.parent / 'output') and p.suffix.lower() in ('.json', '.jsonl', '.png', '.mp4', '.npz'):
            raise RuntimeError('Live inference cannot read output cache or truth: ' + str(p))
    sys.addaudithook(guard)
    files = [Path(__file__), ROOT/'forecast_runtime.py', ROOT/'live_stream.py', ROOT/'live_process.py', ROOT/'live_decode.py', ROOT/'live_motion.py',ROOT/'live_publication.py',ROOT/'live_measurement_kf.py',ROOT/'body_association.py',ROOT/'live_covariance.py',
             ROOT/'live_worker.py',ROOT/'src/event_drone_forecast/detector_backend.py',
             ROOT/'src/event_drone_forecast/raw_io.py',ROOT/'live_frames.py',ROOT/'live_startup.py']
    if periodic_axis: files += [ROOT/'research/stage141/native_axis141.py', ROOT/'research/stage79/native79.py',ROOT/'src/event_drone_forecast/geometry.py']
    if scale_rollout=='compatible':files.append(ROOT/'live_scale_rollout.py')
    if local_flow!='none':files.append(ROOT/'live_local_flow.py')
    if flow_acceleration!='none':files.append(ROOT/'live_flow_acceleration.py')
    if periodic_axis and periodic_sampling=='fixed-quarter':files.append(ROOT/'live_pixel_axis.py')
    if periodic_packet=='latest-pixel':files.append(ROOT/'live_periodic_packet.py')
    if secondary_runtime is not None:files.append(ROOT/'live_center_ensemble.py')
    if recipe_id is not None:files += [ROOT/'forecast_tracks.py',ROOT/'configs/live_tracks.json']
    files += list((ROOT/'src/event_drone_forecast/position').glob('*.py'))
    files += [p for p in (ROOT/'src/event_drone_forecast/_vendor').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    hashes = lambda: {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    initial = hashes()
    snapshot_root=out/'source_snapshot';snapshot_root.mkdir()
    for p in files:
        relative=p.relative_to(ROOT);data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=initial[relative.as_posix()]:raise RuntimeError('Source changed while snapshotting')
        destination=snapshot_root/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
    (out/'source_snapshot_manifest.json').write_text(js(initial),encoding='utf-8')
    axis = None
    measure_axis = None
    if periodic_axis:
        sys.path.insert(0, str(ROOT/'research/stage141'))
        from native_axis141 import measure_current
        measure_axis = measure_current
        if producer=='thread':
            if periodic_sampling=='fixed-quarter':
                from live_pixel_axis import SDKFixedPixelAxis as NativeAxis
            else:from native_axis141 import NativeAxis
            axis = NativeAxis(w, h)
    noise = buffer = None
    if producer=='thread':
        from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
        noise = ActivityNoiseFilterAlgorithm(w, h, 10000)
        buffer = noise.get_empty_output_buffer()
    native_frame_gate = RateGate(start, tracker_hz)
    def prepare(t, events):
        noise.process_events(events, buffer)
        q = axis.feed(events, t) if axis is not None else None
        # Keep every event in native SDK state, but copy large frame payloads
        # only at the detector cadence. No future event is merged into a frame.
        wanted = t > start + 1e-8 and native_frame_gate.take(t)
        return (buffer.numpy().copy(), q.copy() if q is not None else None) if wanted else (None,None)
    begin = max(0., start - .1)
    source = None
    track_gate, publish_gate = RateGate(start, tracker_hz), MinimumIntervalGate(start, publish_hz)
    forecast_motion=predictions
    if scale_rollout=='compatible':
        from live_scale_rollout import ScaleRolloutMotion,scale_predictions
        motion=ScaleRolloutMotion(tracker_hz,noise_model);forecast_motion=scale_predictions
    else:motion = LiveMotion(tracker_hz,noise_model)
    baseline_motion=None
    if secondary_runtime is not None:
        from live_center_ensemble import blend_current
        baseline_motion=LiveMotion(tracker_hz,noise_model)
    flow_observer=None;flow_costs=[];flow_applied=0;support_costs=[];NN_wait_costs=[];NN_roundtrip_costs=[]
    if local_flow=='klt':
        from live_local_flow import FeatureFlow,frame_support,velocity_coefficient
        cv2.setNumThreads(2);flow_observer=FeatureFlow(crop=True,padding=112)
        flow_background=BaseFrameGenerationAlgorithm.bg_color_default()[0]
        assert flow_background not in [BaseFrameGenerationAlgorithm.on_color_default()[0],BaseFrameGenerationAlgorithm.off_color_default()[0]]
    acceleration_observer=None;acceleration_costs=[];acceleration_applied=0
    if flow_acceleration!='none':
        from live_flow_acceleration import FlowAcceleration
        acceleration_observer=FlowAcceleration()
    memory = gpu = log = image = video = None
    tracks = publications = valid_publications = stale_inputs_skipped = 0
    ages, processing, intervals, producer_ages = [], [], [], []
    previous_publish = None
    publisher=None
    began = time.perf_counter()
    def make_source():
        return (ProcessReplay(raw,start,duration,tracker_hz,periodic_axis,frame_transport,
                              preview=preview,preview_window_ms=preview_window_ms,
                              verify_frames=verify_frames,frame_destination=image,
                              defer_start=startup_sync=='reader-ready',reader_index=reader_index,
                              maximum_packet_age=.16,periodic_sampling=periodic_sampling,periodic_packet=periodic_packet) if producer=='process' else
                LatestReplay(raw_chunks(reader,begin,duration+start-begin,20000),start,prepare))
    try:
        memory = shared_memory.SharedMemory(create=True, size=h*w*3)
        image = np.ndarray((h, w, 3), np.uint8, buffer=memory.buf)
        log = (out/'gpu_worker.log').open('x', encoding='utf-8')
        gpu_args=[runtime['gpu_python'], str(ROOT/'live_worker.py'),
                                '--shared-memory', memory.name, '--precision', '32',
                                '--weights', runtime['detector_weights'], '--expected-sha256', runtime['detector_sha256'],
                                '--cache-dir', runtime['cache_dir']]
        if body_association=='bytetrack':gpu_args+=['--body-association','--frame-rate',str(tracker_hz)]
        if secondary_runtime is not None:gpu_args+=['--center-ensemble-weights',secondary_runtime['detector_weights'],'--center-ensemble-sha256',secondary_runtime['detector_sha256']]
        gpu = subprocess.Popen(gpu_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=log, text=True, encoding='utf-8', bufsize=1,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        ready = receive(gpu, log)
        if not ready.get('ready'): raise RuntimeError(str(ready))
        if startup_sync=='reader-ready':
            source=make_source()
            source.wait_prepared()
        # graph45's tensor warmup does not import/prewarm LetterBox/NMS. Do the
        # complete path before replay starts; this frame is not a measurement.
        image.fill(127)
        warm_begin = time.perf_counter()
        gpu.stdin.write(json.dumps(dict(id=0,verify_frame=False,warmup=True))+'\n'); gpu.stdin.flush()
        warm = receive(gpu,log)
        if warm.get('id') != 0: raise RuntimeError('Detector warmup protocol mismatch')
        complete_detector_warmup_ms = (time.perf_counter()-warm_begin)*1000
        if source is None:source=make_source()
        protocol = dict(mode='latest-paced-RAW-position-branch', raw=str(raw), start=start, duration=duration,
                        tracker_target_hz=tracker_hz, publish_target_hz=publish_hz, native_step_ms=20,
                        reader_backend='SDK EventsIterator' if producer=='process' else 'legacy RawReader',
                        detector_window_ms=20, preview_window_ms=preview_window_ms, runtime=runtime,
                        center_ensemble_runtime=secondary_runtime,recipe_id=recipe_id,local_flow=local_flow,flow_preparation=flow_preparation,flow_acceleration=flow_acceleration,
                        geometry_used=False, thrust_estimated=False, producer=producer,frame_transport=frame_transport,periodic_axis_observation_only=periodic_axis,
                        body_association=body_association,noise_model=noise_model,
                        publication_clock=publication_clock,startup_sync=startup_sync,scale_rollout=scale_rollout,reader_index=reader_index,periodic_sampling=periodic_sampling,periodic_packet=periodic_packet,
                        time_origin='t is publication in source time; observation_t is native event cutoff',
                        compensation='predictions target publication_t+h, observation_predictions target observation_t+h',
                        measurements='Each native batch reaches the preprocessing/optional periodic engine, queue holds only latest observation.',
                        complete_detector_warmup_ms=complete_detector_warmup_ms,
                        publication_gate=('independent fixed deadlines; explicit invalid heartbeat; skip missed slots' if publication_clock=='independent' else 'minimum wall-time interval; no catchup burst'),
                        producer_generation_age_limit_s=.16 if producer=='process' else None,
                        clean_machine_or_live_camera_verified=False, gpu_worker=ready, sources=initial)
        (out/'protocol.json').write_text(js(protocol), encoding='utf-8')
        if preview:
            # Diagnostic video is keyed by publication slots. JSON carries true
            # irregular timestamps; fixed-rate MP4 cannot represent missed slots.
            video = cv2.VideoWriter(str(out/'predictions.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), publish_hz, (w, h))
            if not video.isOpened(): raise RuntimeError('Video encoder unavailable')
        def emit(snapshot,publish_wall,slot=None,scheduled_t=None):
            nonlocal publications,valid_publications,previous_publish
            publish_t=source.start+publish_wall-source.epoch
            if snapshot is None:
                t=c=clean=periodic=seen=None
                preview_frame=None
                body=dict(detected=False,center=None);valid=False
            else:
                t=snapshot['t'];c=snapshot['coefficient'];clean=snapshot['clean'];periodic=snapshot['periodic']
                body=snapshot['body'];valid=snapshot['valid'];seen=snapshot['last_seen']
                preview_frame=snapshot.get('preview_frame')
                if snapshot['state_available_wall']>publish_wall:raise RuntimeError('Estimate published before it was available')
            age=max(0.,publish_t-t) if t is not None else None
            body_age=max(0.,publish_t-seen) if seen is not None else None
            effective_valid=bool(valid and age is not None and age<=.16+1e-8 and body_age is not None and body_age<=.16+1e-8)
            pred,cv=forecast_motion(c,age or 0.);old_pred,old_cv=forecast_motion(c)
            unmodified=predictions(c,age or 0.)[0] if scale_rollout=='compatible' else pred
            row=dict(t=publish_t,observation_t=t,observation_age_ms=age*1000 if age is not None else None,
                     body_age_ms=body_age*1000 if body_age is not None else None,
                     publication_wall_from_replay_start_s=publish_wall-source.epoch,
                     publication_slot=slot,scheduled_publication_t=scheduled_t,
                     estimate_available_t=source.start+snapshot['state_available_wall']-source.epoch if snapshot else None,
                     body=body,coefficients={'live_motion':c},predictions=dict(live_motion=pred,body_cv=cv),
                     observation_predictions=dict(live_motion=old_pred,body_cv=old_cv),preferred_forecast='live_motion',
                     valid=effective_valid,tracking_valid=valid,stale_output_suppressed=bool(valid and not effective_valid),
                     no_estimate=snapshot is None,valid_until_t=seen+.16 if seen is not None else None,
                     forecast_target_t={str(h):publish_t+h for h in (.1,.2,.4)},periodic_axis=periodic,
                     geometry_used=False,thrust_estimated=False)
            if scale_rollout=='compatible':
                row['predictions']['unmodified_motion']=unmodified
                row['scale_rollout']=dict(mode=scale_rollout,rate=c.get('conditional_scale_rate') if c else None,
                    changed={str(h):pred.get(str(h))!=unmodified.get(str(h)) for h in (.1,.2,.4)},
                    interpretation='Conditional apparent-scale correction; not measured depth, attitude or force')
            if baseline_motion is not None:
                baseline=snapshot.get('full_frame_baseline') if snapshot else None
                original_seen=snapshot.get('full_frame_last_seen') if snapshot else None
                row['predictions']['full_frame_baseline']=predictions(baseline,age or 0.)[0]
                row['coefficients']['full_frame_baseline']=baseline
                row['full_frame_baseline_valid']=bool(snapshot and snapshot.get('full_frame_valid') and age is not None and age<=.16+1e-8 and original_seen is not None and publish_t-original_seen<=.16+1e-8)
                row['center_ensemble']=snapshot.get('center_ensemble') if snapshot else None
            if flow_observer is not None:
                position_only=snapshot.get('position_only_coefficient') if snapshot else None
                row['predictions']['position_only']=forecast_motion(position_only,age or 0.)[0]
                row['coefficients']['position_only']=position_only
                row['local_flow']=snapshot.get('local_flow') if snapshot else None
            if acceleration_observer is not None:
                velocity_only=snapshot.get('velocity_only_coefficient') if snapshot else None
                row['predictions']['velocity_only']=forecast_motion(velocity_only,age or 0.)[0]
                row['coefficients']['velocity_only']=velocity_only
                row['flow_acceleration']=snapshot.get('flow_acceleration') if snapshot else None
            stream.write(js(row)+'\n');stream.flush()
            publications+=1;valid_publications+=int(effective_valid)
            if age is not None:ages.append(age*1000)
            if previous_publish is not None:intervals.append(publish_wall-previous_publish)
            previous_publish=publish_wall
            if video is not None:
                display=np.zeros((h,w,3),np.uint8)
                if preview_frame is not None:display=preview_frame.copy()
                elif clean is not None:
                    show=clean[clean['t']>=round(t*1e6-preview_window_ms*1000)]
                    BaseFrameGenerationAlgorithm.generate_frame(show,display)
                if body.get('bbox'):
                    b=np.rint(body['bbox']).astype(int);cv2.rectangle(display,tuple(b[:2]),tuple(b[2:]),(80,200,0),1)
                if effective_valid:
                    pts=np.rint([c['position']]+[pred[str(h)] for h in (.1,.2,.4)]).astype(np.int32)
                    cv2.polylines(display,[pts],False,(180,30,210),2)
                lines=[f'Paced RAW | target publish {publish_hz:g}Hz | {publication_clock}',
                       f'observed {t:.3f}s | published {publish_t:.3f}s | age {age*1000:.1f}ms' if t is not None else f'published {publish_t:.3f}s | no estimate',
                       f'Position branch | scale={scale_rollout} | no thrust',f'valid={effective_valid}; weak={body.get("low_confidence",False)}']
                for j,line in enumerate(lines):cv2.putText(display,line,(15,25+25*j),0,.55,(0,180,255),1)
                video.write(display)
        with (out/'tracking.jsonl').open('x', encoding='utf-8') as tracking, (out/'predictions.jsonl').open('x', encoding='utf-8') as stream:
            with (TimedPublisher(source,duration,publish_hz,emit) if publication_clock=='independent' else nullcontext()) as publisher:
                if startup_sync=='reader-ready':source.release_start()
                for packet in source:
                    t, events, clean = packet.t, packet.events, packet.clean
                    if t <= start + 1e-8 or not track_gate.take(t): continue
                    if source.source_now()-t > .16 + 1e-8:
                        # Do not spend CUDA/CPU time reconstructing a state that
                        # would already fail publication freshness. Native SDK
                        # processing continues independently and preserves periods.
                        stale_inputs_skipped += 1
                        continue
                    tick = time.perf_counter()
                    if packet.frame is not None:
                        if packet.frame is not image:np.copyto(image,packet.frame)
                    else:BaseFrameGenerationAlgorithm.generate_frame(clean, image)
                    digest = hashlib.sha256(image.tobytes()).hexdigest() if verify_frames else None
                    if verify_frames and packet.frame_sha256 is not None and packet.frame_sha256!=digest:
                        raise RuntimeError('Shared producer image generation mismatch')
                    if verify_frames and packet.preview_frame is not None and packet.preview_sha256 is not None:
                        if hashlib.sha256(packet.preview_frame.tobytes()).hexdigest()!=packet.preview_sha256:
                            raise RuntimeError('Shared preview generation mismatch')
                    gpu.stdin.write(json.dumps(dict(id=tracks+1, verify_frame=verify_frames,t=t)) + '\n')
                    gpu.stdin.flush()
                    NN_sent=time.perf_counter();current_support=None;support_ms=0.
                    if flow_observer is not None and flow_preparation=='gpu-overlap':
                        support_start=time.perf_counter();current_support=frame_support(image,flow_background)
                        support_ms=(time.perf_counter()-support_start)*1000
                    NN_wait_start=time.perf_counter()
                    nn = receive(gpu, log)
                    NN_wait_ms=(time.perf_counter()-NN_wait_start)*1000
                    NN_roundtrip_ms=(time.perf_counter()-NN_sent)*1000
                    NN_wait_costs.append(NN_wait_ms);NN_roundtrip_costs.append(NN_roundtrip_ms)
                    if nn.get('id') != tracks+1 or (verify_frames and nn.get('frame_sha256') != digest): raise RuntimeError('Shared-memory frame mismatch')
                    if verify_frames and flow_observer is not None and hashlib.sha256(image.tobytes()).hexdigest()!=digest:
                        raise RuntimeError('Frame changed during current support/GPU overlap')
                    choices=nn.get('associated_boxes',[]) if body_association=='bytetrack' else nn['boxes']
                    if choices:
                        b = max(choices, key=lambda q: q['confidence'])
                        box = b['bbox']
                        body = dict(t=t, detected=True, bbox=box, confidence=b['confidence'],
                                    center=[(box[0]+box[2])/2, (box[1]+box[3])/2])
                        if body_association=='bytetrack':
                            body.update({k:v for k,v in b.items() if k not in ('bbox','confidence','class_id')})
                    else: body = dict(t=t, detected=False, center=None)
                    original_body=body;ensemble_info=None;baseline_c=None;baseline_valid=False
                    if baseline_motion is not None:
                        body,ensemble_info=blend_current(original_body,nn['ensemble_boxes'])
                        baseline_c,baseline_valid,_=baseline_motion.step(t,original_body)
                    c, valid, diagnostic = motion.step(t, body)
                    position_only_c=c;flow_info=None
                    if flow_observer is not None:
                        flow_start=time.perf_counter()
                        early_support_ms=support_ms
                        if current_support is None:
                            support_start=time.perf_counter();current_support=frame_support(image,flow_background)
                            support_ms=(time.perf_counter()-support_start)*1000
                        flow_observation=flow_observer.step(t,current_support,body)
                        c,flow_fusion=velocity_coefficient(c,motion.kf.P[2:4,2:4] if c else None,flow_observation,t)
                        flow_info=dict(observation=flow_observation,fusion=flow_fusion)
                        flow_applied+=int(flow_fusion['applied']);flow_costs.append((time.perf_counter()-flow_start)*1000+early_support_ms)
                        support_costs.append(support_ms)
                        diagnostic['local_flow']=flow_info
                    velocity_only_c=c;acceleration_info=None
                    if acceleration_observer is not None:
                        acceleration_start=time.perf_counter()
                        c,acceleration_info=acceleration_observer.step(t,c,body,flow_observation,flow_acceleration)
                        acceleration_costs.append((time.perf_counter()-acceleration_start)*1000)
                        acceleration_applied+=int(acceleration_info['applied']);diagnostic['flow_acceleration']=acceleration_info
                    if baseline_motion is not None:
                        assert baseline_valid==valid and baseline_motion.seen==motion.seen
                    periodic = None
                    if measure_axis is not None:
                        # The SDK producer owns the stateful engine. Measure
                        # its current immutable packet without a duplicate engine.
                        periodic, _ = measure_axis(packet.periodic, body, t, (c or {}).get('velocity', [0., 0.]), w, h)
                    elapsed = (time.perf_counter() - tick)*1000
                    snapshot=dict(t=t,body=body,coefficient=c,valid=valid,clean=clean,periodic=periodic,preview_frame=packet.preview_frame,
                                  position_only_coefficient=position_only_c,local_flow=flow_info,
                                  velocity_only_coefficient=velocity_only_c,flow_acceleration=acceleration_info,
                                  full_frame_baseline=baseline_c,full_frame_valid=baseline_valid,
                                  full_frame_last_seen=baseline_motion.seen if baseline_motion else None,center_ensemble=ensemble_info,
                                  last_seen=motion.seen,available_wall=packet.available_wall,state_available_wall=time.perf_counter())
                    if publisher is not None:publisher.update(snapshot)
                    tracking.write(js(dict(t=t, body=body, coefficient=c, valid=valid, diagnostic=diagnostic,
                                           position_only_coefficient=position_only_c if flow_observer is not None else None,
                                           velocity_only_coefficient=velocity_only_c if acceleration_observer is not None else None,
                                           estimate_available_t=source.start+snapshot['state_available_wall']-source.epoch,
                                           detector_boxes=nn['boxes'],candidates=nn.get('candidates'),associated_boxes=nn.get('associated_boxes'),
                                           association=nn.get('association'),high_score_invariant=nn.get('high_score_invariant'),
                                           original_body=original_body if baseline_motion else None,
                                           full_frame_baseline=baseline_c,ensemble_boxes=nn.get('ensemble_boxes'),
                                           center_ensemble=ensemble_info,ensemble_inference_ms=nn.get('ensemble_inference_ms'),
                                           detector_frame_sha256=digest,
                                           producer_frame_sha256=packet.frame_sha256,
                                           gpu_frame_sha256=nn.get('frame_sha256'),
                                           input_transport_lag_ms=packet.transport_lag_ms,
                                           event_count=packet.event_count if events is None else len(events),
                                           max_event_t=packet.max_event_t if events is None else (int(events['t'].max()) if len(events) else None),
                                           periodic_axis=periodic, processing_ms=elapsed, NN_ms=nn['inference_ms'],
                                           NN_receive_wait_ms=NN_wait_ms,NN_roundtrip_ms=NN_roundtrip_ms,
                                           support_preparation_ms=support_ms,
                                           support_sha256=hashlib.sha256(current_support.tobytes()).hexdigest() if verify_frames and current_support is not None else None)) + '\n')
                    tracks += 1
                    processing.append(elapsed)
                    if publisher is None and publish_gate.take(source.source_now()):emit(snapshot,time.perf_counter())
        source.close()
        summary = dict(mode=protocol['mode'], producer=producer, tracking_updates=tracks, publication_slots=publications,
                       body_association=body_association,noise_model=noise_model,frame_transport=frame_transport,
                       center_ensemble_enabled=baseline_motion is not None,recipe_id=recipe_id,local_flow=local_flow,
                       local_flow_applied=flow_applied,local_flow_median_ms=float(np.median(flow_costs)) if flow_costs else None,
                       flow_preparation=flow_preparation,support_preparation_median_ms=float(np.median(support_costs)) if support_costs else None,
                       flow_acceleration=flow_acceleration,flow_acceleration_applied=acceleration_applied,
                       flow_acceleration_median_ms=float(np.median(acceleration_costs)) if acceleration_costs else None,
                       NN_receive_wait_median_ms=float(np.median(NN_wait_costs)) if NN_wait_costs else None,
                       NN_roundtrip_median_ms=float(np.median(NN_roundtrip_costs)) if NN_roundtrip_costs else None,
                       publication_clock=publication_clock,startup_sync=startup_sync,scale_rollout=scale_rollout,reader_index=reader_index,periodic_sampling=periodic_sampling,periodic_packet=periodic_packet,
                       periodic_sampling_stats=getattr(source,'periodic_sampling_stats',None),
                       periodic_packet_stats=getattr(source,'periodic_packet_stats',None),
                       valid_publications=valid_publications, tracker_target_hz=tracker_hz, publish_target_hz=publish_hz,
                       achieved_wall_publish_hz=(len(intervals)/sum(intervals)) if intervals else None,
                       publish_interval_p95_ms=float(np.percentile(intervals,95)*1000) if intervals else None,
                       observation_age_median_ms=float(np.median(ages)) if ages else None,
                       observation_age_p95_ms=float(np.percentile(ages,95)) if ages else None,
                       processing_median_ms=float(np.median(processing)) if processing else None,
                       native_batches=source.native_batches, native_events=source.native_events,
                       produced=source.produced, consumed=source.consumed, overwritten=source.overwritten,
                       producer_max_lag_ms=source.max_producer_lag_s*1000,
                       producer_profile=getattr(source,'producer_profile',None),
                       transport_lag_p95_ms=float(np.percentile(source.transport_lags_ms,95)) if getattr(source,'transport_lags_ms',None) else None,
                       frame_descriptor_mismatches=getattr(source,'descriptor_mismatches',0),
                       frame_pool_busy_skipped=getattr(source,'frame_pool_busy_skipped',0),
                       stale_before_generation=getattr(source,'stale_before_generation',0),
                       stale_inputs_skipped=stale_inputs_skipped,
                       complete_detector_warmup_ms=complete_detector_warmup_ms,
                       tracker_missed_slots=track_gate.missed, publication_missed_slots=publish_gate.missed,
                       independent_missed_slots=publisher.missed if publisher else None,
                       publication_lateness_p95_ms=float(np.percentile(publisher.lateness_ms,95)) if publisher and publisher.lateness_ms else None,
                       sources_unchanged=initial==hashes(), raw_only=True, output_caches_read=False, labels_read=False,
                       wall_with_cold_start_s=time.perf_counter()-began,
                       limitations='Paced file replay on this machine; position-only forecast. Not a live-camera or hardware-wide5Hz guarantee.')
        (out/'completion.json').write_text(js(summary), encoding='utf-8')
        print(js(summary), flush=True)
    finally:
        if publisher is not None:publisher.abort()
        if source is not None:source.close()
        if video is not None: video.release()
        if gpu is not None:
            if gpu.poll() is None:
                try:
                    gpu.stdin.write(json.dumps(dict(stop=True))+'\n'); gpu.stdin.flush(); gpu.wait(timeout=15)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired): gpu.terminate(); gpu.wait(timeout=10)
            for stream in (gpu.stdin,gpu.stdout):
                if stream is not None: stream.close()
        image = None
        if memory is not None: memory.close(); memory.unlink()
        if log is not None: log.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True); p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.); p.add_argument('--duration',type=float,default=2.)
    p.add_argument('--preview',action='store_true'); p.add_argument('--verify-frames',action='store_true')
    p.add_argument('--tracker-hz',type=float,default=25.); p.add_argument('--publish-hz',type=float,default=5.)
    p.add_argument('--producer',choices=['process','thread'],default='process')
    p.add_argument('--frame-transport',choices=['events','frames'],default='events')
    p.add_argument('--startup-sync',choices=['none','reader-ready'],default='none')
    p.add_argument('--scale-rollout',choices=['none','compatible'],default='none')
    p.add_argument('--reader-index',choices=['default','disabled'],default='default')
    p.add_argument('--periodic-sampling',choices=['full','fixed-quarter'],default='full')
    p.add_argument('--periodic-packet',choices=['full','latest-pixel'],default='full')
    p.add_argument('--body-association',choices=['none','bytetrack'],default='none')
    p.add_argument('--noise-model',choices=['legacy','continuous'],default='legacy')
    p.add_argument('--publication-clock',choices=['tracking','independent'],default='independent')
    p.add_argument('--preview-window-ms',type=float,default=5.); p.add_argument('--periodic-axis',action='store_true')
    p.add_argument('--gpu-python'); p.add_argument('--detector-weights'); p.add_argument('--detector-sha256'); p.add_argument('--cache-dir')
    p.add_argument('--center-ensemble-weights');p.add_argument('--center-ensemble-sha256')
    p.add_argument('--recipe-id',help='launcher recipe provenance; actual model SHA is recorded separately')
    p.add_argument('--local-flow',choices=['none','klt'],default='none')
    p.add_argument('--flow-preparation',choices=['serial','gpu-overlap'],default='serial')
    p.add_argument('--flow-acceleration',choices=['none','velocity-only','position-consistent'],default='none')
    run(**vars(p.parse_args()))
