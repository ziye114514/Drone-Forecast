"""RAW physical forecast with explicit geometry, period and localization profiles.

Source updates, current NN requests, effective observations and independently
timed publications are measured separately; a30Hz clock is not30 valid frames.
"""
from pathlib import Path
import argparse
import gc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,TimeoutError as FutureTimeout
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import shared_memory
# Bound tiny matrix BLAS work before importing NumPy/SciPy. Large default BLAS
# thread teams contend with CUDA preprocessing and introduce latency spikes.
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage79')]
from live_decode import native_batches
from forecast_runtime import resolve
from fast_structure_front import FastStructure,roi_bounds,NativeROI,rotor_rates
from fast_force_response import FastForceResponse
from visibility_period import LegacyPeriodProxy,VisibilityPeriodProxy,ConservativeSlotPeriodProxy
from axis_amplitude_bridge import current_axis_qualified
from live_frequency_pixels import latest_pixels
from live_scale_rollout import ScaleRolloutMotion,scale_predictions
from live_structure_bridge import QualifiedCenter,GeometryMotion
import native79


def js(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())


def receive(p):
    for _ in range(64):
        s=p.stdout.readline()
        if not s:raise RuntimeError('Worker exited; inspect gpu_worker.log')
        try:return json.loads(s)
        except json.JSONDecodeError:pass
    raise RuntimeError('GPU protocol failure')


def track(previous,current,body,rigid_motion=False,center_hint=None,spatial_motion=False,inertial_seed=None,seed_backward=False,rigid_box_center=False,body_core_points=False,wide_motion_search=False):
    """Current-image translation only; no inertial extrapolation called detection."""
    if previous is None or body is None or not body.get('detected'):return None,dict(reason='no_previous_ROI')
    box=np.asarray(body['bbox']);search=box;seed=None
    if inertial_seed is not None:
        candidate=np.asarray(inertial_seed,float)
        allowed=.5*np.hypot(*current.shape) if wide_motion_search else .4*max(box[2:]-box[:2])
        if candidate.shape==(2,) and np.isfinite(candidate).all() and np.linalg.norm(candidate)<=allowed:
            seed=candidate
            if wide_motion_search:
                shifted=box+np.tile(seed,2)
                search=np.r_[np.minimum(box[:2],shifted[:2]),np.maximum(box[2:],shifted[2:])]
    x,y,X,Y=roi_bounds(search,padding=.35)
    if X-x<12 or Y-y<12:return None,dict(reason='previous_ROI_outside_sensor_or_too_small')
    a=previous[y:Y,x:X];b=current[y:Y,x:X]
    mask=np.zeros(a.shape,np.uint8);lo=np.maximum(np.floor(box[:2]).astype(int)-[x,y],0)
    hi=np.minimum(np.ceil(box[2:]).astype(int)-[x,y],[X-x,Y-y]);mask[lo[1]:hi[1],lo[0]:hi[0]]=255
    core_ids=core_candidates=None
    if body_core_points:
        from body_core_motion import points as select_points
        points,core_ids,core_candidates=select_points(a,mask,np.asarray(body['center'])-[x,y],min(box[2:]-box[:2])/4)
    else:points=cv2.goodFeaturesToTrack(a,60,.01,4,mask=mask)
    if points is None or len(points)<6:return None,dict(reason='too_few_current_features')
    initial=None if seed is None else (points+np.asarray(seed,np.float32)).astype(np.float32)
    q,ok,_=cv2.calcOpticalFlowPyrLK(a,b,points,initial,winSize=(15,15),maxLevel=2,
        flags=0 if initial is None else cv2.OPTFLOW_USE_INITIAL_FLOW)
    back,ok2,_=cv2.calcOpticalFlowPyrLK(b,a,q,points.copy() if seed_backward else None,winSize=(15,15),maxLevel=2,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW if seed_backward else 0)
    d=(q-points).reshape(-1,2);fb=np.linalg.norm((back-points).reshape(-1,2),axis=1)
    keep=ok.ravel().astype(bool)&ok2.ravel().astype(bool)&(fb<=1.5)&np.isfinite(d).all(axis=1)
    body_motion=None
    if body_core_points:
        from body_core_motion import qualify
        origin=np.array([x,y])
        selected,body_motion=qualify(points.reshape(-1,2)+origin,q.reshape(-1,2)+origin,fb,keep,core_ids,
            body['center'] if center_hint is None else center_hint,box)
        if selected is not None:keep=selected
    if keep.sum()<6:return None,dict(reason='failed_forward_backward_support',points=int(keep.sum()))
    delta=np.median(d[keep],axis=0);scatter=np.linalg.norm(d-delta,axis=1)
    keep&=scatter<=max(2.,3.*float(np.median(scatter[keep])))
    if keep.sum()<6:return None,dict(reason='incoherent_translation')
    delta=np.median(d[keep],axis=0)
    large=np.linalg.norm(delta)>.4*max(box[2:]-box[:2]);large_support=None
    if large and wide_motion_search:
        from live_local_flow import reused_similarity
        origin=np.array([x,y])
        large_support=reused_similarity(points.reshape(-1,2)[keep]+origin,q.reshape(-1,2)[keep]+origin,fb[keep],body['center'],center_support=True)
        qualified=bool(large_support.get('valid') and large_support.get('inliers',0)>=12 and
            large_support.get('inlier_fraction',0)>=.75 and large_support.get('center_in_track_hull') and
            np.linalg.eigvalsh(large_support['displacement_covariance_px2']).max()<=9.)
        if not qualified:return None,dict(reason='large_translation_without_strict_current_rigid_support',support=large_support)
        delta=np.asarray(large_support['displacement'])
    elif large:return None,dict(reason='translation_too_large')
    shifted=box+np.tile(delta,2)
    if shifted[0]>=current.shape[1] or shifted[1]>=current.shape[0] or shifted[2]<=0 or shifted[3]<=0:
        return None,dict(reason='current_translated_ROI_outside_sensor')
    result=dict(body,bbox=(box+np.tile(delta,2)).tolist(),center=(np.asarray(body['center'])+delta).tolist(),
        measurement_source='current_KLT_translation',detected=True)
    covariance=np.cov(d[keep].T)+np.eye(2)*max(1.,float(np.median(fb[keep]**2)))
    info=dict(reason='current_image_translation',points=int(keep.sum()),delta=delta.tolist(),
        displacement_covariance_px2=covariance.tolist(),covariance_uses_full_track_scatter_without_independence_gain=True)
    if body_core_points:info['body_core_motion']=dict(body_motion,candidates=core_candidates)
    if wide_motion_search:info['wide_motion_search']=dict(search_bounds=[int(x),int(y),int(X),int(Y)],
        seed_is_not_a_measurement=True,large_motion_current_support=large_support,
        old_size_displacement_gate_exceeded=bool(large),new_flow_passes=0)
    if seed is not None or seed_backward:
        info['optimization_initialisation']=dict(forward_displacement=None if seed is None else seed.tolist(),
            backward_original_coordinates=seed_backward,prediction_is_only_search_initialisation=True,
            current_bidirectional_image_validation_unchanged=True)
    if rigid_motion:
        from live_local_flow import reused_similarity
        start=time.perf_counter();origin=np.array([x,y])
        info['similarity_motion']=reused_similarity(points.reshape(-1,2)[keep]+origin,q.reshape(-1,2)[keep]+origin,fb[keep],
            body['center'] if center_hint is None else center_hint,spatial=spatial_motion and body.get('low_confidence',False),center_support=rigid_box_center)
        info['similarity_motion']['CPU_ms']=(time.perf_counter()-start)*1000
        info['similarity_motion']['evaluation_center_source']='previous_qualified_five_circle_center' if center_hint is not None else 'previous_body_box_center'
        if rigid_box_center:
            from rigid_center_translation import choose
            delta,diagnostic=choose(delta,info['similarity_motion'],box)
            info['rigid_center_transport']=diagnostic
            if diagnostic['applied']:
                result.update(bbox=(box+np.tile(delta,2)).tolist(),center=(np.asarray(body['center'])+delta).tolist())
                info.update(delta=delta.tolist(),displacement_covariance_px2=info['similarity_motion']['displacement_covariance_px2'])
    return result,info


class PhysicalPositionState:
    """Only p/v needed by mechanics, with no parallel acceleration/scale fit."""
    def __init__(self):
        from live_structure_bridge import NoiseKF
        self.kf=NoiseKF();self.seen=None;self.observations=0
    def step(self,t,body):
        detected=bool(body.get('detected'))
        if self.seen is not None and t-self.seen>.5:self.__init__()
        if detected:self.seen=t;self.observations+=1
        c=self.kf.step(t,body.get('center') if detected else None)
        valid=bool(c is not None and self.observations>=5 and self.seen is not None and t-self.seen<=.16)
        return c,valid,dict(position_velocity_initial_state_only=True,parallel_kinematic_acceleration=False)


def run(args):
    pipeline_started=time.perf_counter()
    # Import finite helpers before source time starts; the first tracking
    # frame otherwise pays module/file-loader latency inside its deadline.
    from native_render import occupancy_frame
    if args.rigid_box_center:
        import rigid_center_translation
    from fast_huber import configure as configure_huber,counters as huber_counters
    huber_provenance=configure_huber(args.jit_huber)
    conic_residual=None
    if args.jit_conic_residual:
        from fast_geometry_compute import configure_residual
        conic_residual=configure_residual(True)
    raw=Path(args.raw).resolve();out=Path(args.output).resolve()
    for protected in [Path('D:/yolov10-main'),Path('E:/dataset/software/yolov10-main')]:
        if raw.is_relative_to(protected) or out.is_relative_to(protected):raise ValueError('Protected project')
    if out.exists():raise FileExistsError(out)
    if args.duration<=0 or args.hz<=0:raise ValueError('Positive duration/hz required')
    model=Path(args.detector_model).resolve() if getattr(args,'detector_model',None) else ROOT/'models/live_position_accumulated_v1.onnx'
    expected=getattr(args,'detector_sha256',None) or 'ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4'
    runtime=resolve(args.gpu_python,model,expected,None)
    verifier_model=getattr(args,'weak_verifier_model',None)
    verifier_domain=None;verifier_max_side=getattr(args,'weak_verifier_max_side',None)
    if verifier_model:
        verifier_model=Path(verifier_model).resolve()
        if hashlib.sha256(verifier_model.read_bytes()).hexdigest()!=args.weak_verifier_sha256:
            raise ValueError('Weak verifier SHA256 mismatch')
        domain_path=verifier_model.parent/'verifier_domain.json'
        if verifier_max_side is None and domain_path.exists():
            data=domain_path.read_bytes();verifier_domain=json.loads(data)
            if verifier_domain['verifier_sha256']!=args.weak_verifier_sha256:raise ValueError('Verifier domain/weight hash mismatch')
            verifier_max_side=float(verifier_domain['maximum_native_box_side_px'])
            verifier_domain=dict(verifier_domain,path=str(domain_path),sha256=hashlib.sha256(data).hexdigest())
    if verifier_max_side is None:verifier_max_side=192.
    if not np.isfinite(verifier_max_side) or not 12.<=verifier_max_side<=192.:raise ValueError('Verifier maximum side must be12..192 native pixels')
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
    from metavision_sdk_core import BaseFrameGenerationAlgorithm
    cv2.setNumThreads(2)
    out.mkdir(parents=True);snapshot=out/'source_snapshot';snapshot.mkdir()
    sources={}
    for name in ('fast_huber.py','native_huber.py','native_conic_residual.py','nn_recovery_budget.py','position_net_force.py','native_body_workspace.py','singer_position_state.py','current_box_gate.py'):
        content=(ROOT/name).read_bytes();(snapshot/name).write_bytes(content)
        sources[name]=hashlib.sha256(content).hexdigest()
    for f in ['run_fast_structure.py','fast_raw_reader.py','rigid_center_translation.py','body_core_motion.py','physical_publication.py','live_publication.py','period_process.py','sdk_activity_backend.py','native/sdk_activity_avx2.cpp','fast_structure_front.py','fast_geometry_compute.py','native_counts_numba.py','native_geometry_numba.py','native_latest_frequency.py','shared_image_motion.py','curved_flow_motion.py','flow_anchor_gauge.py','adaptive_force_memory.py','thrust_trend_forecast.py','flow_force_balance.py','projected_unknown_force.py','axis_amplitude_bridge.py','live_local_flow.py','live_flow_acceleration.py','native_render.py','responsive_force.py','body_association.py','shared_arc_recovery.py','partial_arc_information.py','continuous_five_circle.py','fast_force_response.py','visibility_period.py','rotor_thrust_bank.py','propeller_physics.py','normal_turn_forecast.py','temporal_period_support.py','live_frequency_pixels.py','live_decode.py',
              'forecast_runtime.py','nn_search_schedule.py','position_history_gauge.py','aligned_period_recovery.py','body_period_stream.py','endpoint_force_balance.py','five_circle_model.py','live_worker.py','live_roi_refine.py','motion_observability.py','live_flow_spatial.py','forecast_output.py','live_structure_bridge.py','live_scale_rollout.py',
              'src/event_drone_forecast/geometry.py','research/stage6/silhouette.py','research/stage79/native79.py',
              'research/stage146/branches146.py','research/stage147/curvature147.py']:
        content=(ROOT/f).read_bytes();p=snapshot/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
        sources[f]=hashlib.sha256(content).hexdigest()
    stats=raw.stat()
    if args.sdk_avx2 and not args.prefilter_decode:
        from sdk_activity_backend import SDKActivityAVX2
        noise=SDKActivityAVX2(1280,720,10000)
    else:noise=ActivityNoiseFilterAlgorithm(1280,720,10000)
    clean_buffer=noise.get_empty_output_buffer()
    from live_pixel_axis import SDKFixedPixelAxis
    periodic=SDKFixedPixelAxis(1280,720,all_pixels=getattr(args,'period_all_pixels',False));period_roi=NativeROI()
    frequency_latest=None
    if args.jit_counts and not args.process_period:
        from native_latest_frequency import LatestFrequency
        frequency_latest=LatestFrequency(1280,720,periodic.output.numpy().dtype)
    structure=FastStructure(args.reuse_conic_uncertainty,not args.legacy_structure,not args.no_shared_arc_recovery,args.guide_full_only,args.bipolar_guide_area and not args.strict_guide_area,not args.strict_guide_area,not args.no_current_shape_feedback,args.partial_arc_information,optimized_compute=args.optimized_compute,side_axis_budget=args.side_axis_budget,native_histogram=args.native_histogram,jit_counts=args.jit_counts,jit_conics=args.jit_conics,body_workspace=args.body_workspace,legacy_visibility=True);motion=GeometryMotion(args.hz) if args.with_kinematic_control else PhysicalPositionState()
    if args.singer_state:
        if args.with_kinematic_control:raise ValueError('Singer state is one physical position state, separate from kinematic control')
        from singer_position_state import SingerPositionState
        motion=SingerPositionState()
    image_motion=None
    if args.reuse_image_motion:
        from shared_image_motion import SharedImageMotion
        image_motion=SharedImageMotion(reanchor_velocity=not args.legacy_motion_anchor_gate,
            same_window_motion=getattr(args,'same_window_motion',False),curved_motion=getattr(args,'curved_motion',False),
            anchor_gauge=getattr(args,'flow_anchor_gauge',False),weak_motion_support=getattr(args,'weak_motion_support',False),
            visible_motion_support=getattr(args,'visible_motion_support',False),
            spatial_weak_motion=getattr(args,'spatial_weak_motion',False),
            position_history_rebase='stable-axis' if args.position_history_rebase else False,
            contained_reanchor_motion=args.contained_reanchor_motion)
    position_net_force=None
    if args.position_net_force:
        from position_net_force import PositionNetForce
        position_net_force=PositionNetForce()
    amplitude_bridge=None
    if args.axis_amplitude_bridge:
        from axis_amplitude_bridge import AxisAmplitudeBridge
        amplitude_bridge=AxisAmplitudeBridge()
    normal_preview=temporal_period=force_memory=thrust_trend=aligned_period=None
    if getattr(args,'aligned_period_recovery',False):
        from aligned_period_recovery import AlignedPeriodRecovery
        aligned_period=AlignedPeriodRecovery(configuration=args.aligned_period_filter)
    if getattr(args,'body_period_stream',False):
        from body_period_stream import BodyPeriodStream
        aligned_period=BodyPeriodStream(configuration=args.aligned_period_filter)
    if getattr(args,'adaptive_force_memory',False):
        from adaptive_force_memory import AdaptiveForceMemory
        force_memory=AdaptiveForceMemory(extend_calibrated=not args.memory_requires_current_period,
            allow_shorter=getattr(args,'shorter_force_memory',False))
    if getattr(args,'thrust_trend_forecast',False):
        from thrust_trend_forecast import SourceAwareThrustTrendForecast
        thrust_trend=SourceAwareThrustTrendForecast()
    if getattr(args,'normal_turn_forecast',False):
        from normal_turn_forecast import NormalTurnForecast
        normal_preview=NormalTurnForecast()
    if getattr(args,'temporal_period_support',False):
        from temporal_period_support import TemporalPeriodSupport
        temporal_period=TemporalPeriodSupport()
    force_class=FastForceResponse
    if args.responsive_force:
        from responsive_force import DampedResponsiveForce
        force_class=DampedResponsiveForce
    if args.flow_force_balance:
        from flow_force_balance import FlowBalancedForce
        force_class=FlowBalancedForce
    if args.tangent_disturbance:
        from projected_unknown_force import ProjectedResidualForce
        force_class=ProjectedResidualForce
    if args.endpoint_force_balance:
        from endpoint_force_balance import EndpointForceBalance
        force_class=EndpointForceBalance
    centering=QualifiedCenter();force=force_class(early_disturbance=not args.delayed_disturbance_control,
        hold_acceleration_gaps=not args.no_force_gap_hold and not args.delayed_disturbance_control,
        gain_signal_to_noise=0. if args.legacy_gain_quality or args.delayed_disturbance_control else 2.)
    if getattr(args,'direct_force_balance',False):force.direct_balance=True
    if getattr(args,'curved_force_balance',False):force.curved_balance=True
    if getattr(args,'evidence_gain',False):force.evidence_gain=True
    period_selector=(VisibilityPeriodProxy() if getattr(args,'pooled_axis_period',False) else
        ConservativeSlotPeriodProxy() if getattr(args,'visibility_period',False) else LegacyPeriodProxy())
    if getattr(args,'rotor_thrust_bank',False):
        from rotor_thrust_bank import RotorThrustBankProxy
        period_selector=RotorThrustBankProxy()
    memory=gpu=log=video=None;records=[];counter=0;nn_count=0;previous=None;body=None;last_nn=None;last_nn_request=None;last_body_observed=None
    if getattr(args,'lean_history',False):
        from forecast_output import LeanHistory
        records=LeanHistory()
    pool=ThreadPoolExecutor(max_workers=1);reader_pool=ThreadPoolExecutor(max_workers=1)
    period_pool=ThreadPoolExecutor(max_workers=1)
    raw_process=None;period_process=None;read_future=None;pending=None;pending_gray=None;pending_t=None;pending_hash=None;pending_reference=None
    coeff=None;first_decode=True;roi_memory=None;weak_semantic_memory=None;nn_service_history=[]
    protocol=dict(entry='run_fast_structure.py',strategy=3,target_fps=args.hz,nn_stride=args.nn_stride,
        native_step_us=round(1e6/args.hz),geometry_window_us=20000,paced=args.paced,async_nn=args.async_nn,
        detector_window_us=round(1e6/args.hz) if args.native_detector_window else 20000,
        detector_uses_complete_native_packet=args.native_detector_window,
        detector_native_window_fallback=args.detector_window_fallback,
        detector_fallback_max_recent_side_px=192.,
        detector_native_window_small_targets_only=args.small_target_native_window,
        current_small_target_NN_every_source_tick=args.synchronous_small_nn,
        detector_max_window_us=round(1e6/args.hz) if args.native_detector_window or args.small_target_native_window or args.detector_window_fallback else 20000,
        detector_window_policy='full_native' if args.native_detector_window else 'known_small_without_model_or_recent9ticks_geometry' if args.small_target_native_window else '20ms_with_optional_empty_retry' if args.detector_window_fallback else '20ms',
        joint_singer_position_state=args.singer_state,
        unresolved_small_geometry_stride=args.unresolved_geometry_stride,
        synchronous_first_source_acquisition=args.synchronous_start,
        synchronous_current_NN_during_acquisition=args.synchronous_acquisition,
        current_NN_box_gate=args.current_box_gate,
        physical_only=not args.with_kinematic_control,
        periodic_sampling=('All ROI pixels; ALL native event times retained' if getattr(args,'period_all_pixels',False) else 'Existing SDKFixedPixelAxis seed141000 fixed25% pixels; ALL native times at each retained pixel'),
        period_selector=type(period_selector).__name__,optimized_compute=args.optimized_compute,
        frequency_observer_all_pixels=getattr(args,'period_all_pixels',False),
        headless_normal_turn_forecast=getattr(args,'normal_turn_forecast',False),
        same_window_motion_consistency=getattr(args,'same_window_motion',False),
        current_curved_motion=getattr(args,'curved_motion',False),
        direct_force_balance=getattr(args,'direct_force_balance',False),
        curved_force_balance=getattr(args,'curved_force_balance',False),
        evidence_gain=getattr(args,'evidence_gain',False),
        endpoint_force_balance=getattr(args,'endpoint_force_balance',False),
        flow_anchor_gauge=getattr(args,'flow_anchor_gauge',False),
        side_position_history_rebase=args.position_history_rebase,
        deadline_bounded_NN_reply_wait=args.deadline_nn,
        cold_acquisition_at_ordinary_NN_stride=args.empty_search_stride,
        contained_weak_patch_reanchor_motion=args.contained_reanchor_motion,
        adaptive_force_memory=getattr(args,'adaptive_force_memory',False),
        calibrated_force_memory=bool(getattr(args,'adaptive_force_memory',False) and not args.memory_requires_current_period),
        weak_small_ROI_verification=getattr(args,'weak_roi_verify',False),
        weak_semantic_motion_support=getattr(args,'weak_motion_support',False),
        spatially_supported_weak_motion=getattr(args,'spatial_weak_motion',False),
        physical_state_initialises_KLT=getattr(args,'inertial_klt',False),
        weak_verifier_model=None if not verifier_model else str(verifier_model),
        weak_verifier_sha256=getattr(args,'weak_verifier_sha256',None),
        weak_verifier_max_side_px=verifier_max_side,weak_verifier_training_domain=verifier_domain,
        partially_visible_body_motion_support=getattr(args,'visible_motion_support',False),
        shorter_force_memory=getattr(args,'shorter_force_memory',False),
        thrust_trend_forecast=getattr(args,'thrust_trend_forecast',False),
        aligned_period_recovery=getattr(args,'aligned_period_recovery',False),
        aligned_period_filter=getattr(args,'aligned_period_filter','strict7'),
        body_period_stream=getattr(args,'body_period_stream',False),
        temporal_distinct_pixel_period_support=getattr(args,'temporal_period_support',False),
        prefilter_in_decoder=args.prefilter_decode,
        parallel_period_geometry=args.parallel_period,
        adaptive_period_execution=args.adaptive_period,
        separate_period_process=args.process_period,
        side_axis_budget=args.side_axis_budget,force_model=type(force).__name__,
        goal_profile=args.goal_profile,
        histogram_backend='Numba_fused_ROI_count' if args.jit_counts else 'OpenCV_float32_exact_integer_counts' if args.native_histogram else 'NumPy_int64_then_float32',
        count_kernel_provenance=None if structure.count_current is None else structure.count_current.provenance,
        conic_kernel_provenance=None if structure.uncertainty_current is None else structure.uncertainty_current.provenance,
        latest_frequency_kernel_provenance=None if frequency_latest is None else frequency_latest.provenance,
        frame_hash_verification=args.verify_frames,reuse_current_image_motion=args.reuse_image_motion,
        same_window_flow_force_balance=args.flow_force_balance,
        current_axis_with_recent_amplitude=args.axis_amplitude_bridge,
        current_strong_reanchor_reuses_velocity=not args.legacy_motion_anchor_gate,
        reuse_rigid_current_track_pairs=args.rigid_motion,
        rigid_center_transports_box=args.rigid_box_center,
        body_core_points=args.body_core_points,
        weak_body_continuation=args.weak_body_continuation,
        wide_motion_search=args.wide_motion_search,
        budgeted_current_NN_recovery=args.budgeted_nn_recovery,
        position_net_force_fallback=args.position_net_force,
        body_workspace=None if structure.body_workspace is None else structure.body_workspace.provenance,
        legacy_realtime30_profile=args.legacy_realtime30,
        huber_solver=huber_provenance,
        conic_residual_backend=None if conic_residual is None else conic_residual.provenance,
        independent_publication_hz=args.publish_hz,
        partial_tangent_unknown_force=args.tangent_disturbance,
        structure_guided_NN_cadence=args.structure_guided_nn,
        startup_clock_scope='run entry, includes optional JIT import/compile and module initialization',
        current_box_association=args.body_association,
        current_NN_when_KLT_unavailable=args.current_nn_on_loss,roi_recovery_after_current_full_frame_miss=args.roi_recovery,
        decode_prefetch=not args.serial_decode,decode_queue_capacity=2 if not args.thread_decode and not args.serial_decode else 1,drop_source_frames=False,
        decoder_overflow_policy='bounded queued immutable packet; preserve all native events',
        raw_read_chunk=args.raw_read_chunk,
        raw_native_read_buffers=getattr(args,'raw_read_buffers',3),
        native_RAW_reader_backend=getattr(args,'raw_reader_backend','iterator'),
        native_RAW_shared_slot_events=args.raw_slot_events,
        output_schema='compact.v2' if getattr(args,'compact_output',False) else 'full_research_diagnostics',
        lean_history=args.lean_history,
        defer_cyclic_GC_during_finite_stream=args.defer_cyclic_gc,
        defer_decoder_cyclic_GC=args.defer_decoder_gc,
        local_SDK_AVX2_filter=args.sdk_avx2,
        conic_uncertainty_reuse=args.reuse_conic_uncertainty,
        continuous_shared_geometry=not args.legacy_structure,
        shared_current_arc_recovery=not args.no_shared_arc_recovery,
        guide_reference_policy='full_only' if args.guide_full_only else 'rank6_current_anchors_and_hub_uncertainty',
        guide_area_source='existing124_binary_bipolar_occupancy' if args.bipolar_guide_area and not args.strict_guide_area else 'strong_count_threshold',
        guide_area_evidence='union_iou_only' if args.strict_guide_area else 'union_iou_or_current_coverage_and_ring_contrast',
        current_conic_shape_feedback=not args.no_current_shape_feedback,
        partial_arc_information=args.partial_arc_information,
        disturbance_learning='first_matured_0.2s_row' if not args.delayed_disturbance_control else 'historical_8row_gate',
        force_reference_gauge='adjusted_position_measurement' if not args.delayed_disturbance_control else 'historical_raw_body_center',
        force_gap_hold_s=.12 if not args.no_force_gap_hold and not args.delayed_disturbance_control else 0.,
        gain_signal_to_noise_minimum=0. if args.legacy_gain_quality or args.delayed_disturbance_control else 2.,
        geometry_input='reuse_current_expanded_frequency_ROI; all native times inside geometry ROI retained',
        separate_decoder_process=not args.thread_decode and not args.serial_decode,
        raw=str(raw),raw_size=stats.st_size,raw_mtime_ns=stats.st_mtime_ns,start=args.start,duration=args.duration,
        sources=sources,runtime=runtime,input='RAW and fixed detector only; no labels or output cache read',
        input_force='relative flicker squared times image normal; sign uses image-up prior',
        limits=['2D weak projection','no true RPM, metric force or external pose truth',
                'short pilot; no claim of maneuver accuracy before scoring',
                'KLT translations distinguishable from fresh NN boxes','no default v7 replacement'])
    step_us=round(1e6/args.hz)
    # Identical source alignment in both compute paths.
    aligned_start=(round(args.start*1e6)//step_us)*step_us/1e6
    aligned_duration=(round(args.duration*1e6)//step_us)*step_us/1e6
    protocol.update(actual_iterator_start=aligned_start,actual_duration=aligned_duration)
    iterator=iter(native_batches(raw,aligned_start,aligned_duration,False,step_us,read_chunk=args.raw_read_chunk,
        read_buffers=getattr(args,'raw_read_buffers',3),reader_backend=getattr(args,'raw_reader_backend','iterator')))
    def read_owned():
        before=time.perf_counter();packet=next(iterator,None)
        if packet is None:return None
        stamp,e=packet
        if not e.flags.c_contiguous:raise RuntimeError('Expected contiguous native event batch')
        owned=e.view(np.uint8).copy().view(e.dtype);owned.setflags(write=False)
        return stamp,owned,(time.perf_counter()-before)*1000
    def start_reader():
        nonlocal raw_process,period_process,read_future
        protocol['decoder_started_after_s']=time.perf_counter()-pipeline_started
        if not args.thread_decode and not args.serial_decode:
            from fast_raw_reader import ProcessRAW
            raw_process=ProcessRAW(raw,aligned_start,aligned_duration,step_us,filter_events=args.prefilter_decode,read_chunk=args.raw_read_chunk,avx2_filter=args.sdk_avx2,
                read_buffers=getattr(args,'raw_read_buffers',3),reader_backend=getattr(args,'raw_reader_backend','iterator'),slot_events=args.raw_slot_events,defer_cyclic_gc=args.defer_decoder_gc)
        elif not args.serial_decode:read_future=reader_pool.submit(read_owned)
        if args.process_period:
            from period_process import PeriodProcess
            period_process=PeriodProcess(jit=args.jit_counts)
    def measure_period(selected,t):
        if period_process is not None:return period_process.compute(selected,t)
        started=time.perf_counter();q=periodic.feed(selected,t)
        q=q[q['t']>=round(t*1e6)-20000];pooled_input=q
        in_range=bool(np.all((q['frequency']>=60.)&(q['frequency']<=1200.)))
        q=latest_pixels(q,1280) if frequency_latest is None else frequency_latest(q)
        if in_range:pooled_input=q
        return q,pooled_input,in_range,(time.perf_counter()-started)*1000
    gc_was_enabled=gc.isenabled();gc_events=[];gc_started={};gc_callback=None
    publisher=None;publication_stream=None;publication_records=[];publication_cost=[]
    from types import SimpleNamespace
    publication_clock=SimpleNamespace(epoch=None,start=args.start,done=False)
    try:
        if args.optimized_compute:start_reader()
        memory=shared_memory.SharedMemory(create=True,size=720*1280*3*(2 if args.detector_window_fallback else 1))
        shared_image=np.ndarray((720,1280,3),np.uint8,buffer=memory.buf);shared_image[:]=52
        shared_native_image=np.ndarray((720,1280,3),np.uint8,buffer=memory.buf,offset=720*1280*3) if args.detector_window_fallback else None
        image=np.empty((720,1280,3),np.uint8)
        log=(out/'gpu_worker.log').open('x',encoding='utf-8')
        worker_args=[runtime['gpu_python'],str(ROOT/'live_worker.py'),'--shared-memory',memory.name,
            '--weights',runtime['detector_weights'],'--expected-sha256',runtime['detector_sha256'],
            '--cache-dir',runtime['cache_dir']]
        if args.body_association:worker_args+=['--body-association','--frame-rate',str(args.hz if args.synchronous_small_nn or args.synchronous_start else args.hz/args.nn_stride)]
        if args.current_box_gate:worker_args+=['--current-box-gate']
        if verifier_model:worker_args+=['--weak-verifier-weights',str(verifier_model),'--weak-verifier-sha256',args.weak_verifier_sha256,
            '--weak-verifier-max-side',str(verifier_max_side)]
        gpu=subprocess.Popen(worker_args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
            text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        protocol['gpu_ready']=receive(gpu)
        for _ in range(3):
            gpu.stdin.write(js(dict(id=counter,warmup=True))+'\n');gpu.stdin.flush();receive(gpu);counter+=1
        protocol['gpu_ready_after_s']=time.perf_counter()-pipeline_started
        if period_process is not None:protocol['period_worker_provenance']=period_process.receive_ready()
        (out/'protocol.json').write_text(js(protocol),'utf-8')
        if args.preview:
            video=cv2.VideoWriter(str(out/'predictions.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.hz,(1280,720))
            if not video.isOpened():raise RuntimeError('Cannot open video writer')
        if not args.optimized_compute:start_reader()
        if args.defer_cyclic_gc:gc.collect();gc.disable()
        if args.audit_gc or args.defer_cyclic_gc:
            def gc_callback(phase,info):
                generation=info['generation']
                if phase=='start':gc_started[generation]=time.perf_counter()
                elif generation in gc_started:
                    gc_events.append(dict(generation=generation,completed_after_s=time.perf_counter()-pipeline_started,
                        duration_ms=(time.perf_counter()-gc_started.pop(generation))*1000,collected=info['collected']))
            gc.callbacks.append(gc_callback)
        epoch=None;last_state_completed_wall=None
        if args.publish_hz:
            from live_publication import TimedPublisher
            from physical_publication import publication,freeze
            publication_stream=(out/'publications.jsonl').open('x',encoding='utf-8')
            def emit_publication(state,wall,index,deadline):
                begin=time.perf_counter();now=args.start+wall-publication_clock.epoch
                value=publication(state,now,deadline,index)
                publication_stream.write(js(value)+'\n')
                publication_records.append((now,value['valid'],value['source_age_s']))
                publication_cost.append((time.perf_counter()-begin)*1000)
            publisher=TimedPublisher(publication_clock,args.duration,args.publish_hz,emit_publication)
            publisher.start()
        with (out/'predictions.jsonl').open('x',encoding='utf-8') as stream:
            while True:
                read_start=time.perf_counter()
                if raw_process is not None:packet=raw_process.read()
                elif args.serial_decode:packet=read_owned()
                else:
                    packet=read_future.result()
                    if packet is not None:read_future=reader_pool.submit(read_owned)
                if packet is None:break
                t,events,producer_decode_ms=packet
                decoded=time.perf_counter();decode_ms=(decoded-read_start)*1000
                if epoch is None:
                    epoch=decoded-(t-args.start);protocol['initial_seek_decode_s']=decode_ms/1000
                    publication_clock.epoch=epoch
                    protocol['first_source_ready_after_s']=decoded-pipeline_started
                if args.paced:
                    remaining=epoch+(t-args.start)-time.perf_counter()
                    if remaining>0:time.sleep(remaining)
                tick=time.perf_counter();cost=dict(nn_deadline_wait_ms=0.)
                a=tick
                if args.prefilter_decode:clean=raw_process.filtered_events
                else:noise.process_events(events,clean_buffer);clean=clean_buffer.numpy()
                last=clean[int(np.searchsorted(clean['t'],round(t*1e6)-20000)):]
                recent_detector_box=body.get('bbox') if body is not None and body.get('detected') else None
                if recent_detector_box is None and roi_memory is not None and t-roi_memory[0]<=.3:recent_detector_box=roi_memory[1]
                small_or_unknown=bool(recent_detector_box is None or max(np.subtract(recent_detector_box[2:],recent_detector_box[:2]))<=192.)
                recent_geometric_support=any(r.get('structure',{}).get('mode','insufficient')!='insufficient' for r in records[-9:])
                known_small_unmodeled=bool(recent_detector_box is not None and small_or_unknown and
                    getattr(structure.model,'beta',None) is None and not recent_geometric_support)
                native_detector_current=bool(args.native_detector_window or (args.small_target_native_window and known_small_unmodeled))
                detector_events=clean if native_detector_current else last
                image_current=False;render_during_localize_ms=0.
                if args.optimized_compute:
                    gray=occupancy_frame(last) if structure.count_current is None else structure.count_current.occupancy(last)
                else:
                    BaseFrameGenerationAlgorithm.generate_frame(last,image);image_current=not native_detector_current
                    gray=np.uint8(image[:,:,0]!=52)*255
                cost['filter_render_ms']=(time.perf_counter()-a)*1000
                a=time.perf_counter();center_hint=None
                if args.rigid_motion and getattr(structure.model,'beta',None) is not None:
                    model=structure.model
                    if t-model.last_update<=.12 and np.linalg.eigvalsh(model.P[np.ix_([0,3],[0,3])]).max()<=36.:
                        center_hint=model.beta[[0,3]]
                seed=None
                if (getattr(args,'inertial_klt',False) or args.wide_motion_search) and coeff is not None and records:
                    delta_t=t-records[-1]['t']
                    if 0.<delta_t<=.08:
                        prior_acceleration=np.asarray(records[-1]['force'].get('acceleration_px_s2',[0.,0.]))
                        seed=delta_t*np.asarray(coeff['velocity'])+.5*delta_t*delta_t*prior_acceleration
                tracked,tracking=track(previous,gray,body,rigid_motion=args.rigid_motion,center_hint=center_hint,
                    spatial_motion=getattr(args,'spatial_weak_motion',False),inertial_seed=seed,
                    seed_backward=getattr(args,'inertial_klt',False) or args.wide_motion_search,rigid_box_center=args.rigid_box_center,body_core_points=args.body_core_points,wide_motion_search=args.wide_motion_search)
                due=len(records)%args.nn_stride==0 or tracked is None or last_nn is None or t-last_nn>.3
                schedule=None
                if args.structure_guided_nn:
                    from live_local_flow import box as qualified_box
                    prior_structure=records[-1]['structure'] if records else {}
                    mandatory=tracked is None or last_nn is None or t-last_nn>.3
                    covariance=np.asarray(tracking.get('displacement_covariance_px2',np.eye(2)*1e9))
                    eligible=bool(not mandatory and qualified_box(tracked) is not None and tracking.get('points',0)>=12 and
                        np.linalg.eigvalsh(covariance).max()<=36. and current_axis_qualified(prior_structure))
                    original_due=due;interval=min(.3,2*args.nn_stride/args.hz)
                    if eligible:
                        requested_at=pending_t if pending_t is not None else last_nn
                        due=bool(t-requested_at>=interval-.5/args.hz)
                    schedule=dict(eligible=eligible,previous_geometry_guidance_only=True,
                        current_KLT_support_checked=True,target_request_interval_s=interval if eligible else args.nn_stride/args.hz,
                        deferred_original_due=bool(original_due and not due),mandatory_refresh=mandatory,
                        last_request_t=pending_t,current_geometry_not_yet_measured_at_this_scheduling_step=True)
                acquisition_schedule=None
                if args.empty_search_stride:
                    from nn_search_schedule import acquisition_due
                    due,acquisition_schedule=acquisition_due(t,tracked,last_body_observed,last_nn_request,args.hz,args.nn_stride,due,source_start=args.start)
                fresh_nn=False;nn_answer_age=None;association_diag=None;recovery_diag=None;weak_roi_diag=None;verifier_diag=None;window_recovery_diag=None;current_gate_diag=None
                recovery_allowed=args.current_nn_on_loss;nn_recovery_schedule=None
                if args.budgeted_nn_recovery and tracked is None:
                    from nn_recovery_budget import decision
                    nn_service_history=[r for r in nn_service_history if t-r[0]<=.3+1e-8]
                    recovery_allowed,nn_recovery_schedule=decision(time.perf_counter(),epoch+(t-args.start)+1/args.hz,1/args.hz,
                        [records[-i]['cost'] for i in range(1,min(len(records),3)+1)],
                        [r[1] for r in nn_service_history[-3:]],pending is not None and not pending.done())
                use_async=bool(args.async_nn and not(recovery_allowed and tracked is None))
                current_small_nn=bool(args.synchronous_small_nn and known_small_unmodeled and (pending is None or pending.done()))
                if current_small_nn:due=True;use_async=False
                if args.synchronous_start and not records:due=True;use_async=False
                if args.synchronous_acquisition and tracked is None:due=True;use_async=False
                discarded_reply=False
                if not use_async and pending is not None:
                    stale=pending.result();pending=None;discarded_reply=True
                    if args.verify_frames and pending_hash!=stale['frame_sha256']:raise RuntimeError('Recovery worker frame mismatch')
                    pending_gray=None;pending_reference=None
                hint=None
                if args.roi_recovery:
                    if tracked is not None and not tracked.get('low_confidence'):
                        hint=tracked['bbox']
                    elif roi_memory is not None and t-roi_memory[0]<=.3:
                        dt_roi=t-roi_memory[0]
                        delta=np.zeros(2) if coeff is None else np.asarray(coeff['velocity'])*dt_roi
                        hint=(roi_memory[1]+np.r_[delta,delta]).tolist()
                weak_continuation_diag=None
                request_extra=dict(recover_roi=args.roi_recovery,roi_hint=hint,
                    verify_weak_roi=getattr(args,'weak_roi_verify',False))
                if args.current_box_gate and tracked is not None and tracking.get('reason')=='current_image_translation':
                    request_extra['current_association_hint']=tracked['bbox']
                if args.detector_window_fallback:
                    recent_box=body.get('bbox') if body is not None and body.get('detected') else None
                    if recent_box is None and roi_memory is not None and t-roi_memory[0]<=.3:recent_box=roi_memory[1]
                    request_extra['native_window_fallback']=bool(recent_box is None or max(np.subtract(recent_box[2:],recent_box[:2]))<=192.)
                if args.weak_body_continuation and weak_semantic_memory is not None and t-weak_semantic_memory[0]<=.3+1e-8:
                    if tracked is not None:
                        weak_hint=tracked['bbox']
                    else:
                        _,observed_t,old_box=weak_semantic_memory
                        displacement=np.zeros(2) if coeff is None else (t-observed_t)*np.asarray(coeff['velocity'])
                        weak_hint=(old_box+np.tile(displacement,2)).tolist()
                    request_extra['weak_continuation_hint']=weak_hint
                if verifier_model:
                    current_hint=None
                    if tracked is not None and tracking.get('reason')=='current_image_translation' and tracking.get('points',0)>=8:
                        current_hint=tracked['bbox']
                    elif roi_memory is not None and t-roi_memory[0]<=.3:
                        shift=np.zeros(2) if coeff is None else np.asarray(coeff['velocity'])*(t-roi_memory[0])
                        current_hint=(roi_memory[1]+np.r_[shift,shift]).tolist()
                    request_extra['weak_verifier_hint']=current_hint
                if use_async:
                    body=tracked if tracked is not None else dict(detected=False)
                    if args.deadline_nn and args.paced and pending is not None and not pending.done():
                        from forecast_runtime import nn_reply_wait_budget
                        budget=nn_reply_wait_budget(time.perf_counter(),epoch+(t-args.start)+1/args.hz,1/args.hz,
                            [records[-i]['cost'] for i in range(1,min(len(records),3)+1)])
                        if budget>0.:
                            wait_start=time.perf_counter()
                            try:pending.result(timeout=budget)
                            except FutureTimeout:pass
                            cost['nn_deadline_wait_ms']=(time.perf_counter()-wait_start)*1000
                    if pending is not None and pending.done():
                        answer=pending.result();pending=None;nn_answer_age=t-pending_t
                        if args.verify_frames and pending_hash!=answer['frame_sha256']:raise RuntimeError('Async worker frame mismatch')
                        boxes=answer.get('associated_boxes',[]) if args.body_association else answer['boxes']
                        association_diag=answer.get('association')
                        recovery_diag=answer.get('roi_recovery')
                        weak_roi_diag=answer.get('weak_roi_verification')
                        verifier_diag=answer.get('weak_verifier')
                        weak_continuation_diag=answer.get('weak_body_continuation')
                        window_recovery_diag=answer.get('native_window_recovery')
                        current_gate_diag=answer.get('current_box_gate')
                        if boxes:
                            best=max(boxes,key=lambda b:b['confidence']);b=np.asarray(best['bbox'])
                            measured=dict(best,detected=True,center=((b[:2]+b[2:])/2).tolist(),
                                measurement_source='NN_at_earlier_time',low_confidence=best['confidence']<.45)
                            # Reuse the already measured current KLT trajectory
                            # rather than extracting/tracking the same frame twice.
                            if tracked is not None and pending_reference is not None and nn_answer_age<=.12:
                                delta=np.asarray(tracked['center'])-pending_reference
                                projected=dict(measured,bbox=(b+np.r_[delta,delta]).tolist(),
                                    center=(np.asarray(measured['center'])+delta).tolist())
                                transfer=dict(reason='reused_current_KLT_translation_since_NN_request',delta=delta.tolist())
                            else:projected,transfer=track(pending_gray,gray,measured)
                            if projected is not None and nn_answer_age<=.16:
                                body=dict(projected,measurement_source='current_KLT_from_delayed_NN',nn_observed_t=pending_t)
                                last_nn=pending_t;fresh_nn=True;tracking['delayed_NN_transfer']=transfer
                        pending_gray=None
                    if pending is None and due:
                        if not image_current:
                            render_start=time.perf_counter();BaseFrameGenerationAlgorithm.generate_frame(detector_events,image)
                            render_during_localize_ms+=(time.perf_counter()-render_start)*1000;image_current=True
                        np.copyto(shared_image,image)
                        if request_extra.get('native_window_fallback'):
                            render_start=time.perf_counter();BaseFrameGenerationAlgorithm.generate_frame(clean,shared_native_image)
                            render_during_localize_ms+=(time.perf_counter()-render_start)*1000
                            if args.verify_frames:request_extra['native_frame_sha256']=hashlib.sha256(shared_native_image.tobytes()).hexdigest()
                        pending_hash=hashlib.sha256(image.tobytes()).hexdigest() if args.verify_frames else None
                        gpu.stdin.write(js(dict(id=counter,t=t,verify_frame=args.verify_frames,**request_extra))+'\n');gpu.stdin.flush()
                        counter+=1;nn_count+=1;pending_t=t;pending_gray=gray.copy();last_nn_request=t
                        pending_reference=np.asarray(body['center']).copy() if body.get('detected') else None
                        pending=pool.submit(receive,gpu)
                elif due:
                    current_NN_started=time.perf_counter()
                    if not image_current:
                        render_start=time.perf_counter();BaseFrameGenerationAlgorithm.generate_frame(detector_events,image)
                        render_during_localize_ms+=(time.perf_counter()-render_start)*1000;image_current=True
                    np.copyto(shared_image,image)
                    if request_extra.get('native_window_fallback'):
                        render_start=time.perf_counter();BaseFrameGenerationAlgorithm.generate_frame(clean,shared_native_image)
                        render_during_localize_ms+=(time.perf_counter()-render_start)*1000
                        if args.verify_frames:request_extra['native_frame_sha256']=hashlib.sha256(shared_native_image.tobytes()).hexdigest()
                    gpu.stdin.write(js(dict(id=counter,t=t,verify_frame=args.verify_frames,**request_extra))+'\n');gpu.stdin.flush()
                    answer=receive(gpu);counter+=1;nn_count+=1;fresh_nn=True;last_nn_request=t
                    nn_service_history.append((t,time.perf_counter()-current_NN_started))
                    nn_service_history=nn_service_history[-3:]
                    if args.verify_frames and hashlib.sha256(image.tobytes()).hexdigest()!=answer['frame_sha256']:
                        raise RuntimeError('Worker frame mismatch')
                    boxes=answer.get('associated_boxes',[]) if args.body_association else answer['boxes']
                    association_diag=answer.get('association')
                    recovery_diag=answer.get('roi_recovery')
                    weak_roi_diag=answer.get('weak_roi_verification')
                    verifier_diag=answer.get('weak_verifier')
                    weak_continuation_diag=answer.get('weak_body_continuation')
                    window_recovery_diag=answer.get('native_window_recovery')
                    current_gate_diag=answer.get('current_box_gate')
                    if boxes:
                        best=max(boxes,key=lambda b:b['confidence']);b=np.asarray(best['bbox'])
                        body=dict(best,detected=True,center=((b[:2]+b[2:])/2).tolist(),
                            measurement_source='current_NN_zoom_recovery' if best.get('roi_recovered') else 'current_NN',low_confidence=best['confidence']<.45)
                        last_nn=t
                    elif tracked is not None and last_nn is not None and t-last_nn<=.16:body=tracked
                    else:body=dict(detected=False)
                else:body=tracked if tracked is not None else dict(detected=False)
                cost['localize_ms']=(time.perf_counter()-a)*1000-render_during_localize_ms
                cost['filter_render_ms']+=render_during_localize_ms
                if body.get('detected'):last_body_observed=t
                if fresh_nn and body.get('detected') and body.get('confidence',0)>=.25 and not body.get('weak_continuation'):
                    weak_semantic_memory=(body.get('nn_observed_t',t),t,np.asarray(body['bbox']).copy())
                if body.get('detected') and not body.get('low_confidence'):
                    roi_memory=(t,np.asarray(body['bbox']).copy())
                previous=gray
                a=time.perf_counter()
                # Frequency uses ALL original timestamps in the selected expanded ROI,
                # including the13ms not rendered in the20ms shape window.
                selected=period_roi.select(events,roi_bounds(body['bbox'],padding=.35)) if body.get('detected') else events[:0]
                select_ms=(time.perf_counter()-a)*1000
                velocity=[0.,0.] if coeff is None else coeff['velocity']
                geometry_deferred=bool(args.unresolved_geometry_stride>1 and len(records)>=9 and
                    body.get('detected') and max(np.subtract(body['bbox'][2:],body['bbox'][:2]))<=192. and
                    getattr(structure.model,'beta',None) is None and
                    all(r.get('structure',{}).get('mode')=='insufficient' and not r.get('structure',{}).get('observed_count') for r in records[-9:]) and
                    len(records)%args.unresolved_geometry_stride!=0)
                overlap_period=args.parallel_period and (not args.adaptive_period or not structure.last_side_qualified)
                if overlap_period:
                    frequency_future=period_pool.submit(measure_period,selected,t)
                    a=time.perf_counter();s,_,geom,density=structure.step(selected,t,body,None,
                        velocity,frequency_pixels_unique=True,defer_rates=True,defer_geometry=geometry_deferred)
                    shape_ms=(time.perf_counter()-a)*1000
                    wait_started=time.perf_counter()
                    q,pooled_input,all_frequencies_in_range,engine_ms=frequency_future.result()
                    cost['period_wait_ms']=(time.perf_counter()-wait_started)*1000
                    a=time.perf_counter()
                    rates=rotor_rates(q,geom.get('rotors',[]),t,velocity,1280,unique_pixels=True) if geom.get('rotors') else []
                    cost['structure_ms']=shape_ms+(time.perf_counter()-a)*1000
                    cost['period_engine_ms']=select_ms+engine_ms
                else:
                    q,pooled_input,all_frequencies_in_range,engine_ms=measure_period(selected,t)
                    cost['period_engine_ms']=select_ms+engine_ms;cost['period_wait_ms']=0.
                    a=time.perf_counter();s,rates,geom,density=structure.step(selected,t,body,q,
                        velocity,frequency_pixels_unique=True,defer_geometry=geometry_deferred)
                    cost['structure_ms']=(time.perf_counter()-a)*1000
                a=time.perf_counter()
                # Fixed geometry/body gauge from existing bridge, without another filter.
                adjusted,center_diag=centering.step(body,s,geom.get('rotors',[]))
                motion.kf.noise_scale=center_diag.get('noise_scale',1.)
                coeff,valid,motion_diag=motion.step(t,adjusted)
                image_motion_diag=None
                if image_motion is not None:
                    coeff,image_motion_diag=image_motion.step(t,coeff,motion.kf.P[2:4,2:4] if coeff else np.eye(2),adjusted,tracking,structure=s)
                base,cv=scale_predictions(coeff) if coeff is not None and args.with_kinematic_control else ({},{})
                pooled=native79.summarize(pooled_input,body.get('bbox'),t,1280,720,unique_pixels=all_frequencies_in_range)
                period_support_diag=None
                if temporal_period is not None:
                    pooled,period_support_diag=temporal_period.step(t,q,body.get('bbox') if body.get('detected') else None,velocity,pooled)
                aligned_period_diag=None
                if aligned_period is not None:
                    current_translation=bool((image_motion_diag or {}).get('observation',{}).get('valid'))
                    if args.body_period_stream:
                        pooled,aligned_period_diag=aligned_period.step(t,selected,body,tracking,pooled,current_translation)
                    else:pooled,aligned_period_diag=aligned_period.step(t,selected,body,tracking,pooled,current_axis_qualified(s),current_translation)
                u,quality,period_diag=period_selector.step(t,s,geom.get('rotors',[]),rates,pooled)
                amplitude_diag=None
                if amplitude_bridge is not None:
                    u,quality,amplitude_diag=amplitude_bridge.step(t,s,u,quality,period_diag)
                    force.input_learning_current=amplitude_diag['learning_input_current']
                f=period_diag['frequency_hz'];proxy_source=period_diag['source'];period_reason=period_diag['reason']
                force_reference=body if args.delayed_disturbance_control else adjusted
                position=force_reference.get('center') if force_reference.get('detected') else None
                if args.flow_force_balance:
                    force.observe_flow(image_motion_diag['observation'],image_motion_diag['acceleration'].get('applied',False))
                if args.tangent_disturbance:force.observe_axis_motion(s,image_motion_diag)
                physical,force_diag=force.step(t,coeff,position,u,quality)
                normal_forecast_diag=None
                if normal_preview is not None:
                    trend=normal_preview.observe(t,s,coeff,image_motion_diag)
                    physical,normal_forecast_diag=normal_preview.correct(physical,force_diag,trend)
                memory_diag=thrust_trend_diag=None
                if force_memory is not None:
                    physical,memory_diag=force_memory.step(t,coeff,position,physical,force_diag,normal_forecast_diag,valid,
                        bool((image_motion_diag or {}).get('acceleration',{}).get('applied')),current_axis_qualified(s))
                if thrust_trend is not None:
                    trend=thrust_trend.observe(t,period_diag,pooled,amplitude_diag,s.get('geometry_epoch'))
                    physical,thrust_trend_diag=thrust_trend.correct(physical,force_diag,normal_forecast_diag,trend,memory_diag)
                if image_motion_diag is not None:
                    inferred_acceleration=bool(image_motion_diag['acceleration'].get('applied'))
                    force_diag['event_motion_net_acceleration_available']=inferred_acceleration
                    if inferred_acceleration and not force_diag['active'] and not force_diag['short_gap_acceleration_held']:
                        force_diag['acceleration_source']='qualified_event_flow_total_acceleration'
                position_net_force_diag=None
                if position_net_force is not None:
                    physical,force_diag,position_net_force_diag=position_net_force.step(dict(t=t,body=body,valid=valid,
                        coefficient=coeff,force=force_diag,image_motion=image_motion_diag,physical=physical,
                        adaptive_force_memory=memory_diag))
                cost['motion_force_ms']=(time.perf_counter()-a)*1000
                if period_process is not None:cost['period_worker_engine_ms']=period_process.last_worker_engine_ms
                if raw_process is not None:
                    cost.update(producer_native_decode_ms=raw_process.last_native_decode_ms,producer_activity_filter_ms=raw_process.last_activity_filter_ms)
                    cost['producer_cyclic_GC_ms']=sum(event['duration_ms'] for event in raw_process.last_gc_events)
                    cost.update(producer_native_decode_CPU_ms=raw_process.last_native_decode_CPU_ms,
                        producer_native_decode_wait_proxy_ms=raw_process.last_native_decode_wait_proxy_ms)
                done=time.perf_counter();compute_ms=(done-tick)*1000
                row=dict(t=t,body=body,structure=s,geometry=geom,rates=rates,period=pooled,
                    period_reason=period_reason,period_proxy_source=proxy_source,
                    squared_frequency_proxy=None if f is None else f*f,
                    relative_frequency_squared_proxy=period_diag['relative_f2'],period_selection=period_diag,
                    coefficient=coeff,valid=valid,baseline=base,cv=cv,physical=physical,
                    force=force_diag,center_bridge=center_diag,tracking=tracking,motion=motion_diag,
                    image_motion=image_motion_diag,
                    axis_amplitude_bridge=amplitude_diag,
                    temporal_period_support=period_support_diag,normal_turn_forecast=normal_forecast_diag,
                    adaptive_force_memory=memory_diag,thrust_trend_forecast=thrust_trend_diag,
                    aligned_period_recovery=aligned_period_diag,
                    period_execution='overlapped_current_process' if overlap_period and period_process is not None else 'overlapped_current_thread' if overlap_period else 'current_serial',
                    nn_structure_schedule=schedule,
                    nn_acquisition_schedule=acquisition_schedule,
                    nn_recovery_schedule=nn_recovery_schedule,
                    force_observed_position=position,
                    nn_fresh=fresh_nn,nn_answer_age_s=nn_answer_age,nn_pending=pending is not None,
                    association=association_diag,
                    weak_roi_verification=weak_roi_diag,
                    weak_verifier=verifier_diag,
                    weak_body_continuation=weak_continuation_diag,
                    position_net_force=position_net_force_diag,
                    nn_execution_mode='asynchronous' if use_async else 'current_synchronous',
                    nn_native_window_recovery=window_recovery_diag,
                    nn_current_box_gate=current_gate_diag,
                    detector_native_window_current=native_detector_current,
                    current_small_target_NN=current_small_nn,
                    stale_NN_reply_discarded_for_current_recovery=discarded_reply,nn_roi_recovery=recovery_diag,
                    raw_events=len(events),frequency_input_events=len(selected),
                    decoder_packet_overflow=False if raw_process is None else raw_process.packet_overflow,
                    geometry_input_events=len(selected),
                    max_event_t=None if not len(events) else int(events['t'].max()),
                    cost=dict(cost,decode_ms=decode_ms,producer_decode_ms=producer_decode_ms,compute_ms=compute_ms,
                        service_ms=compute_ms+(0. if first_decode else decode_ms),
                        completion_lag_ms=(done-epoch-(t-args.start))*1000),first_decode=first_decode)
                first_decode=False
                if len(records)==0:protocol['first_result_after_s']=done-pipeline_started
                a=time.perf_counter()
                if getattr(args,'compact_output',False):
                    from forecast_output import compact_record
                    stream.write(js(compact_record(row))+'\n')
                else:stream.write(js(row)+'\n')
                if video is not None:
                    if not image_current or native_detector_current:BaseFrameGenerationAlgorithm.generate_frame(last,image)
                    display=image.copy()
                    if body.get('detected'):
                        b=np.rint(body['bbox']).astype(int);cv2.rectangle(display,tuple(b[:2]),tuple(b[2:]),(0,220,0),1)
                    for r in s['observed_centers']:cv2.circle(display,tuple(np.rint(r).astype(int)),3,(0,0,255),-1)
                    for pred,color in [(base,(255,140,0)),(physical,(0,220,255))]:
                        if valid and pred:cv2.polylines(display,[np.rint([coeff['position']]+[pred[str(H)] for H in (.1,.2,.4)]).astype(np.int32)],False,color,2)
                    axis=s.get('axis_xy')
                    if body.get('detected') and axis is not None:
                        p=np.asarray(body['center']);cv2.arrowedLine(display,tuple(np.rint(p).astype(int)),tuple(np.rint(p+60*np.asarray(axis)).astype(int)),(0,0,255),2)
                    cv2.putText(display,f'{t:.3f}s {s["mode"]} NN={due} force={force_diag["active"]} {compute_ms:.1f}ms',(15,28),0,.65,(0,180,255),1)
                    video.write(display)
                row['cost']['output_ms']=(time.perf_counter()-a)*1000
                row['cost']['total_service_ms']=row['cost']['service_ms']+row['cost']['output_ms']
                records.append(row)
                if args.lean_history:
                    # Include columnar bookkeeping in this option's output cost.
                    row['cost']['output_ms']=(time.perf_counter()-a)*1000
                    row['cost']['total_service_ms']=row['cost']['service_ms']+row['cost']['output_ms']
                    records.finish_current_costs()
                last_state_completed_wall=time.perf_counter()
                if publisher is not None:publisher.update(freeze(row,last_body_observed))
        if publisher is not None:publisher.finish()
        end=time.perf_counter()
        if gc_callback is not None:gc.callbacks.remove(gc_callback);gc_callback=None
        if args.defer_cyclic_gc and gc_was_enabled:gc.enable()
        # Output-write cost is known only after its prediction line is written.
        # Export the completed per-tick costs separately, after the timed run;
        # do not pretend those late fields exist in the original JSONL line.
        with (out/'timings.jsonl').open('w',encoding='utf-8',newline='\n') as timing_stream:
            for r in records:timing_stream.write(js(dict(t=r['t'],**r['cost']))+'\n')
        samples=records[1:];cost_keys=['decode_ms','producer_decode_ms','compute_ms','service_ms','total_service_ms',
            'filter_render_ms','localize_ms','period_engine_ms','period_wait_ms','structure_ms','motion_force_ms','output_ms','completion_lag_ms']
        timing={key:dict(median=float(np.median([r['cost'][key] for r in samples])),
            p95=float(np.percentile([r['cost'][key] for r in samples],95)),mean=float(np.mean([r['cost'][key] for r in samples]))) for key in cost_keys} if samples else {}
        observed=sum(r['body'].get('detected',False) for r in records)
        service=np.array([r['cost']['total_service_ms'] for r in samples])
        summary=dict(frames=len(records),fresh_image_observations=observed,nn_frames=nn_count,
            valid_forecasts=sum(r['valid'] for r in records),modes=dict(Counter(r['structure']['mode'] for r in records)),
            current_force_inputs=sum(r['force']['active'] and not (r.get('axis_amplitude_bridge') or {}).get('applied',False) for r in records),
            recent_amplitude_current_axis_inputs=sum((r.get('axis_amplitude_bridge') or {}).get('applied',False) for r in records),
            gain_learning_updates=sum(r['force']['gain_learned'] for r in records),
            final_gain=force.gain,timing_ms=timing,
            steady_capacity_fps=None if not len(service) else float(1000/service.mean()),
            service_within_target_fraction=None if not len(service) else float(np.mean(service<=1000/args.hz)),
            paced=args.paced,paced_fresh_observation_rate=None if not args.paced else observed/max(last_state_completed_wall-epoch,1e-9),
            paced_observation_rate_including_eof=None if not args.paced else observed/max(end-epoch,1e-9),
            eof_wait_s=0. if last_state_completed_wall is None else max(0.,end-last_state_completed_wall),
            paced_deadline_fraction=None if not args.paced else float(np.mean([r['cost']['completion_lag_ms']<=1000/args.hz for r in samples])),
            initial_seek_decode_s=protocol['initial_seek_decode_s'],raw_unchanged=raw.stat().st_mtime_ns==stats.st_mtime_ns and raw.stat().st_size==stats.st_size,
            first_result_after_s=protocol['first_result_after_s'],
            decoder_overflow_packets=0 if raw_process is None else raw_process.overflow_packets,
            decoder_max_packet_events=None if raw_process is None else raw_process.max_packet_events,
            decoder_shared_memory_bytes=None if raw_process is None else raw_process.allocated_shared_bytes,
            startup_note='initial_seek_decode_s is consumer blocking wait only; first_result_after_s includes overlapped model/decoder startup',
            target_achieved=False,accuracy='not scored yet; annotations never read by this runner')
        summary['period_worker_overflow_packets']=0 if period_process is None else period_process.overflow
        if publisher is not None:
            intervals=np.diff([r[0] for r in publication_records])
            summary['publication']=dict(target_hz=args.publish_hz,outputs=publisher.published,
                valid_outputs=sum(r[1] for r in publication_records),missed_deadlines=publisher.missed,
                actual_mean_hz=None if not len(intervals) else float(1/intervals.mean()),
                interval_p95_ms=None if not len(intervals) else float(np.percentile(intervals,95)*1000),
                max_source_age_s=max((r[2] for r in publication_records if r[1]),default=None),
                callback_p95_ms=float(np.percentile(publication_cost,95)) if publication_cost else None,
                source_observation_rate_measured_separately=True,publisher_invents_no_new_image_observations=True,
                stale_invalid_outputs_not_counted_as_predictions=True,
                horizon_reference='actual publication time, with same physical rollout advanced by source age')
        summary['history_storage']=dict(full_diagnostic_history_retained=not args.lean_history,
            exact_numeric_timing_history_retained=True,
            numeric_storage_bytes=getattr(records,'numeric_storage_bytes',None),
            full_current_record_retained=True,constant_memory_unbounded_stream=False)
        summary['structure_guided_NN_deferred_ticks']=sum((r.get('nn_structure_schedule') or {}).get('deferred_original_due',False) for r in records)
        summary['cyclic_GC']=dict(deferred=args.defer_cyclic_gc,events=gc_events,
            audited=args.audit_gc or args.defer_cyclic_gc,
            measured_stream_scope_only=True,reference_count_reclamation_remains_enabled=True,
            original_enabled_state_restored=gc.isenabled()==gc_was_enabled)
        protocol['actual_activity_filter']=getattr(noise,'provenance',None) if raw_process is None or not args.prefilter_decode else raw_process.filter_provenance
        summary['huber_solver']=dict(huber_provenance,counters=huber_counters())
        summary['conic_residual_backend']=None if conic_residual is None else dict(conic_residual.provenance,calls=conic_residual.calls)
        summary['decoder_cyclic_GC']=None if raw_process is None else dict(deferred=args.defer_decoder_gc,events=raw_process.gc_events,reference_count_reclamation_remains_enabled=True,finite_stream_scope=True)
        (out/'summary.json').write_text(js(summary),'utf-8')
        protocol['finished']=True;(out/'protocol.json').write_text(js(protocol),'utf-8')
        print(js(summary),flush=True)
    finally:
        publication_clock.done=True
        if publisher is not None:publisher.abort()
        if publication_stream is not None:publication_stream.close()
        if gc_callback is not None:gc.callbacks.remove(gc_callback)
        if gc_was_enabled:gc.enable()
        elif not gc_was_enabled:gc.disable()
        if video is not None:video.release()
        if pending is not None:
            try:pending.result(timeout=15)
            except Exception:pass
        pool.shutdown(wait=True)
        period_pool.shutdown(wait=True)
        reader_pool.shutdown(wait=True)
        if period_process is not None:period_process.close()
        if raw_process is not None:raw_process.close()
        if gpu is not None:
            if gpu.poll() is None:
                try:gpu.stdin.write(js(dict(stop=True))+'\n');gpu.stdin.flush();gpu.wait(timeout=15)
                except Exception:gpu.terminate();gpu.wait(timeout=10)
            gpu.stdin.close();gpu.stdout.close()
        if log is not None:log.close()
        if memory is not None:image=None;shared_image=None;shared_native_image=None;memory.close();memory.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--rotor-only',action='store_true',help='Explicit native rotor-only front/geometry/forecast route; bypasses body NN and model loading, preserves original baseline. Experimental source-time replay, not realtime30 certification.')
    p.add_argument('--rotor-legacy-visibility',action='store_true',help='Rotor-only control: old visibility/concavity path; NN baseline always retains its old path')
    p.add_argument('--rotor-allow-invalid-motion-fallback',action='store_true',help='Rotor-only legacy control: permit rejected displacement to re-enter as an absolute center')
    p.add_argument('--rotor-gap-prediction',choices=['legacy','coherent','short'],help='Rotor-only relative-observer missing-position prior; default short, explicit legacy control available')
    p.add_argument('--rotor-unknown-force-model',choices=['free','plane','plane-separated','plane-adaptive'],default='free',help='Rotor-only relative/fixed: plane, plane-separated, plane-adaptive matured-error credit, or176 free disturbance')
    p.add_argument('--rotor-reference-mode',choices=['legacy','fixed'],help='Rotor-only relative route: fixed reference plus uncertain layout offset, or legacy absolute centre replacement')
    p.add_argument('--rotor-legacy-force',action='store_true',help='Rotor-only control: keep the original delayed residual force model')
    p.add_argument('--rotor-legacy-decomposition',action='store_true',help='Rotor-only stage161 control: previous force partition')
    p.add_argument('--rotor-no-phase',action='store_true',help='Rotor-only stage162 control: disable spatial phase arbitration')
    p.add_argument('--rotor-native-gpu',action='store_true',help='Rotor-only fused native CUDA preprocessing and spectrum')
    p.add_argument('--rotor-fast-geometry',action='store_true',help='Rotor-only existing optimized geometry and histogram')
    p.add_argument('--rotor-prefetch',action='store_true',help='Rotor-only bounded one-packet decode/GPU overlap')
    p.add_argument('--rotor-prefetch-depth',type=int,default=1,help='Bounded1..4 stateless feature packets')
    p.add_argument('--rotor-local-phase',action='store_true',help='Rotor-only local temporal template support for side views')
    p.add_argument('--rotor-track-local-source',action='store_true',help='Rotor-only experimental continuous local-period sources; requires --rotor-local-phase')
    p.add_argument('--rotor-memory-cache',action='store_true',help='Rotor-only bounded saved-event memory replay')
    p.add_argument('--rotor-read-chunk',type=int,default=100000,help='Rotor-only native RAW decoder batching')
    p.add_argument('--rotor-read-buffers',type=int,default=3,help='Rotor-only native RAW read buffer count')
    p.add_argument('--rotor-subpixel-motion',action='store_true',help='Rotor-only current correlation peak subpixel refinement')
    p.add_argument('--rotor-motion-prior',action='store_true',help='Rotor-only elapsed-time-aware search around past current-image displacement')
    p.add_argument('--rotor-plane-motion',action='store_true',help='Current shared filtered-rotor affine/side-line transform')
    p.add_argument('--rotor-coarse-motion',action='store_true',help='Coarse discovery, native correlation refinement and boundary fallback')
    p.add_argument('--rotor-entry-init',action='store_true',help='Initialize velocity after partial entry using first two complete centers')
    p.add_argument('--rotor-regional-seeds',action='store_true',help='Collective temporal evidence for weak individual periodic pixels')
    p.add_argument('--rotor-remote-front',action='store_true',help='Materialize native feature maps in the service process to reduce GIL contention')
    p.add_argument('--rotor-reader-ready',action='store_true',help='Prepare RAW decoder before representative GPU warmup')
    p.add_argument('--rotor-max-backlog-ms',type=float,default=0.,help='Rotor-only paced source backlog budget; zero retains every packet')
    p.add_argument('--rotor-relative-observation',action='store_true',help='Rotor-only displacement observation with cloned-position covariance')
    p.add_argument('--rotor-relock-support-guard',action='store_true',help='Rotor-only small-fragment reacquisition is not a complete center fix')
    p.add_argument('--rotor-border-motion',action='store_true',help='Rotor-only supported interior matching near the sensor boundary')
    p.add_argument('--rotor-retain-host-events',action='store_true',help='Retain redundant CPU native event copy as transport control')
    p.add_argument('--rotor-reader',choices=['rolling','iterator','process-rolling','process-iterator'],default='rolling',help='Rotor-only optional existing two-slot decoder process')
    p.add_argument('--rotor-gpu-front',action='store_true',help='Rotor-only sparse CUDA front; requires --gpu-python')
    p.add_argument('--no-preview',action='store_true',help='Rotor-only: skip diagnostic image/video export')
    p.add_argument('--start',type=float,default=110.);p.add_argument('--duration',type=float,default=3.)
    p.add_argument('--hz',type=float,default=30.);p.add_argument('--nn-stride',type=int,default=3)
    p.add_argument('--gpu-python');p.add_argument('--paced',action='store_true')
    p.add_argument('--publish-hz',type=float,default=0.,help='Optional independent wall-clock output rate for target-candidate; retime the same physics, reject old states, log invalid gaps; does not increase current-image observation Hz')
    p.add_argument('--realtime30',action='store_true',help='Working30Hz profile: target-candidate, supported rigid center translation, separate period process and independently retimed30Hz publications; accuracy target is not certified')
    p.add_argument('--recall30',action='store_true',help='Recall-oriented30Hz candidate: realtime30 plus native small-target NN windows and current per-tick small-target NN; retains the original realtime30 baseline')
    p.add_argument('--recall90',action='store_true',help='Small-target candidate: recall30, one current first-frame acquisition and one joint Singer state; requires explicit compatible detector model/hash, not a certified90-percent claim')
    p.add_argument('--detector-model',help='Explicit local experimental ONNX; original model remains the default')
    p.add_argument('--detector-sha256',help='Required exact SHA256 for an explicit experimental ONNX')
    p.add_argument('--weak-verifier-model',help='Experimental isolated small-target ONNX verifier; original detector keeps primary localization')
    p.add_argument('--weak-verifier-sha256',help='Required SHA256 for the optional weak verifier')
    p.add_argument('--weak-verifier-max-side',type=float,help='Override the hash-bound training-domain box-size limit; default uses adjacent verifier_domain.json or192px')
    p.add_argument('--async-nn',action='store_true')
    p.add_argument('--native-detector-window',action='store_true',help='Use the complete current native event packet for NN only; geometry and optical flow keep the original20ms window; no extra buffering or future events')
    p.add_argument('--detector-window-fallback',action='store_true',help='Preserve successful20ms detections; after an empty NN result retry current native packet for unknown/small targets, using the same model/threshold; one extra bounded image buffer')
    p.add_argument('--small-target-native-window',action='store_true',help='Use the current complete native NN window for recent<=192px targets without a confirmed five-circle model; cold search and modeled/large targets retain20ms; one forward pass')
    p.add_argument('--synchronous-small-nn',action='store_true',help='Request one current NN per source tick for known small unmodeled targets when no older GPU request is running; cold search and modeled/large targets retain async cadence')
    p.add_argument('--singer-state',action='store_true',help='Experimental single joint position/velocity/net-acceleration Singer state, consistent with the existing force rollout; no parallel trajectory branch')
    p.add_argument('--synchronous-start',action='store_true',help='Use one current synchronous NN at the first real source tick, retaining actual latency; later cadence is unchanged')
    p.add_argument('--synchronous-acquisition',action='store_true',help='When current image tracking is unavailable, obtain a current NN rather than repeatedly discarding untransferable delayed boxes; actual blocking time remains measured')
    p.add_argument('--current-box-gate',action='store_true',help='Gate current NN target identity by recent actual NN boxes/current image translation; rejected frames remain missing, no predicted measurement')
    p.add_argument('--unresolved-geometry-stride',type=int,default=None,choices=[1,3],help='10Hz geometry rescan only after9 completely insufficient small-target updates; default1, recall90 defaults3; current NN/physics/native period feed are retained')
    p.add_argument('--optimized-compute',action='store_true',help='Equivalent geometry work reduction, lazy native color render and overlapped decoder/model startup')
    p.add_argument('--reference-compute',dest='optimized_compute',action='store_false',help='Retain the original geometry/render/startup compute path as a control')
    p.set_defaults(optimized_compute=True)
    p.add_argument('--prefilter-decode',action='store_true',help='Overlap unchanged SDK activity filter with geometry via two additional bounded shared buffers')
    p.add_argument('--sdk-avx2',action='store_true',help='Optional locally built AVX2 instantiation of unchanged installed SDK activity filter; SDK license/local binary required; implies JIT dependency resolver')
    p.add_argument('--parallel-period',action='store_true',help='Overlap independent current geometry and SDK period calculations; rates join before force prediction')
    p.add_argument('--period-all-pixels',action='store_true',help='Experimental period observer uses every ROI pixel; native timestamps retained; does not change geometry/detector input')
    p.add_argument('--temporal-period-support',action='store_true',help='Weak current periods may use distinct-pixel support from <=100ms history; no rotor ID; implies all period pixels')
    p.add_argument('--normal-turn-forecast',action='store_true',help='Headless plane-normal trend rotates only the known thrust contribution in future rollout; no rotor IDs')
    p.add_argument('--no-normal-turn-forecast',action='store_true',help='Control: disable the new disk-normal rollout even with target-candidate')
    p.add_argument('--adaptive-force-memory',action='store_true',help='Learn qualified physical net-force persistence from matured own observations, with no extra trajectory filter')
    p.add_argument('--shorter-force-memory',action='store_true',help='Experimental two-sided force-memory calibration from matured own positions; may shorten as well as lengthen the net-acceleration prior')
    p.add_argument('--no-adaptive-force-memory',action='store_true',help='Control: preserve fixed Singer memory in the target candidate')
    p.add_argument('--memory-requires-current-period',action='store_true',help='Control: omit calibrated net-force persistence during missing periods even with current qualified motion and plane axis')
    p.add_argument('--thrust-trend-forecast',action='store_true',help='Experimental current same-source frequency-squared trend in known-input rollout; does not scale the unknown force')
    p.add_argument('--aligned-period-recovery',action='store_true',help='Experimental weak-period retry at 10Hz using <=160ms measured body-translation-aligned ROI history; native periods and geometric pixels retained')
    p.add_argument('--aligned-period-filter',choices=['strict7','fast3'],default='strict7',help='Existing SDK stable7-cycle or short3-cycle configuration for the optional aligned channel')
    p.add_argument('--body-period-stream',action='store_true',help='Experimental persistent fixed body-translation frequency observer; each event processed once, no historical replay')
    p.add_argument('--process-period',action='store_true',help='Run the unchanged SDK period observer in a bounded separate CPU process; implies parallel period')
    p.add_argument('--adaptive-period',action='store_true',help='With parallel period, keep cheap qualified side-axis ticks serial and overlap more expensive conic discovery; computations join at same source tick')
    p.add_argument('--side-axis-budget',action='store_true',help='Current qualified side axis each tick; full conic rediscovery every3ticks, with rotor count unresolved between')
    p.add_argument('--responsive-force',action='store_true',help='Recent causal disturbance correction and Singer conditional mean net-acceleration rollout')
    p.add_argument('--goal-profile',action='store_true',help='Candidate50px/30FPS profile: asyncNN + prefilter decode + side axis budget + responsive force; not a performance guarantee')
    p.add_argument('--target-candidate',action='store_true',help='20261029 candidate: five-circle geometry, headless normal, curved current motion, evidence gain, qualified endpoint force balance and matured force memory; native kernels and bounded scheduling')
    p.add_argument('--body-association',action='store_true',help='Reuse installed ByteTrack identity association; return only current NMS boxes, including matched low-score boxes')
    p.add_argument('--native-histogram',action='store_true',help='Exact-count C++ histogram directly in float32, with large-count fallback; no event sampling or saturation')
    p.add_argument('--jit-counts',action='store_true',help='Optional fused native CPU ROI/count kernel; requires compatible NumPy/Numba and times startup compilation')
    p.add_argument('--current-nn-on-loss',action='store_true',help='Use current synchronous NN when current KLT support fails; retain async when current image tracking succeeds')
    p.add_argument('--roi-recovery',action='store_true',help='Reuse current-frame zoom detector after full-frame miss, guided only by a recent current box/past velocity; implies current NN on loss')
    p.add_argument('--weak-roi-verify',action='store_true',help='Experimental: verify a weak <=192px current detection using one existing spatially matched crop; no temporal confidence promotion')
    p.add_argument('--weak-motion-support',action='store_true',help='Experimental: qualify current coherent small-target motion after three real NN anchors; body/rotor confidence stays unchanged; requires rigid motion')
    p.add_argument('--no-weak-motion-support',action='store_true',help='Control: retain the earlier motion qualification for weak small body detections')
    p.add_argument('--spatial-weak-motion',action='store_true',help='Experimental: reuse spatial leave-group covariance and center hull for 8-11 well-distributed existing tracks; implies weak motion support')
    p.add_argument('--inertial-klt',action='store_true',help='Experimental: existing previous physical p/v/force initializes OpenCV LK search only; current bidirectional image tests remain necessary')
    p.add_argument('--visible-motion-support',action='store_true',help='Experimental: current rigid motion may use a >=90%% visible body box; strict rotor/FOV geometry unchanged')
    p.add_argument('--jit-conics',action='store_true',help='Fuse the same finite-difference conic Jacobian; implies optional JIT counts')
    p.add_argument('--reuse-image-motion',action='store_true',help='Reuse existing current KLT displacement for shared-state velocity/net-acceleration correction; no extra image flow pass or position KF')
    p.add_argument('--same-window-motion',action='store_true',help='Check current flow acceleration against positions using its same-window fitted velocity intercept; preserves original residual gates')
    p.add_argument('--curved-motion',action='store_true',help='Optional qualified quadratic velocity curve for current acceleration only; held-out recent velocities must improve; no future jerk extrapolation')
    p.add_argument('--no-curved-motion',action='store_true',help='Control: preserve linear current-flow acceleration in the target candidate')
    p.add_argument('--flow-anchor-gauge',action='store_true',help='Calibrate coherent feature-motion drift from fresh body anchors; leaves current body positions/eligibility unchanged')
    p.add_argument('--deadline-nn',action='store_true',help='Use spare paced source-time deadline budget for an already submitted NN reply, reserving recent downstream work')
    p.add_argument('--empty-search-stride',action='store_true',help='Use ordinary NN stride for cold/long-empty acquisition; recent target loss still retries immediately and image tracking remains at every source cutoff')
    p.add_argument('--contained-reanchor-motion',action='store_true',help='Allow measured rigid patch velocity at weak-to-strong body expansion only with current containment and strict bidirectional similarity support; positions and rotor gates unchanged')
    p.add_argument('--no-contained-reanchor-motion',action='store_true',help='Control: retain original weak-to-strong box reanchor motion gate even with target-candidate')
    p.add_argument('--position-history-rebase',action='store_true',help='Rebase old consistency coordinates after a small current detector shape reanchor under a stable side axis; no four-circle template required')
    p.add_argument('--no-position-history-rebase',action='store_true',help='Control: retain the original position-consistency history in the target candidate')
    p.add_argument('--flow-force-balance',action='store_true',help='Estimate lumped disturbance from same-window velocity minus known input integral; implies shared image motion and responsive force')
    p.add_argument('--direct-force-balance',action='store_true',help='Qualified same-window residual replaces an older less precise disturbance component without extra EWMA lag; preserves old balance control')
    p.add_argument('--curved-force-balance',action='store_true',help='Experimental current curved residual after removing the known input integral; implies flow-force-balance')
    p.add_argument('--evidence-gain',action='store_true',help='Qualified novel gain evidence replaces the arbitrary initial gain without an extra 1s lag; preserve instantaneous input gauge')
    p.add_argument('--no-evidence-gain',action='store_true',help='Control: retain fixed-time slow effective-gain updates')
    p.add_argument('--endpoint-force-balance',action='store_true',help='Qualified current acceleration endpoint updates an unidentified unknown-force component when integral input history is incomplete')
    p.add_argument('--no-endpoint-force-balance',action='store_true',help='Control: preserve earlier disturbance when its full input history is incomplete')
    p.add_argument('--axis-amplitude-bridge',action='store_true',help='Current observed axis may use <=0.12s recent period amplitude; held amplitude never teaches gain')
    p.add_argument('--legacy-motion-anchor-gate',action='store_true',help='Control: reject motion from a weak previous box even when a current strong NN reanchor overlaps the measured tracks')
    p.add_argument('--rigid-motion',action='store_true',help='Experimental: fit existing current LK pairs to a similarity at the body/five-circle center; boxes stay original; implies reuse image motion')
    p.add_argument('--rigid-box-center',action='store_true',help='Experimental: reuse strictly supported current center similarity for box translation; requires center inside current inlier hull; no extra flow pass')
    p.add_argument('--body-core-points',action='store_true',help='Experimental: balance body-core and outside features in the same60-point LK pass; strictly validate the body subset before using it')
    p.add_argument('--jit-huber',action='store_true',help='Compile the existing small convex Huber objective; warm before pacing and fall back to original SciPy solver if gradient is not certified')
    p.add_argument('--jit-conic-residual',action='store_true',help='Compile unchanged per-point conic residual arithmetic; no contour or hypothesis removed; implies optimized compute')
    p.add_argument('--weak-body-continuation',action='store_true',help='Experimental ByteTrack-style low-score current NN continuation, only within300ms of an actual >=.25 NN anchor; same forward pass, no confidence promotion')
    p.add_argument('--wide-motion-search',action='store_true',help='Search union of past ROI and past-state displacement proposal; large motion still needs12 current rigid inliers/.75/9px2 and center support; no predicted detections')
    p.add_argument('--defer-decoder-gc',action='store_true',help='Experimental: defer cyclic GC in the finite RAW decoder subprocess; ordinary reference counts stay enabled, GC events audited separately')
    p.add_argument('--budgeted-nn-recovery',action='store_true',help='Use existing current-NN loss recovery only when estimated service plus recent downstream work fits the actual paced source slot')
    p.add_argument('--legacy-realtime30',action='store_true',help='Control: previous realtime30 schedule and original numerical solvers; optional flags can still be explicitly enabled')
    p.add_argument('--position-net-force',action='store_true',help='Experimental single-model net-acceleration fallback from past observed body positions; preserves current/held thrust and qualified flow')
    p.add_argument('--body-workspace',action='store_true',help='Reuse installed LSD object and an exact lookup of its existing log(native-count) input; no reconstruction, sampling or threshold change')
    p.add_argument('--tangent-disturbance',action='store_true',help='Experimental: qualified current motion updates only the unknown-force component perpendicular to current axis; implies flow-force balance')
    p.add_argument('--structure-guided-nn',action='store_true',help='When past geometry and current strong KLT agree, double the NN interval up to0.3s; current image/geometry/period updates keep full source rate')
    p.add_argument('--with-kinematic-control',action='store_true',help='Research control only; default physical-only runs no parallel traditional forecast')
    period_options=p.add_mutually_exclusive_group()
    period_options.add_argument('--visibility-period',action='store_true',help='Per-model-slot f^2 normalization; keeps original unmodelled side control')
    period_options.add_argument('--rotor-thrust-bank',action='store_true',help='Experimental four distinct f-squared states and bounded missing-rotor uncertainty; does not infer metric force from attitude alone')
    period_options.add_argument('--pooled-axis-period',action='store_true',help='Experimental distinct pooled-side gauge; first side mean-error pilot regressed')
    p.add_argument('--delayed-disturbance-control',action='store_true',help='Control: hold lumped disturbance until the historical8-row gain-learning gate')
    p.add_argument('--no-force-gap-hold',action='store_true',help='Control: immediately drop force acceleration on every missing period/axis observation')
    p.add_argument('--legacy-gain-quality',action='store_true',help='Control: historical gain-uncertainty gate without the additional2sigma-proxy requirement')
    p.add_argument('--strict-gain-quality',dest='legacy_gain_quality',action='store_false',help='Experimental2sigma-proxy gain gate; current pilot predictions unchanged')
    p.set_defaults(legacy_gain_quality=True)
    p.add_argument('--serial-decode',action='store_true',help='Disable bounded one-packet native decoder prefetch')
    p.add_argument('--raw-read-chunk',type=int,default=100000,help='SDK RAW decode read chunk; source time windows and every native event remain unchanged')
    p.add_argument('--raw-read-buffers',type=int,default=3,help='SDK public RAW native read-buffer count; bounded memory; source windows remain unchanged')
    p.add_argument('--raw-slot-events',type=int,default=2000000,help='Each of two RAW/filtered shared slots holds this many events,1..16000000; larger slots avoid queued byte copies on bursts; original2M default retained')
    p.add_argument('--raw-reader-backend',choices=['iterator','rolling'],default='iterator',help='Public SDK delta-time iterator or bounded rolling reader; no SDK modifications')
    p.add_argument('--audit-gc',action='store_true',help='Record cyclic-GC pauses in the timed main-process stream')
    p.add_argument('--defer-cyclic-gc',action='store_true',help='Collect before this finite replay and defer cyclic-GC until cleanup; Python reference-count reclamation remains active')
    p.add_argument('--thread-decode',action='store_true',help='Use thread rather than separate decoder process')
    p.add_argument('--reuse-conic-uncertainty',action='store_true',help='Experimental conservative covariance reuse; no demonstrated speed benefit yet')
    p.add_argument('--legacy-structure',action='store_true',help='Frozen original four/three/axis model instead of continuous shared-state extension')
    p.add_argument('--no-shared-arc-recovery',action='store_true',help='Disable shared-template guidance of current genuine contour arcs')
    p.add_argument('--guide-full-only',action='store_true',help='Retain original guidance expiry since last full quad; control arm')
    p.add_argument('--strict-guide-area',action='store_true',help='Use strong count mask also for area evidence; original control')
    p.add_argument('--bipolar-guide-area',action='store_true',help='Experimental124 occupancy area; negative short pilot, disabled by default')
    p.add_argument('--no-current-shape-feedback',action='store_true',help='Control: retain transported conic shapes after partial updates')
    p.add_argument('--partial-arc-information',action='store_true',help='Experimental: current short arcs constrain only observable template directions; no extra detected rotors')
    p.add_argument('--no-partial-arc-information',dest='partial_arc_information',action='store_false',help='Control: disable partial normal-distance constraints')
    p.set_defaults(partial_arc_information=True)
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    p.add_argument('--compact-output',action='store_true',help='Small forecast/body/five-circle output records; inference and timing logs unchanged; full diagnostics remain default')
    p.add_argument('--lean-history',action='store_true',help='Keep one full current record and exact columnar timings/counters; older diagnostics remain on disk; no state or output schema changes')
    args=p.parse_args()
    if (args.rotor_gpu_front or args.rotor_legacy_force or args.rotor_legacy_decomposition or args.rotor_no_phase or args.rotor_native_gpu or args.rotor_fast_geometry or args.rotor_prefetch or args.rotor_local_phase or args.rotor_memory_cache or args.no_preview or args.rotor_relative_observation or args.rotor_relock_support_guard or args.rotor_border_motion or args.rotor_max_backlog_ms or args.rotor_legacy_visibility or args.rotor_allow_invalid_motion_fallback or args.rotor_gap_prediction is not None or args.rotor_reference_mode is not None) and not args.rotor_only:
        p.error('rotor-gpu-front, rotor-legacy-force, rotor-legacy-decomposition and no-preview require --rotor-only')
    if args.rotor_only:
        if args.realtime30 or args.recall30 or args.recall90 or args.publish_hz:
            p.error('rotor-only is a separate candidate; omit realtime/recall/independent publication profiles. Its --paced profile measures actual source release.')
        from run_rotor_only import run as run_native_rotors
        args.cache=None
        return run_native_rotors(args)
    if args.unresolved_geometry_stride is None:args.unresolved_geometry_stride=3 if args.recall90 else 1
    if args.current_box_gate and args.body_association:p.error('Choose one current-box association adapter')
    if args.recall90:
        if not args.detector_model or not args.detector_sha256:p.error('recall90 requires explicit small-target detector-model and detector-sha256')
        args.recall30=True;args.singer_state=True;args.synchronous_start=True;args.synchronous_acquisition=True
        args.current_box_gate=True;args.native_detector_window=True
        if args.body_association:p.error('recall90 current-box gate and fixed-rate ByteTrack are separate controls')
    if args.recall30:
        if args.legacy_realtime30:p.error('recall30 and legacy-realtime30 are separate profiles')
        args.realtime30=True;args.small_target_native_window=True;args.synchronous_small_nn=True
    if args.detector_window_fallback and (args.native_detector_window or args.small_target_native_window or args.synchronous_small_nn or args.body_association or args.roi_recovery or args.weak_body_continuation or args.weak_roi_verify or args.weak_verifier_model):
        p.error('Native-window fallback is separate from other experimental semantic recovery modes')
    if args.legacy_realtime30:args.realtime30=True
    if args.realtime30:
        if args.hz!=30.:p.error('realtime30 requires hz30')
        args.target_candidate=True;args.rigid_box_center=True;args.process_period=True;args.publish_hz=30.
        if not args.legacy_realtime30:
            args.jit_huber=True;args.jit_conic_residual=True;args.budgeted_nn_recovery=True;args.body_workspace=True
    if not np.isfinite(args.publish_hz) or args.publish_hz<0 or args.publish_hz>args.hz:p.error('publish-hz must be0..hz')
    if args.publish_hz:
        if not args.target_candidate or args.thrust_trend_forecast:p.error('independent publisher requires target-candidate and no experimental thrust trend')
        args.paced=True
    if args.budgeted_nn_recovery:
        if not args.paced:p.error('budgeted-nn-recovery requires paced source deadlines')
        args.current_nn_on_loss=True
    if bool(args.detector_model)!=bool(args.detector_sha256):p.error('Provide both --detector-model and --detector-sha256')
    if bool(args.weak_verifier_model)!=bool(args.weak_verifier_sha256):p.error('Provide both weak verifier model and SHA256')
    if args.weak_verifier_model and (args.body_association or args.weak_roi_verify):p.error('Weak verifier is separate from body association and crop verification')
    if args.weak_body_continuation and (args.body_association or args.weak_roi_verify or args.weak_verifier_model or args.roi_recovery):
        p.error('Keep experimental low-score continuation separate from other semantic recovery adapters')
    if args.target_candidate:
        args.goal_profile=True;args.jit_conics=True;args.flow_force_balance=True
        args.normal_turn_forecast=not args.no_normal_turn_forecast
        args.curved_motion=not args.no_curved_motion
        args.adaptive_force_memory=not args.no_adaptive_force_memory
        args.evidence_gain=not args.no_evidence_gain
        args.endpoint_force_balance=not args.no_endpoint_force_balance
        args.position_history_rebase=not args.no_position_history_rebase
        args.contained_reanchor_motion=not args.no_contained_reanchor_motion
        args.weak_motion_support=not args.no_weak_motion_support
        args.axis_amplitude_bridge=True;args.defer_cyclic_gc=True
        args.parallel_period=True;args.adaptive_period=not args.process_period
    if args.goal_profile:
        args.async_nn=True;args.prefilter_decode=True;args.side_axis_budget=True;args.responsive_force=True
    if args.roi_recovery:args.current_nn_on_loss=True
    if args.spatial_weak_motion:args.weak_motion_support=True
    if args.no_weak_motion_support:args.weak_motion_support=False;args.spatial_weak_motion=False
    if args.no_position_history_rebase:args.position_history_rebase=False
    if args.no_contained_reanchor_motion:args.contained_reanchor_motion=False
    if args.contained_reanchor_motion:args.rigid_motion=True;args.reuse_image_motion=True
    if args.weak_motion_support:args.rigid_motion=True
    if args.visible_motion_support:args.rigid_motion=True
    if args.jit_conics:args.jit_counts=True
    if args.jit_conic_residual:args.optimized_compute=True
    if args.sdk_avx2:args.jit_counts=True
    if args.rigid_box_center:args.rigid_motion=True
    if args.body_core_points:args.rigid_box_center=True;args.rigid_motion=True
    if args.rigid_motion:args.reuse_image_motion=True
    if args.same_window_motion:args.reuse_image_motion=True
    if args.curved_motion:args.reuse_image_motion=True
    if args.flow_anchor_gauge:args.reuse_image_motion=True
    if args.position_history_rebase:args.reuse_image_motion=True
    if args.no_endpoint_force_balance:args.endpoint_force_balance=False
    if args.endpoint_force_balance:args.tangent_disturbance=True
    if args.tangent_disturbance:args.flow_force_balance=True
    if args.flow_force_balance:args.reuse_image_motion=True;args.responsive_force=True
    if args.direct_force_balance:args.reuse_image_motion=True;args.responsive_force=True;args.flow_force_balance=True
    if args.curved_force_balance:args.reuse_image_motion=True;args.responsive_force=True;args.flow_force_balance=True
    if args.process_period:args.parallel_period=True
    if args.temporal_period_support:args.period_all_pixels=True
    if args.aligned_period_recovery:args.reuse_image_motion=True
    if args.body_period_stream:args.reuse_image_motion=True
    if args.no_normal_turn_forecast:args.normal_turn_forecast=False
    if args.normal_turn_forecast:args.responsive_force=True;args.reuse_image_motion=True
    if args.no_curved_motion:args.curved_motion=False
    if args.no_adaptive_force_memory:args.adaptive_force_memory=False
    if args.adaptive_force_memory:args.responsive_force=True
    if args.shorter_force_memory:args.adaptive_force_memory=True;args.responsive_force=True
    if args.thrust_trend_forecast:args.responsive_force=True
    if args.no_evidence_gain:args.evidence_gain=False
    if args.period_all_pixels and args.process_period:p.error('--period-all-pixels is not implemented in the separate period process')
    if args.aligned_period_recovery and args.body_period_stream:p.error('Choose either historical aligned retry or persistent body period stream')
    if args.nn_stride<1:p.error('nn-stride must be positive')
    if args.raw_read_chunk<1:p.error('raw-read-chunk must be positive')
    if args.raw_read_buffers<1:p.error('raw-read-buffers must be positive')
    if not 1<=args.raw_slot_events<=16000000:p.error('raw-slot-events must be in1..16000000')
    if args.jit_counts and args.native_histogram:p.error('Choose only one optional count backend')
    if args.jit_conics and not args.optimized_compute:p.error('--jit-conics requires optimized compute')
    if args.roi_recovery and args.body_association:p.error('ROI recovery currently uses its own strict current-box matching; do not combine with body association')
    if args.weak_roi_verify and args.body_association:p.error('Weak ROI verification is not combined with the independent body association adapter')
    if args.structure_guided_nn and args.body_association:p.error('Structure-guided variable NN cadence is not validated with the fixed-rate body-association adapter')
    if args.prefilter_decode and (args.thread_decode or args.serial_decode):p.error('--prefilter-decode requires separate decoder process')
    run(args)


if __name__=='__main__':main()
