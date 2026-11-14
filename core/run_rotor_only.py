"""RAW -> rotor-only native ON/OFF front -> shared state -> physical forecasts.

The default factory reads no model weights, body NN, G reconstruction or labels.
An explicit learned-response factory may load native-trained scalar parameters.
"""
from pathlib import Path
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import argparse,hashlib,json,time
import cv2,numpy as np
from rotor_only_front import RotorOnlyFront,filtered_rgb
from singer_position_state import SingerPositionState
from responsive_force import DampedResponsiveForce
from physical_publication import evaluate
from rotor_thrust_bank import RotorThrustBankProxy, LegacyRotorThrustBankProxy
from shared_phase_fft import SharedPhaseFFT
from phase_rate_bridge import phase_supported_rates
from local_pixel_phase import local_pixel_phase

ROOT=Path(__file__).resolve().parents[1]
SOURCE_FILES = ['run_rotor_only.py','rotor_only_front.py','rotor_period_worker.py',
    'rotor_support_front.py','predictive_harmonics.py','predictive_harmonics_cuda.py',
    'fast_structure_front.py','rotor_force_observer.py','rotor_cuda_device.py',
    'rotor_cuda_worker.py','rotor_thrust_bank.py','visibility_period.py','shared_phase_fft.py','phase_rate_bridge.py',
    'rotor_nvrtc.py','native/rotor_harmonics.cu','fast_geometry_compute.py','local_pixel_phase.py',
    'fast_raw_reader.py','live_decode.py','native_plane_motion.py','live_local_flow.py','native_correlation.py',
    'native_axis_workspace.py','native_histogram_front.py','native_history_buffer.py','relative_rotor_observer.py','research/stage6/silhouette.py','physical_publication.py','singer_position_state.py']
SOURCE_FILES += ['continuous_five_circle.py','five_circle_model.py','shared_arc_recovery.py',
    'partial_arc_information.py','responsive_force.py','adaptive_force_memory.py',
    'local_period_tracks.py',
    'src/event_drone_forecast/geometry.py','research/stage146/branches146.py',
    'research/stage147/curvature147.py','research/stage148/guided148.py','research/stage79/native79.py']
SOURCE_FILES += ['curvature_visibility.py']
SOURCE_FILES += ['batched_arc_fit.py']
SOURCE_FILES += ['coherent_gap_observer.py']
SOURCE_FILES += ['layout_reference_observer.py']
SOURCE_FILES += ['isolated_rim_visibility.py']
SOURCE_FILES += ['native_cycle_consistency.py']
SOURCE_FILES += ['smooth_layout_force.py','matern_force_observer.py']
SOURCE_FILES += ['integrated_force_rate.py']
SOURCE_FILES += ['normal_response_observer.py','normal_turn_forecast.py','axis_amplitude_bridge.py']
SOURCE_FILES += ['separated_force_response.py','polarity_motion_quality.py']
SOURCE_FILES += ['adaptive_response_observer.py','causal_response_credit.py']
SOURCE_FILES += ['native_geometry_kernels.py','native_ellipse_hypotheses.py',
    'native/ellipse_hypotheses.cpp','native/geometry_kernels206.cpp']
SOURCE_FILES += ['regional_graph_front.py','cuda_regional_graph.py','regional_timing_period.py','native/regional_graph211.cu']
SOURCE_FILES += ['native_arc_search.py','native/arc_search215.cpp']
SOURCE_FILES += ['native/arc_uncertainty216.cpp','ellipse_visibility.py']
SOURCE_FILES += ['run_rotor_candidate.py']
SOURCE_FILES += ['runtime_gc_trace.py']
SOURCE_FILES += ['cooperative_raw_reader.py']
SOURCE_FILES += ['identifiable_tangent_memory.py','tangential_response_memory.py']
SOURCE_FILES += ['rotor_torque_innovation.py','native_torque_inputs.py','rotor_torque_partition.py','rotor_torque_response.py']
SOURCE_FILES += ['inverse_normal_response.py']
SOURCE_FILES += ['periodic_track_continuation.py','rotor_spatial_bridge.py']
SOURCE_FILES += ['anisotropic_response.py','visible_force_rate.py','rotor_frequency_drive.py','run_rotor_force_candidate.py']
SOURCE_FILES += ['conditional_unknown_force.py','calibrated_force_rollout.py','learned_rotor_predictor.py','run_rotor_learned_candidate.py','models/native_response_v1.json']
SOURCE_FILES += ['models/native_response_fixed_frame_v1.json']
SOURCE_FILES += ['visibility_response.py','conic_observability.py','models/native_response_visibility_v1.json']
SOURCE_FILES += ['plane_conditioned_response.py','future_drive_response.py','models/native_response_plane_v1.json']
SOURCE_FILES += ['observation_roles.py','observed_rotor_predictor.py','run_rotor_observation_candidate.py']
SOURCE_FILES += ['native_fov_observation.py','rotor_template_projection.py']
SOURCE_FILES += ['reference_qualification.py']
SOURCE_FILES += ['shared_endpoint_observer.py','run_rotor_endpoint_candidate.py']
SOURCE_FILES += ['run_rotor_projection_candidate.py','rotor_scale_response.py']
SOURCE_FILES += ['temporal_rotor_birth.py','bootstrap_native_reference.py','run_rotor_complete_candidate.py','window_position_observation.py']
SOURCE_FILES += ['diffuse_rotor_startup.py','run_rotor_accuracy_candidate.py']
SOURCE_FILES += ['local_conic_opportunity.py','run_rotor_local_geometry_candidate.py']
SOURCE_FILES += ['positive_normal_startup.py','run_rotor_positive_candidate.py']
SOURCE_FILES += ['equilibrium_gain_prior.py','anonymous_rotor_group.py','run_rotor_partial_candidate.py','run_rotor_polarity_candidate.py']
SOURCE_FILES += ['native_peak_shape.py','run_rotor_aperture_candidate.py','run_rotor_joint_peak_candidate.py']
SOURCE_FILES += ['run_rotor_joint_only_candidate.py']
SOURCE_FILES += ['run_rotor_full_joint_candidate.py']
SOURCE_FILES += ['run_rotor_multichannel_candidate.py']
SOURCE_FILES += ['symmetric_native_motion.py','run_rotor_symmetric_candidate.py']
SOURCE_FILES += ['run_rotor_supported_plane_candidate.py']
SOURCE_FILES += ['timestamped_rotor_motion.py','run_rotor_timestamped_candidate.py']
SOURCE_FILES += ['symmetric_match_surface.py','run_rotor_symmetric_surface_candidate.py']
SOURCE_FILES += ['run_rotor_complete_surface_candidate.py']
SOURCE_FILES += ['local_contrast_motion.py','run_rotor_local_contrast_candidate.py']
SOURCE_FILES += ['run_rotor_contrast_symmetric_candidate.py']
SOURCE_FILES += ['run_rotor_contrast_plane_candidate.py']
SOURCE_FILES += ['background_unknown_response.py','background_relative_force.py','background_endpoint_observer.py',
    'background_reference_predictor.py','run_rotor_background_candidate.py','run_rotor_background_endpoint_candidate.py']
SOURCE_FILES += ['camera_reference_selection.py','background_selection_predictor.py',
    'run_rotor_background_selection_candidate.py','run_rotor_background_memory_candidate.py']


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


class RotorPredictor:
    def __init__(self,gpu_python=None,legacy_force=False,gpu_front=False,legacy_decomposition=False,no_phase=False,native_gpu=False,fast_geometry=False,local_phase=False,subpixel_motion=False,motion_prior=False,plane_motion=False,coarse_motion=False,entry_init=False,regional_seeds=False,remote_front=False,front_slots=3,relative_observation=False,relock_support_guard=False,border_motion=False,track_local_source=False,legacy_visibility=False,guard_invalid_motion=None,gap_prediction=None,reference_mode=None,unknown_force_model='free',regional_graph=False,curvature_hz=0.,projected_fov_guard=False,tangential_memory=False,torque_innovation=False,inverse_normal_response=False,anisotropic_response=False,visible_force_rate=False,periodic_continuation=False):
        if (anisotropic_response or visible_force_rate) and not inverse_normal_response:
            raise ValueError('New force rollout requires --rotor-inverse-normal-response')
        if anisotropic_response and tangential_memory:
            raise ValueError('Anisotropic and isotropic tangential memory are alternative models')
        if periodic_continuation and (not relative_observation or legacy_visibility or not fast_geometry):
            raise ValueError('Current-period continuation requires the modern native relative front')
        self.anisotropic_response=None;self.visible_force_rate=None
        if anisotropic_response:
            from anisotropic_response import AnisotropicResponse
            self.anisotropic_response=AnisotropicResponse()
        if visible_force_rate:
            from visible_force_rate import VisibleForceRate
            self.visible_force_rate=VisibleForceRate()
        if inverse_normal_response and (legacy_force or legacy_decomposition or not relative_observation or unknown_force_model!='plane-adaptive'):
            raise ValueError('Inverse normal response requires native relative observations and the plane-adaptive observer')
        self.inverse_normal_response=None
        if inverse_normal_response:
            from inverse_normal_response import InverseNormalResponse
            self.inverse_normal_response=InverseNormalResponse()
        if torque_innovation and (legacy_force or legacy_decomposition or not relative_observation or unknown_force_model!='plane-adaptive'):
            raise ValueError('Torque innovation requires current native relative observations and the plane-adaptive observer')
        self.torque_innovation=None
        if torque_innovation:
            from rotor_torque_innovation import RotorTorqueInnovation
            self.torque_innovation=RotorTorqueInnovation()
        if tangential_memory and (legacy_force or not relative_observation or unknown_force_model not in ('plane','plane-separated','plane-adaptive')):
            raise ValueError('Tangential memory requires native relative observations and a plane-conditioned force observer')
        self.tangential_memory=None
        if tangential_memory:
            from identifiable_tangent_memory import IdentifiableTangentMemory
            self.tangential_memory=IdentifiableTangentMemory()
        if curvature_hz and not 1<=curvature_hz<=30:raise ValueError('Curvature rate must be0 or1..30Hz')
        if regional_graph and not (gpu_python and gpu_front and native_gpu):raise ValueError('Regional graph requires native GPU front')
        if gap_prediction is not None and not relative_observation:raise ValueError('Explicit gap profile requires relative observations')
        reference_mode=('fixed' if relative_observation and not legacy_visibility else 'legacy') if reference_mode is None else reference_mode
        if reference_mode not in ('legacy','fixed'):raise ValueError('Reference mode must be legacy/fixed')
        if reference_mode=='fixed' and (not relative_observation or legacy_force):raise ValueError('Fixed layout reference requires the joint relative observer')
        if unknown_force_model not in ('free','plane','plane-separated','plane-adaptive'):raise ValueError('Unknown force model must be free/plane/plane-separated/plane-adaptive')
        if unknown_force_model in ('plane','plane-separated','plane-adaptive') and (reference_mode!='fixed' or legacy_visibility or gap_prediction not in (None,'short')):
            raise ValueError('Plane-conditioned unknown force requires current relative/fixed reference and short gap policy')
        self.reference_mode=reference_mode
        self.period_worker=None
        if gpu_python:
            if gpu_front:
                from rotor_cuda_worker import CudaRotorFrontWorker
                self.period_worker=CudaRotorFrontWorker(gpu_python,native_front=native_gpu,regional_seeds=regional_seeds,remote_unpack=remote_front,slots=front_slots,regional_graph=regional_graph)
            else:
                from rotor_period_worker import CudaPeriodWorker
                self.period_worker=CudaPeriodWorker(gpu_python)
        elif gpu_front:raise ValueError('GPU front requires the existing CUDA Python interpreter')
        from rotor_force_observer import RotorForceObserver
        if relative_observation:
            if legacy_force:raise ValueError('Relative observations require the joint force observer')
            from relative_rotor_observer import RelativeRotorForceObserver
            RotorForceObserver=RelativeRotorForceObserver
            mode=('legacy' if legacy_visibility else 'short') if gap_prediction is None else gap_prediction
            if mode not in ('legacy','coherent','short'):raise ValueError('Gap mode must be legacy/coherent/short')
            if mode!='legacy':
                from coherent_gap_observer import CoherentGapObserver,ShortGapObserver
                RotorForceObserver=ShortGapObserver if mode=='short' else CoherentGapObserver
            if reference_mode=='fixed':
                from layout_reference_observer import LayoutReferenceObserver,CoherentLayoutReferenceObserver,LegacyGapLayoutReferenceObserver
                RotorForceObserver={'short':LayoutReferenceObserver,'coherent':CoherentLayoutReferenceObserver,'legacy':LegacyGapLayoutReferenceObserver}[mode]
        self.relative_observation=bool(relative_observation)
        self.front=RotorOnlyFront(self.period_worker,fast_geometry=fast_geometry,subpixel_motion=subpixel_motion,motion_prior=motion_prior,plane_motion=plane_motion,coarse_motion=coarse_motion,relock_support_guard=relock_support_guard,border_motion=border_motion,legacy_visibility=legacy_visibility,guard_invalid_motion=guard_invalid_motion,rebase_geometry_reference=reference_mode=='fixed',fixed_output_reference=reference_mode=='fixed');self.legacy_force=legacy_force
        self.state=SingerPositionState() if legacy_force else None
        self.front.current_region_search=bool(regional_graph)
        if periodic_continuation:
            from periodic_track_continuation import PeriodicTrackContinuation
            self.front.periodic_continuation=PeriodicTrackContinuation(appearance_uncertainty=True,structural_motion=True)
        self.front.geometry.curvature_interval_s=1/curvature_hz if curvature_hz else 0.
        self.front.geometry.projected_fov_guard=bool(projected_fov_guard)
        if unknown_force_model in ('plane','plane-separated','plane-adaptive'):
            if unknown_force_model=='plane-adaptive':
                from adaptive_response_observer import AdaptiveResponseObserver
                self.force=AdaptiveResponseObserver()
            elif unknown_force_model=='plane-separated':
                from separated_force_response import SeparatedForceObserver
                self.force=SeparatedForceObserver()
            else:
                from normal_response_observer import NormalResponseObserver
                self.force=NormalResponseObserver()
        else:self.force=DampedResponsiveForce() if legacy_force else RotorForceObserver()
        self.unknown_force_model=unknown_force_model
        self.reference_frequency=None
        self.legacy_decomposition=bool(legacy_force or legacy_decomposition)
        self.period_proxy=LegacyRotorThrustBankProxy() if self.legacy_decomposition else RotorThrustBankProxy(vector_pool=local_phase)
        self.local_phase=local_phase
        self.source_tracker=None
        if track_local_source:
            if not local_phase or self.legacy_decomposition:raise ValueError('Local source tracking requires --rotor-local-phase and current observed-force decomposition')
            from local_period_tracks import LocalPeriodTracks
            self.source_tracker=LocalPeriodTracks()
        self.entry_init=bool(entry_init);self.last_provisional_t=None
        self.phase=None if no_phase or self.legacy_decomposition else SharedPhaseFFT()
        self.front.provide_phase_events=self.phase is not None

    def step(self,events,t):
        started=time.perf_counter();frame=self.front.step(events,t);body=frame['body'];front_ms=1000*(time.perf_counter()-started)
        assignment_started=time.perf_counter()
        # The detector already returns all eligible native pixel indices.
        # Frequency assignment only needs the accepted subset, not a dense
        # mesh covering hundreds of thousands of empty pixels per rotor.
        eligible=np.asarray(frame['period_pixels_unique'],dtype=np.int64)
        values=frame['frequency_hz'].reshape(-1)[eligible];accepted=values>0
        accepted_pixels=eligible[accepted];frequencies=values[accepted]
        px,py=accepted_pixels%1280,accepted_pixels//1280
        frequency=float(np.median(frequencies)) if len(frequencies)>=12 else None
        if frequency and self.reference_frequency is None:self.reference_frequency=frequency
        axis=frame['structure'].get('axis_xy');u=None;quality=0.
        current_pose=bool(len(frame['geometry'].get('rotors',[]))>=2 or
            frame['geometry'].get('side',{}).get('valid'))
        rotors=frame['geometry'].get('rotors',[]);rates=[]
        for rotor in rotors:
            cx,cy=rotor['cx'],rotor['cy'];a=rotor['major']/2;b=rotor['minor']/2
            angle=np.deg2rad(rotor['angle'])
            u0=(px-cx)*np.cos(angle)+(py-cy)*np.sin(angle)
            v0=-(px-cx)*np.sin(angle)+(py-cy)*np.cos(angle)
            inside=(u0/max(a,1e-6))**2+(v0/max(b,1e-6))**2<=1.
            fs=frequencies[inside]
            median=float(np.median(fs)) if len(fs) else None
            spread=float(np.median(np.abs(np.log(fs/median)))) if len(fs) else None
            candidates=[]
            if self.phase is not None and frame['structure'].get('mode') in ('four','three','partial_two'):
                flat=accepted_pixels[inside]
                values=[v for p in flat for v in frame.get('candidate_lookup',{}).get(int(p),())]
                if values:
                    f,count=np.unique(values,return_counts=True)
                    candidates=f[np.argsort(-count)[:6]].tolist()
            rates.append(dict(valid=bool(len(fs)>=6 and spread<=.08),frequency_hz=median,
                spectral_candidates_hz=candidates,
                periodic_pixels=int(len(fs)),log_frequency_MAD=spread,mechanical_RPM_measured=False))
        assignment_ms=1000*(time.perf_counter()-assignment_started)
        # Geometry completion does not create a force measurement. Only current
        # supported contributions enter u; hidden motors belong to unknown d.
        nonflat=frame['structure'].get('mode') in ('four','three','partial_two')
        phase_info=dict(qualified_count=0,reason='disabled');phase_bridge=dict(selected=[])
        if self.phase is not None:
            shift=(frame.get('motion',{}).get('estimate') or {})
            velocity=np.array([shift.get('dx',0.),shift.get('dy',0.)])*30.
            # Deterministic native-pixel subset retains all timestamps at each
            # selected pixel. This is explicit sampling, not lossless all-events.
            e=frame['filtered_events'] if nonflat else events[:0]
            if len(e):e=e[((e['x'].astype(np.int32)+3*e['y'])%4)==0]
            phase_info=self.phase.step(e,t,body,frame['structure'],rotors,velocity)
            rates,phase_bridge=phase_supported_rates(frame['structure'],rotors,rates,phase_info)
        pooled=dict(qualified=bool(frequency and current_pose),frequency_hz=frequency,
            selection_source='global_pixel_median' if self.local_phase else None)
        local_info=dict(qualified=False,reason='disabled')
        if self.local_phase and not nonflat and body.get('detected'):
            local_info=local_pixel_phase(frame,body['center'])
            if local_info['qualified']:pooled=dict(qualified=current_pose,frequency_hz=local_info['frequency_hz'],selection_source='local_pixel_first_harmonic')
        source_tracking=None
        if self.source_tracker is not None:
            source_tracking=self.source_tracker.step(t,local_info)
            if not nonflat:pooled=dict(source_tracking,qualified=bool(source_tracking['qualified'] and current_pose))
        if frame.get('current_support_without_force_update'):
            from observation_roles import unavailable_force_metadata
            u=None;quality=0.;period_info=unavailable_force_metadata()
            for rate in rates:
                rate['raw_pixel_frequency_candidate_valid']=rate['valid']
                rate['valid']=False
                rate['qualification_withheld_reason']='continued_current_support_is_not_a_motor_frequency_measurement'
        else:
            u,quality,period_info=self.period_proxy.step(t,frame['structure'],rotors,
                rates if nonflat or getattr(self.period_proxy,'accepts_anonymous_current_conics',False) else [],pooled)
        if frame.get('regional_only_motor_frequency_unobserved'):
            u=None;quality=0.
            period_info.update(observed_components={},coherent=False,learning_input_current=False,
                source='regional_support_without_motor_frequency',reason='all_thrust_owned_by_unknown_force',
                relative_f2=None,rotor_bank_used=False)
        if not current_pose:u=None;quality=0.
        if self.legacy_force:
            coefficient,valid,state_info=self.state.step(t,body)
            self.force.input_learning_current=period_info.get('learning_input_current',True)
            _,force=self.force.step(t,coefficient,body.get('center') if body['detected'] else None,u,quality)
        else:
            entry_started=False
            if self.entry_init:
                if body.get('detected') and body.get('center_provisional'):self.last_provisional_t=t
                fresh=bool(self.force.x is None or self.force.seen is not None and t-self.force.seen>.5)
                if fresh and body.get('detected') and not body.get('center_provisional'):
                    entry_started=bool(self.last_provisional_t is not None and 0<=t-self.last_provisional_t<=.16)
                    self.force.two_point_init=entry_started
            extra={}
            if getattr(self.force,'requires_current_structure',False):extra['structure']=frame['structure']
            coefficient,valid,force,state_info=self.force.step(t,body,u,quality,
                allow_gain_learning=period_info.get('learning_input_current',True),
                input_meta=None if self.legacy_decomposition else period_info,
                motion=frame['motion'] if self.relative_observation else None,**extra)
            state_info['two_point_initialization_selected_by_partial_entry']=entry_started
        row=dict(t=t,body=body,coefficient=coefficient,valid=bool(valid),force=force,
            structure=frame['structure'],geometry=frame['geometry'],motion=frame['motion'],
            rates=rates,period_selection=period_info,spatial_phase=phase_info,local_temporal_phase=local_info,phase_rate_bridge=phase_bridge,
            local_source_tracking=source_tracking,
            counts=frame['counts'],reconstruction=frame['reconstruction'],state=state_info,
            backend_timing=frame.get('backend_timing',{}),front_ms=front_ms,
            regional_graph_fallback=frame.get('regional_graph_fallback'),
            front_stage_ms=frame['front_stage_ms'],
            rate_assignment_ms=assignment_ms,rate_assignment_uses_all_accepted_native_pixels=True,
            period=dict(frequency_hz=frequency,relative_frequency_squared=None if not frequency else
                period_info.get('relative_f2'),pooled_visible_proxy=not period_info.get('rotor_bank_used',False),
                selected_frequency_hz=pooled.get('frequency_hz'),selected_frequency_source=pooled.get('selection_source'),
                selected_frequency_is_normalized_proxy=pooled.get('normalized_change_proxy',False),
                hidden_rotor_frequencies_measured=False,RPM_calibrated=False),
            input_route='native_rotor_only',whole_body_NN_used=False,
            geometry_uses_original_event_pixels=True,force_direction_sign_prior='image_up')
        if getattr(self.force,'requires_current_structure',False):row['normal_turn_forecast']=state_info.get('normal_turn_forecast')
        if 'periodic_track_continuation' in frame:row['periodic_track_continuation']=frame['periodic_track_continuation']
        if 'observation_qualification' in frame:row['observation_qualification']=frame['observation_qualification']
        if self.inverse_normal_response is not None:
            row=self.inverse_normal_response.step(row)
        if self.anisotropic_response is not None:row=self.anisotropic_response.step(row)
        if self.visible_force_rate is not None:row=self.visible_force_rate.step(row,period_info)
        if self.tangential_memory is not None:
            from identifiable_tangent_memory import adapt
            tangent_info=self.inverse_normal_response.last_tangent_info if self.inverse_normal_response is not None else self.tangential_memory.observe(row)
            row=adapt(row,tangent_info)
        if self.torque_innovation is not None:
            from rotor_torque_innovation import adapt
            row=adapt(row,self.torque_innovation.observe(row,period_info))
        row['physical']=evaluate(row,[.1,.2,.4]) if coefficient is not None else {}
        if self.reference_mode=='fixed':
            row.update(output_reference=state_info.get('output_reference','initial_native_rotor_support_anchor'),
                output_position_is_COM_truth=False,layout_center_is_output_position=False)
        row['compute_ms']=(time.perf_counter()-started)*1000
        return row,frame


def windows(args):
    if hasattr(args,'_prepared_packets'):
        yield from args._prepared_packets;return
    if hasattr(args,'_owned_packets'):
        yield from args._owned_packets;return
    if args.cache:
        manifest=json.loads((Path(args.cache)/'manifest.json').read_text('utf-8'))
        for item in manifest['frames']:
            if manifest.get('storage')=='npy':
                yield float(item['t']),np.load(Path(args.cache)/item['file'],mmap_mode='r');continue
            with np.load(Path(args.cache)/f'{item["index"]:04d}.npz') as d:
                yield float(d['t']),d['events'].copy()
    else:
        backend=getattr(args,'rotor_reader','rolling')
        if backend.startswith('process-'):
            # Reuse the existing bounded two-slot decoder. It owns the SDK
            # interpreter/GIL; the estimator and CUDA service remain separate.
            from fast_raw_reader import ProcessRAW
            reader=ProcessRAW(Path(args.raw),args.start,args.duration,round(1e6/args.hz),
                read_chunk=getattr(args,'rotor_read_chunk',100000),
                read_buffers=getattr(args,'rotor_read_buffers',3),
                reader_backend=backend.removeprefix('process-'),slot_events=4000000,
                defer_cyclic_gc=bool(getattr(args,'defer_decoder_gc',False)))
            args._decode_records={}
            try:
                while True:
                    packet=reader.read()
                    if packet is None:break
                    t,e,elapsed=packet
                    args._decode_records[t]=dict(total_ms=elapsed,native_ms=reader.last_native_decode_ms,
                        cpu_ms=reader.last_native_decode_CPU_ms,overflow=reader.packet_overflow,
                        cyclic_GC_events=reader.last_gc_events)
                    yield t,e
            finally:reader.close()
            return
        from live_decode import native_batches
        yield from native_batches(Path(args.raw),args.start,args.duration,False,round(1e6/args.hz),
            read_chunk=getattr(args,'rotor_read_chunk',100000),read_buffers=getattr(args,'rotor_read_buffers',3),reader_backend=backend)


def prepared_windows(args,predictor):
    """Bounded stateless feature lookahead; model never sees future features."""
    if not getattr(args,'rotor_prefetch',False):
        yield from windows(args);return
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    worker=predictor.period_worker
    if not hasattr(worker,'prepare_uncached'):raise ValueError('Prefetch requires the GPU front')
    source=iter(windows(args))
    feature_only=bool(getattr(args,'rotor_native_gpu',False) and predictor.phase is None and
        not getattr(args,'rotor_retain_host_events',False))
    max_age=getattr(args,'rotor_max_backlog_ms',0.)/1000.
    args._skipped_observations=[];args._native_packets_read=0
    args._preparation_records={}
    def age(t):return args._source_epoch_t+time.perf_counter()-args._source_wall_epoch-t
    def skipped(t,stage):
        args._skipped_observations.append(dict(t=t,source_age_s=age(t),stage=stage,
            preparation_timing=args._preparation_records.pop(t,None),
            decode_timing=getattr(args,'_decode_records',{}).pop(t,None)))
    def read_and_prepare():
        while True:
            read_start=time.perf_counter();read_cpu=time.thread_time()
            try:t,events=next(source)
            except StopIteration:return None
            args._preparation_records[t]=dict(native_read_wall_ms=1000*(time.perf_counter()-read_start),
                native_read_thread_CPU_ms=1000*(time.thread_time()-read_cpu))
            args._native_packets_read+=1
            if getattr(args,'paced',False):
                if not hasattr(args,'_source_wall_epoch'):
                    args._source_wall_epoch=time.perf_counter();args._source_epoch_t=t
                release=args._source_wall_epoch+t-args._source_epoch_t
                time.sleep(max(0.,release-time.perf_counter()))
                if max_age and age(t)>max_age:
                    skipped(t,'before_front');continue
            break
        # Native decoder views may be borrowed. Give the main model ownership
        # before the producer advances the decoder on the next packet.
        borrowed=not getattr(args,'cache',None)
        if borrowed and not feature_only:events=events.copy()
        front_start=time.perf_counter()
        args._preparation_records[t]['front_input_source_age_ms']=1000*age(t) if getattr(args,'paced',False) else None
        features=worker.prepare_uncached(events,t)
        args._preparation_records[t].update(front_wall_ms=1000*(time.perf_counter()-front_start),
            front_ready_source_age_ms=1000*age(t) if getattr(args,'paced',False) else None)
        # The native front has consumed EVERY event before the decoder can
        # reuse its borrowed slot. With count20 geometry and spatial phase
        # disabled, the CPU model consumes feature maps only. Do not retain a
        # redundant full event copy that no downstream operation will read.
        if borrowed and feature_only:events=np.empty(0,dtype=events.dtype)
        return t,events,features
    with ThreadPoolExecutor(max_workers=1) as pool:
        depth=getattr(args,'rotor_prefetch_depth',1)
        futures=deque(pool.submit(read_and_prepare) for _ in range(depth))
        while True:
            value=futures.popleft().result()
            if value is None:break
            t,events,result=value
            futures.append(pool.submit(read_and_prepare))
            if max_age and age(t)>max_age:
                skipped(t,'before_model');continue
            worker.cached=(t,result)
            yield t,events
            if worker.cached is not None:raise RuntimeError('Current prefetched features were not consumed')


def run(args,predictor_factory=RotorPredictor):
    call_started=time.perf_counter()
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(out)
    for protected in [Path('D:/yolov10-main'),Path('E:/dataset/software/yolov10-main')]:
        if out.is_relative_to(protected):raise ValueError('Protected output')
    if args.hz!=30:raise ValueError('Current periodic front uses native33333us/30Hz only')
    if not 1<=getattr(args,'rotor_prefetch_depth',1)<=4:raise ValueError('Prefetch depth must be1..4')
    if getattr(args,'paced',False) and not getattr(args,'rotor_prefetch',False):raise ValueError('Paced rotor replay requires bounded GPU prefetch')
    max_backlog=getattr(args,'rotor_max_backlog_ms',0.)
    if not np.isfinite(max_backlog) or max_backlog<0 or (max_backlog and not getattr(args,'paced',False)):
        raise ValueError('Nonnegative backlog budget requires paced source time; zero retains no-drop baseline')
    if getattr(args,'rotor_native_gpu',False) and not getattr(args,'rotor_gpu_front',False):
        raise ValueError('--rotor-native-gpu requires --rotor-gpu-front')
    if not getattr(args,'cache',None) and (args.duration<=0 or args.start<0):raise ValueError('Nonnegative start and positive duration required')
    if getattr(args,'raw',None) and any(Path(args.raw).resolve().is_relative_to(p) for p in
            [Path('D:/yolov10-main'),Path('E:/dataset/software/yolov10-main')]):raise ValueError('Protected input')
    out.mkdir(parents=True);(out/'images').mkdir();cv2.setNumThreads(2)
    preload_s=0.;preload_bytes=0
    if getattr(args,'rotor_memory_cache',False):
        if not getattr(args,'cache',None):raise ValueError('Memory preload requires a bounded saved-event cache')
        began=time.perf_counter();packets=[]
        for t,e in windows(args):
            preload_bytes+=e.nbytes
            if preload_bytes>2*1024**3:raise ValueError('Memory replay cap2GiB exceeded; use native RAW or NPY streaming')
            packets.append((t,e.copy()))
        args._owned_packets=packets;preload_s=time.perf_counter()-began
    # Capture before workers start. A late snapshot can falsely attribute a
    # running process to edits made while its already-imported code executes.
    sources={};snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in SOURCE_FILES:
        data=(ROOT/name).read_bytes();target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        sources[name]=hashlib.sha256(data).hexdigest()
    decoder_prepare_s=0.
    if getattr(args,'rotor_reader_ready',False) and not getattr(args,'cache',None):
        from itertools import chain
        began=time.perf_counter();source=iter(windows(args));first=next(source,None)
        if first is None:raise ValueError('No native RAW windows in requested interval')
        args._prepared_raw_source=source;args._prepared_packets=chain([first],source)
        decoder_prepare_s=time.perf_counter()-began
    geometry_dll=getattr(args,'rotor_native_geometry_dll',None)
    if geometry_dll:
        if not getattr(args,'rotor_fast_geometry',False):raise ValueError('Native geometry requires fast geometry')
        from native_geometry_kernels import load_backend
        from batched_arc_fit import configure_native
        configure_native(load_backend(geometry_dll))
        (out/'native_geometry.json').write_text(json.dumps(dict(path=str(Path(geometry_dll).resolve()),
            sha256=hashlib.sha256(Path(geometry_dll).read_bytes()).hexdigest(),explicit_opt_in=True,
            thresholds_or_sampling_changed=False),indent=2),'utf-8')
    construction_started=time.perf_counter()
    try:
        predictor=predictor_factory(getattr(args,'gpu_python',None),getattr(args,'rotor_legacy_force',False),
            getattr(args,'rotor_gpu_front',False),getattr(args,'rotor_legacy_decomposition',False),getattr(args,'rotor_no_phase',False),
            getattr(args,'rotor_native_gpu',False),getattr(args,'rotor_fast_geometry',False),getattr(args,'rotor_local_phase',False),getattr(args,'rotor_subpixel_motion',False),getattr(args,'rotor_motion_prior',False),getattr(args,'rotor_plane_motion',False),getattr(args,'rotor_coarse_motion',False),getattr(args,'rotor_entry_init',False),getattr(args,'rotor_regional_seeds',False),getattr(args,'rotor_remote_front',False),front_slots=getattr(args,'rotor_prefetch_depth',1)+2,relative_observation=getattr(args,'rotor_relative_observation',False),relock_support_guard=getattr(args,'rotor_relock_support_guard',False),border_motion=getattr(args,'rotor_border_motion',False),track_local_source=getattr(args,'rotor_track_local_source',False),legacy_visibility=getattr(args,'rotor_legacy_visibility',False),guard_invalid_motion=False if getattr(args,'rotor_allow_invalid_motion_fallback',False) else None,gap_prediction=getattr(args,'rotor_gap_prediction',None),reference_mode=getattr(args,'rotor_reference_mode',None),unknown_force_model=getattr(args,'rotor_unknown_force_model','free'),regional_graph=getattr(args,'rotor_regional_graph',False),curvature_hz=getattr(args,'rotor_curvature_hz',0.),projected_fov_guard=getattr(args,'rotor_projected_fov_guard',False),tangential_memory=getattr(args,'rotor_tangential_memory',False),torque_innovation=getattr(args,'rotor_torque_innovation',False),inverse_normal_response=getattr(args,'rotor_inverse_normal_response',False),anisotropic_response=getattr(args,'rotor_anisotropic_response',False),visible_force_rate=getattr(args,'rotor_visible_force_rate',False),periodic_continuation=getattr(args,'rotor_periodic_continuation',False))
    except BaseException:
        if hasattr(args,'_prepared_raw_source'):args._prepared_raw_source.close()
        raise
    startup_s=time.perf_counter()-construction_started
    rows=[];publication_records=[];started=time.perf_counter();writer=None;previous_end=started
    stream=prepared_windows(args,predictor)
    from runtime_gc_trace import GCTrace
    gc_trace=GCTrace();gc_trace.start()
    try:
        with (out/'predictions.jsonl').open('w',encoding='utf-8') as f,(out/'publication_trace.jsonl').open('w',encoding='utf-8') as publications:
            for index,(t,events) in enumerate(stream):
                row,frame=predictor.step(events,t);rows.append(row)
                row['decode_timing']=getattr(args,'_decode_records',{}).pop(t,None)
                row['preparation_timing']=getattr(args,'_preparation_records',{}).pop(t,None)
                row['pipeline_frame_ms']=(time.perf_counter()-previous_end)*1000
                row['output_wall_elapsed_s']=time.perf_counter()-started
                previous_end=time.perf_counter()
                f.write(json.dumps(row,default=serial,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                if getattr(args,'paced',False):
                    from physical_publication import freeze,publication
                    emitted=time.perf_counter();now=args._source_epoch_t+emitted-args._source_wall_epoch
                    last_image_t=predictor.force.seen if not predictor.legacy_force else predictor.state.seen
                    record=publication(freeze(row,last_image_t),now,emitted,index)
                    record['ready_after_output_write']=True
                    publication_records.append(record)
                    publications.write(json.dumps(record,default=serial,allow_nan=False)+'\n');publications.flush()
                if getattr(args,'no_preview',False):
                    if index%30==0:print(index,round(t,4),row['structure'].get('mode'),flush=True)
                    continue
                # All panels are genuinely masked images, never raw+mask overlays.
                on,off=frame['on'],frame['off'];all_pixels=np.ones(on.shape,bool)
                masks=[all_pixels,frame['density'],frame['period_mask']>0,frame['native_mask']]
                images=[filtered_rgb(on,off,m) for m in masks]
                if index in (0,8,14,23):
                    for name,image in zip(['raw','density','period','native_arcs'],images):
                        cv2.imwrite(str(out/'images'/f'{index:04d}_{name}.png'),image[:,:,::-1])
                    np.savez_compressed(out/'images'/f'{index:04d}_evidence.npz',
                        on=on,off=off,density=frame['density'],period=frame['period_mask']>0,native_arcs=frame['native_mask'])
                panels=[]
                for title,image in zip(['Raw ON/OFF','Density only','Period pixels only','Native arcs guided by period'],images):
                    im=cv2.resize(image[:,:,::-1],(640,360));cv2.putText(im,title,(8,22),0,.55,(255,255,255),1)
                    panels.append(im)
                if row['valid'] and row['physical']:
                    path=np.array([row['coefficient']['position']]+[row['physical'][str(h)] for h in [.1,.2,.4]])/2
                    cv2.polylines(panels[3],[np.rint(path).astype(np.int32)],False,(0,255,255),1)
                    cv2.drawMarker(panels[3],tuple(np.rint(path[-1]).astype(int)),(0,255,255),cv2.MARKER_CROSS,8,1)
                frame_out=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])])
                cv2.putText(frame_out,f't={t:.3f} mode={row["structure"].get("mode")} valid={row["valid"]}',(8,705),0,.5,(0,255,255),1)
                if writer is None:writer=cv2.VideoWriter(str(out/'filtering.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
                writer.write(frame_out)
                if index%15==0:print(index,round(t,4),frame['counts'],row['structure'].get('mode'),flush=True)
    finally:
        gc_trace.stop()
        (out/'gc_trace.json').write_text(json.dumps(gc_trace.events,indent=2),'utf-8')
        stream.close()
        if hasattr(args,'_prepared_raw_source'):args._prepared_raw_source.close()
        if writer is not None:writer.release()
        frame=None;predictor.front.previous=None;predictor.front.geometry.count_current=None
        if predictor.period_worker is not None:predictor.period_worker.close()
    times=[r['compute_ms'] for r in rows];modes={}
    for r in rows:
        mode=r['structure'].get('mode','unknown');modes[mode]=modes.get(mode,0)+1
    ages=[p['source_age_s']*1000 for p in publication_records if p['source_age_s'] is not None]
    actual_valid=[p for p in publication_records if p['valid']]
    summary=dict(frames=len(rows),current_rotor_detections=sum(r['body']['detected'] for r in rows),
        valid_forecasts=sum(r['valid'] for r in rows),current_force_inputs=sum(r['force']['active'] for r in rows),
        modes=modes,compute_ms_median=float(np.median(times)) if times else None,
        compute_ms_p95=float(np.percentile(times,95)) if times else None,
        compute_capacity_fps=1000/np.mean(times) if times else None,
        bounded_prefetch=bool(getattr(args,'rotor_prefetch',False)),
        memory_replay=bool(getattr(args,'rotor_memory_cache',False)),preload_s=preload_s,preload_bytes=preload_bytes,
        prefetch_CPU_model_only_compute_ms=bool(getattr(args,'rotor_prefetch',False)),
        steady_pipeline_fps=(len(rows)-1)/(rows[-1]['output_wall_elapsed_s']-rows[0]['output_wall_elapsed_s']) if len(rows)>1 else None,
        steady_pipeline_p95_ms=float(np.percentile([r['pipeline_frame_ms'] for r in rows[1:]],95)) if len(rows)>1 else None,
        end_to_end_elapsed_s=time.perf_counter()-started,worker_and_model_startup_s=startup_s,whole_body_NN_used=False,
        decoder_prepare_s=decoder_prepare_s,total_call_elapsed_s=time.perf_counter()-call_started,
        source_time_predictions=True,actual_publication_30Hz_proven=False,
        paced_replay=bool(getattr(args,'paced',False)),
        native_packets_read=getattr(args,'_native_packets_read',len(rows)),
        skipped_observations=len(getattr(args,'_skipped_observations',[])),
        actual_output_count=len(publication_records),actual_valid_output_count=len(actual_valid),
        actual_valid_output_fraction=len(actual_valid)/len(publication_records) if publication_records else None,
        actual_output_age_ms_median=float(np.median(ages)) if ages else None,
        actual_output_age_ms_p95=float(np.percentile(ages,95)) if ages else None,
        actual_valid_output_rate_hz=len(actual_valid)/(publication_records[-1]['t']-publication_records[0]['t']) if len(publication_records)>1 else None,
        actual_output_is_one_per_processed_packet=True,
        labels_used=False,pixel_segmentation_precision_measured=False)
    (out/'skipped_observations.json').write_text(json.dumps(getattr(args,'_skipped_observations',[]),indent=2),'utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=serial),'utf-8')
    changed=[name for name in SOURCE_FILES if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sources[name]]
    (out/'protocol.json').write_text(json.dumps(dict(raw=getattr(args,'raw',None),cache=getattr(args,'cache',None),
        no_model_weights_read=not hasattr(predictor,'response_model_info'),
        body_NN_weights_read=False,native_response_model=getattr(predictor,'response_model_info',None),
        source_sha256=sources,source_time_horizon_s=.4,
        frequency_backend='device_native_sparse_front' if getattr(args,'rotor_gpu_front',False) else
            'existing_CUDA_harmonics_shared_worker' if getattr(args,'gpu_python',None) else 'CPU_reference',
        joint_rotor_force_observer=not getattr(args,'rotor_legacy_force',False),
        observed_unobserved_partition=not predictor.legacy_decomposition,
        sparse_device_front=bool(getattr(args,'rotor_gpu_front',False)),
        native_gpu=bool(getattr(args,'rotor_native_gpu',False)),fast_geometry=bool(getattr(args,'rotor_fast_geometry',False)),
        subpixel_motion=bool(getattr(args,'rotor_subpixel_motion',False)),
        motion_search_prior=bool(getattr(args,'rotor_motion_prior',False)),
        rotor_plane_motion=bool(getattr(args,'rotor_plane_motion',False)),
        coarse_native_motion_search=bool(getattr(args,'rotor_coarse_motion',False)),
        partial_entry_velocity_initialization=bool(getattr(args,'rotor_entry_init',False)),
        collective_regional_period_seeds=bool(getattr(args,'rotor_regional_seeds',False)),
        corrected_regional_graph_fallback=bool(getattr(args,'rotor_regional_graph',False)),
        projected_ellipse_FOV_guard=bool(getattr(args,'rotor_projected_fov_guard',False)),
        tangential_response_memory=bool(getattr(args,'rotor_tangential_memory',False)),
        differential_torque_innovation=bool(getattr(args,'rotor_torque_innovation',False)),
        inverse_normal_response=bool(getattr(args,'rotor_inverse_normal_response',False)),
        anisotropic_disk_response=bool(getattr(args,'rotor_anisotropic_response',False)),
        observed_force_rate=bool(getattr(args,'rotor_visible_force_rate',False)),
        current_strict_period_continuation=bool(getattr(args,'rotor_periodic_continuation',False)),
        expensive_curvature_search_hz=float(getattr(args,'rotor_curvature_hz',0.)),
        dense_front_materialized_in_worker=bool(getattr(args,'rotor_remote_front',False)),
        bounded_one_packet_prefetch=bool(getattr(args,'rotor_prefetch',False) and getattr(args,'rotor_prefetch_depth',1)==1),
        bounded_packet_prefetch_depth=getattr(args,'rotor_prefetch_depth',1) if getattr(args,'rotor_prefetch',False) else 0,
        native_feature_slots=getattr(args,'rotor_prefetch_depth',1)+2,
        raw_reader_chunk=getattr(args,'rotor_read_chunk',100000),raw_reader_buffers=getattr(args,'rotor_read_buffers',3),
        raw_reader_backend=getattr(args,'rotor_reader','rolling'),
        native_decode_shared_slots=2 if getattr(args,'rotor_reader','rolling').startswith('process-') else 0,
        native_front_consumes_all_events_before_slot_release=True,
        unused_host_event_copy_removed=bool(getattr(args,'rotor_native_gpu',False) and predictor.phase is None and
            getattr(args,'rotor_prefetch',False) and not getattr(args,'rotor_retain_host_events',False)),
        paced_RAW_replay=bool(getattr(args,'paced',False)),
        future_features_wait_for_source_release=bool(getattr(args,'paced',False)),
        decoder_may_preread_two_native_packets=True,
        decoder_cyclic_GC_deferred=bool(getattr(args,'defer_decoder_gc',False)),
        RAW_reader_prepared_before_GPU_warmup=bool(getattr(args,'rotor_reader_ready',False)),
        maximum_precompute_backlog_ms=max_backlog,
        skip_policy_does_not_claim_30_new_observations_per_second=True,
        previous_observed_ONOFF_maps_owned_outside_reusable_ring=True,
        relative_rotor_displacement_observer=bool(getattr(args,'rotor_relative_observation',False)),
        continuous_local_period_sources=bool(getattr(args,'rotor_track_local_source',False)),
        fragmented_relock_center_guard=bool(getattr(args,'rotor_relock_support_guard',False)),
        supported_border_interior_motion=bool(getattr(args,'rotor_border_motion',False)),
        preview_disabled=bool(getattr(args,'no_preview',False)),
        source_snapshot_before_worker_start=True,source_changed_during_run=changed,
        no_full_pipeline_real_time_claim=True),indent=2),'utf-8')
    if changed:raise RuntimeError('Source changed during inference; do not score this run: '+str(changed))
    print(json.dumps(summary),flush=True)


def make_parser():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--raw');g.add_argument('--cache');p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=2.)
    p.add_argument('--hz',type=float,default=30.)
    p.add_argument('--gpu-python',help='Optional existing CUDA Python interpreter for harmonic worker only; no detector weights')
    p.add_argument('--rotor-legacy-force',action='store_true',help='Control: original Singer state and delayed residual force model')
    p.add_argument('--rotor-legacy-decomposition',action='store_true',help='Stage161 control: hidden short priors and pooled absolute proxy')
    p.add_argument('--rotor-no-phase',action='store_true',help='Stage162 control: disable spatial phase arbitration')
    p.add_argument('--rotor-native-gpu',action='store_true',help='NVRTC native count/density/grouping/spectrum, no temporal sampling')
    p.add_argument('--rotor-fast-geometry',action='store_true',help='Use existing optimized geometry and exact native histogram')
    p.add_argument('--rotor-projected-fov-guard',action='store_true',help='Use rotated ellipse extents for every rotor; clipped conics do not qualify as complete plane evidence')
    p.add_argument('--rotor-tangential-memory',action='store_true',help='Experimental: learn conditional force memory from past motion perpendicular to the rotor normal; not calibrated aerodynamic drag')
    p.add_argument('--rotor-inverse-normal-response',action='store_true',help='Experimental: infer observable unknown normal-response amplitude from past tangent motion; current state unchanged, not calibrated thrust')
    p.add_argument('--rotor-anisotropic-response',action='store_true',help='Experimental disk tangent/normal response; requires inverse normal response; no metric drag identification')
    p.add_argument('--rotor-visible-force-rate',action='store_true',help='Experimental currently observed f_i^2 rate; no hidden rotor imputation')
    p.add_argument('--rotor-periodic-continuation',action='store_true',help='Current strict-period confirmed continuation; coverage can improve while accuracy regresses')
    p.add_argument('--rotor-torque-innovation',action='store_true',help='Experimental unequal visible-rotor force-change -> torque -> future normal; native slot inputs, no CPU author phase tracker; inertia is a prior')
    p.add_argument('--rotor-curvature-hz',type=float,default=0.,help='Optional1..30Hz expensive arc search; cheap current conics/axis every frame,0 preserves original')
    p.add_argument('--rotor-native-geometry-dll',help='Optional locally compiled206 exact arc seed/residual backend; old Python backend is default')
    p.add_argument('--rotor-prefetch',action='store_true',help='Bounded one-packet native decode/GPU prefetch; report actual pipeline timing')
    p.add_argument('--rotor-prefetch-depth',type=int,default=1,help='Bounded1..4 stateless feature packets; uses depth+2 dense slots')
    p.add_argument('--rotor-local-phase',action='store_true',help='Current per-pixel local temporal template evidence without shared plane')
    p.add_argument('--rotor-track-local-source',action='store_true',help='Experimental current local-region relative-period continuity; requires --rotor-local-phase')
    p.add_argument('--rotor-memory-cache',action='store_true',help='Bounded <=2GiB in-memory event replay; separately report preload time')
    p.add_argument('--rotor-read-chunk',type=int,default=100000,help='Native RAW decoder batching; source output window remains33333us')
    p.add_argument('--rotor-read-buffers',type=int,default=3,help='Native RAW read buffer count')
    p.add_argument('--rotor-subpixel-motion',action='store_true',help='Quadratic refinement of current rotor-region correlation peak')
    p.add_argument('--rotor-motion-prior',action='store_true',help='Current rotor search around past displacement, using actual elapsed source time')
    p.add_argument('--rotor-plane-motion',action='store_true',help='Shared current rotor-plane affine/side-line transform, qualified LK/RANSAC with translation fallback')
    p.add_argument('--rotor-coarse-motion',action='store_true',help='Coarse correlation discovery with native-resolution refinement and full-search fallback')
    p.add_argument('--rotor-entry-init',action='store_true',help='Two complete positions initialize velocity only after a recent provisional partial-entry track')
    p.add_argument('--rotor-regional-seeds',action='store_true',help='Collective held-out periodic evidence seeds native arcs when individual pixels are insufficient')
    p.add_argument('--rotor-regional-graph',action='store_true',help='Experimental corrected regional timing fallback; native GPU required; no motor RPM credit')
    p.add_argument('--rotor-remote-front',action='store_true',help='Move sparse-map materialization and native CCL into the CUDA service process using three shared dense slots')
    p.add_argument('--rotor-retain-host-events',action='store_true',help='Control: retain full CPU event copy after native GPU features even with spatial phase off')
    p.add_argument('--rotor-reader',choices=['rolling','iterator','cooperative','process-rolling','process-iterator'],default='rolling',help='Cooperative empty-HAL-buffer yielding, or existing bounded process decoder')
    p.add_argument('--rotor-gpu-front',action='store_true',help='Sparse device ON/OFF, density, period and native-support front')
    p.add_argument('--no-preview',action='store_true',help='Skip diagnostic image/video export; predictions and timings are still recorded')
    p.add_argument('--paced',action='store_true',help='Release GPU observations at30Hz source wall time; record actual output age and adjusted forecasts')
    p.add_argument('--defer-decoder-gc',action='store_true',help='Reuse finite-reader cyclic-GC deferral; native event buffers remain bounded and refcounted')
    p.add_argument('--rotor-reader-ready',action='store_true',help='Prepare first RAW packet before GPU warmup, preventing a long seek from cooling the backend')
    p.add_argument('--rotor-max-backlog-ms',type=float,default=0.,help='Optional paced latest-observation policy; skip and count stale packets before costly front/model work. Zero preserves no-drop baseline')
    p.add_argument('--rotor-relative-observation',action='store_true',help='Preserve previous/current position cross covariance; integrated rotor displacements are not independent absolute fixes')
    p.add_argument('--rotor-relock-support-guard',action='store_true',help='Do not turn a newly acquired small rotor fragment into a full center fix; retain current detection and explicit provisional state')
    p.add_argument('--rotor-border-motion',action='store_true',help='Match supported interior pixels when the old template touches the FOV; do not force out-of-sensor motion to zero')
    p.add_argument('--rotor-legacy-visibility',action='store_true',help='Control: old side shortcut and every15tick single-parent curvature split; no current bipolar-outline recovery')
    p.add_argument('--rotor-allow-invalid-motion-fallback',action='store_true',help='Legacy control: allow a rejected boundary shift to become an absolute position fix')
    p.add_argument('--rotor-gap-prediction',choices=['legacy','coherent','short'],help='Relative-observer gap prior: original, matched0.35s, or conservative matched0.1s; default short for current visibility, legacy for old control')
    p.add_argument('--rotor-unknown-force-model',choices=['free','plane','plane-separated','plane-adaptive'],default='free',help='Plane directional unknown force; plane-separated fast-normal/smooth-external rate; plane-adaptive gives it causal matured-error credit; free preserves176')
    p.add_argument('--rotor-reference-mode',choices=['legacy','fixed'],help='Current relative route defaults to fixed native output reference plus uncertain layout offset; explicit legacy restores centre replacement')
    return p

def main():run(make_parser().parse_args())


if __name__=='__main__':main()
