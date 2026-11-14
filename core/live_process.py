"""SDK/RAW producer in its own process, with a draining latest-packet listener.

Only selected current filtered event arrays cross the pipe. Every native RAW
batch still updates SDK noise/optional periodic state. The listener drains the
pipe even while inference runs; the estimator sees one overwriteable slot.
"""
from multiprocessing import get_context
from dataclasses import replace
from pathlib import Path
from threading import Condition, Thread
import time
import traceback
from live_stream import RateGate, Packet,packet_expired
from live_startup import wait_for_release


def _producer(connection, stop, raw, start, duration, tracker_hz, periodic_axis,
              frame_spec=None,frame_locks=None,frame_generations=None,preview_window_ms=5.,
              verify_frames=False,trace_hashes=False,start_signal=None,reader_index='default',maximum_packet_age=None,periodic_sampling='full',periodic_packet='full'):
    frames=None
    try:
        import sys
        import numpy as np
        import hashlib
        root=Path(__file__).resolve().parent
        sys.path.insert(0,str(root/'src'))
        from live_decode import native_batches as iter_native_batches
        from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
        if frame_spec is not None:
            from live_frames import SharedFrames
            frames=SharedFrames.attach(frame_spec,frame_locks,frame_generations)
        reference_image=reference_preview=None
        if trace_hashes and frames is None:
            from metavision_sdk_core import BaseFrameGenerationAlgorithm
            reference_image=np.zeros((720,1280,3),np.uint8);reference_preview=np.zeros_like(reference_image)
        axis=None
        if periodic_axis:
            sys.path.insert(0,str(root/'research/stage141'))
            if periodic_sampling=='fixed-quarter':
                from live_pixel_axis import SDKFixedPixelAxis as NativeAxis
            else:from native_axis141 import NativeAxis
        h,w=720,1280
        noise=ActivityNoiseFilterAlgorithm(w,h,10000);buffer=noise.get_empty_output_buffer()
        if periodic_axis:axis=NativeAxis(w,h)
        begin=max(0.,start-.1);gate=RateGate(start,tracker_hz)
        epoch=None;native_batches=native_events=produced=0;max_lag=0.
        costs={name:[] for name in ['reader','noise','periodic','periodic_compact','copy','frame_generation','send']};payload_bytes=[]
        packet_input_records=packet_output_records=0
        if periodic_packet=='latest-pixel':from live_periodic_packet import compact_current
        trace=[];frame_pool_busy_skipped=0;stale_before_generation=0
        previous_end=time.perf_counter();maximum_packet_ready_lag=0.
        for t,events in iter_native_batches(raw,begin,duration+start-begin,build_index=reader_index!='disabled'):
            now=time.perf_counter();costs['reader'].append((now-previous_end)*1000)
            if stop.is_set():break
            if t>start+1e-8:
                if epoch is None:
                    if not wait_for_release(connection,start_signal,stop):break
                    epoch=time.perf_counter()
                    connection.send(('ready',epoch))
                if stop.wait(max(0.,epoch+t-start-time.perf_counter())):break
            stage=time.perf_counter();noise.process_events(events,buffer);costs['noise'].append((time.perf_counter()-stage)*1000)
            stage=time.perf_counter()
            q=axis.feed(events,t) if axis is not None else None
            costs['periodic'].append((time.perf_counter()-stage)*1000)
            native_batches+=1;native_events+=len(events)
            if epoch is not None:max_lag=max(max_lag,time.perf_counter()-epoch-(t-start))
            if t<=start+1e-8 or not gate.take(t):
                previous_end=time.perf_counter();continue
            if packet_expired(t,start,epoch,time.perf_counter(),maximum_packet_age):
                # The consumer necessarily receives this later and applies
                # the same age limit. Native SDK state above is still updated.
                stale_before_generation+=1;previous_end=time.perf_counter();continue
            stage=time.perf_counter()
            clean=buffer.numpy() if frames is not None else buffer.numpy().copy()
            if q is not None:
                packet_input_records+=len(q)
                if periodic_packet=='latest-pixel':
                    compact_start=time.perf_counter();q=compact_current(q,t,w)
                    costs['periodic_compact'].append((time.perf_counter()-compact_start)*1000)
                else:q=q.copy()
                packet_output_records+=len(q)
            costs['copy'].append((time.perf_counter()-stage)*1000)
            descriptor=None
            if frames is not None:
                stage=time.perf_counter()
                descriptor=frames.write_events(clean,t,preview_window_ms,verify_frames or trace_hashes)
                costs['frame_generation'].append((time.perf_counter()-stage)*1000)
                if descriptor is None:
                    frame_pool_busy_skipped+=1;previous_end=time.perf_counter();continue
            if trace_hashes:
                if frames is None:
                    stage=time.perf_counter();BaseFrameGenerationAlgorithm.generate_frame(clean,reference_image)
                    BaseFrameGenerationAlgorithm.generate_frame(clean[clean['t']>=round(t*1e6-preview_window_ms*1000)],reference_preview)
                    a=hashlib.sha256(reference_image.tobytes()).hexdigest();b=hashlib.sha256(reference_preview.tobytes()).hexdigest()
                    costs['frame_generation'].append((time.perf_counter()-stage)*1000)
                else:a=descriptor['detector_sha256'];b=descriptor['preview_sha256']
                trace.append(dict(t=t,clean_events=len(clean),detector_sha256=a,preview_sha256=b,
                                  raw_max_event_t=int(events['t'].max()) if len(events) else None))
            payload_bytes.append((clean.nbytes if frames is None else 0)+(q.nbytes if q is not None else 0))
            payload=dict(t=t,clean=clean if frames is None else None,frame_descriptor=descriptor,periodic=q,available_wall=time.perf_counter(),
                         event_count=len(events),max_event_t=int(events['t'].max()) if len(events) else None)
            maximum_packet_ready_lag=max(maximum_packet_ready_lag,payload['available_wall']-epoch-(t-start))
            stage=time.perf_counter();connection.send(('packet',payload));costs['send'].append((time.perf_counter()-stage)*1000);produced+=1
            previous_end=time.perf_counter()
        profile={name:dict(n=len(values),median_ms=float(np.median(values)) if values else None,
                          p95_ms=float(np.percentile(values,95)) if values else None,
                          maximum_ms=max(values) if values else None) for name,values in costs.items()}
        if len(costs['reader'])>1:profile['reader_after_first_seek']=dict(n=len(costs['reader'])-1,
            median_ms=float(np.median(costs['reader'][1:])),p95_ms=float(np.percentile(costs['reader'][1:],95)),maximum_ms=max(costs['reader'][1:]))
        profile['payload_bytes']=dict(total=sum(payload_bytes),median=float(np.median(payload_bytes)) if payload_bytes else None,
                                     maximum=max(payload_bytes) if payload_bytes else None)
        profile['maximum_packet_ready_lag_ms']=maximum_packet_ready_lag*1000
        profile['array_payload_counter_scope']='Array bytes through pipe before pickle overhead; shared image writes excluded'
        sampling_stats=None
        if periodic_axis:
            sampling_stats=dict(mode=periodic_sampling,native_input_events=native_events,
                frequency_input_events=getattr(axis,'selected_events',native_events),
                mask_sha256=hashlib.sha256(axis.mask.tobytes()).hexdigest() if hasattr(axis,'mask') else None)
        connection.send(('done',dict(native_batches=native_batches,native_events=native_events,
                                    produced=produced,max_producer_lag_s=max_lag,producer_profile=profile,
                                    producer_frame_trace=trace,frame_pool_busy_skipped=frame_pool_busy_skipped,
                                    stale_before_generation=stale_before_generation,periodic_sampling_stats=sampling_stats,
                                    periodic_packet_stats=dict(mode=periodic_packet,input_records=packet_input_records,transport_records=packet_output_records))))
    except BaseException:
        try:connection.send(('error',traceback.format_exc()))
        except (BrokenPipeError,EOFError,OSError):pass
    finally:
        if frames is not None:frames.close()
        connection.close()


class ProcessReplay:
    def __init__(self,raw,start,duration,tracker_hz,periodic_axis=False,frame_transport='events',
                 preview=False,preview_window_ms=5.,verify_frames=False,frame_destination=None,trace_hashes=False,
                 defer_start=False,reader_index='default',maximum_packet_age=None,periodic_sampling='full',periodic_packet='full'):
        if frame_transport not in ('events','frames'):raise ValueError('Unknown process transport')
        if reader_index not in ('default','disabled'):raise ValueError('Unknown RAW index mode')
        if maximum_packet_age is not None and (not 0<maximum_packet_age<float('inf')):
            raise ValueError('Packet age limit must be finite and positive')
        if periodic_sampling not in ('full','fixed-quarter'):raise ValueError('Unknown periodic sampling')
        if periodic_packet not in ('full','latest-pixel') or (periodic_packet!='full' and not periodic_axis):
            raise ValueError('Current periodic packet compression requires periodic axis')
        self.start=start;self.epoch=None;self.slot=None;self.error=None;self.done=False
        self.condition=Condition();self.consumed=self.overwritten=self.produced=0
        self.native_batches=self.native_events=0;self.max_producer_lag_s=0.
        self.producer_profile={};self.transport_lags_ms=[]
        self.periodic_sampling_stats=None
        self.periodic_packet_stats=None
        self.producer_frame_trace=[];self.frame_pool_busy_skipped=0;self.descriptor_mismatches=0;self.stale_before_generation=0
        self.frames=None;self.frame_destination=frame_destination;self.closed=False
        self.context=get_context('spawn');self.stop=self.context.Event()
        self.start_signal=self.context.Event() if defer_start else None
        self.prepared=False;self.start_released=False
        self.connection,child=self.context.Pipe(duplex=False)
        if frame_transport=='frames':
            from live_frames import SharedFrames
            self.frames=SharedFrames(self.context,preview=preview,preview_window_ms=preview_window_ms)
        self.process=self.context.Process(target=_producer,args=(child,self.stop,str(raw),start,duration,tracker_hz,periodic_axis,
            self.frames.spec() if self.frames else None,self.frames.locks if self.frames else None,
            self.frames.generations if self.frames else None,preview_window_ms,verify_frames,trace_hashes,self.start_signal,reader_index,maximum_packet_age,periodic_sampling,periodic_packet),name='raw-sdk-producer')
        try:self.process.start()
        except BaseException:
            child.close();self.connection.close()
            if self.frames is not None:self.frames.close()
            raise
        child.close()
        self.thread=Thread(target=self._listen,name='raw-latest-pipe')
        self.thread.start()

    def _listen(self):
        try:
            while True:
                kind,data=self.connection.recv()
                received_wall=time.perf_counter()
                with self.condition:
                    if kind=='prepared':self.prepared=True
                    elif kind=='ready':self.epoch=data
                    elif kind=='packet':
                        lag=max(0.,(received_wall-data['available_wall'])*1000);self.transport_lags_ms.append(lag)
                        if self.slot is not None:self.overwritten+=1
                        self.slot=Packet(data['t'],None,data['clean'],data['periodic'],data['available_wall'],
                                         data['event_count'],data['max_event_t'],lag,data.get('frame_descriptor'))
                    elif kind=='error':
                        self.error=RuntimeError('RAW producer failed:\n'+data);break
                    elif kind=='done':
                        for key,value in data.items():setattr(self,key,value)
                        break
                    self.condition.notify_all()
        except (EOFError,OSError) as error:
            if not self.stop.is_set():self.error=RuntimeError('RAW producer pipe closed before completion: '+str(error))
        finally:
            with self.condition:self.done=True;self.condition.notify_all()

    def __iter__(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.slot is not None or self.done)
                if self.slot is None:
                    if self.error is not None:raise self.error
                    return
                packet,self.slot=self.slot,None
            if packet.frame_descriptor is not None:
                images=self.frames.read(packet.frame_descriptor,self.frame_destination)
                if images is None:
                    self.descriptor_mismatches+=1;continue
                frame,preview=images;desc=packet.frame_descriptor
                packet=replace(packet,frame=frame,preview_frame=preview,frame_sha256=desc['detector_sha256'],preview_sha256=desc['preview_sha256'])
            self.consumed+=1
            yield packet

    def source_now(self):
        return self.start if self.epoch is None else self.start+time.perf_counter()-self.epoch

    def wait_prepared(self,timeout=120.):
        if self.start_signal is None:raise ValueError('Startup synchronization was not requested')
        with self.condition:
            ready=self.condition.wait_for(lambda:self.prepared or self.done or self.error is not None,timeout)
            if self.error is not None:raise self.error
            if not ready:raise TimeoutError('RAW reader preparation timed out; replay has not started')
            if not self.prepared:raise RuntimeError('RAW ended before the requested replay start')

    def release_start(self):
        if self.start_signal is None:raise ValueError('Startup synchronization was not requested')
        with self.condition:
            if not self.prepared or self.done:raise RuntimeError('RAW reader is not waiting at the startup barrier')
            if self.start_released:raise RuntimeError('RAW replay startup was already released')
            self.start_released=True
        self.start_signal.set()

    def close(self):
        if self.closed:return
        self.closed=True
        self.stop.set()
        self.process.join(timeout=15)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=5)
        self.thread.join(timeout=5)
        self.connection.close()
        if self.thread.is_alive():raise RuntimeError('RAW listener did not stop')
        if self.frames is not None:self.frames.close()
