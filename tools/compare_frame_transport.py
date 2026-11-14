"""Compare EVERY produced native detector/preview image, independent of drops.

Events path copies SDK events and renders the canonical images for the audit;
frames path renders directly in shared slots. Audit rendering/SHAs add cost,
so this is NOT a performance benchmark. There is no NN, label or output input.
"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_process import ProcessReplay


def main():
    out=ROOT.parent/'output/shared_frame_transport_20261025';out.mkdir(exist_ok=False)
    result={}
    for transport in ['events','frames']:
        replay=ProcessReplay('E:/dataset/train/6/media/event.raw',110.,1.2,25.,
            frame_transport=transport,preview=True,preview_window_ms=5.,verify_frames=True,trace_hashes=True)
        consumed=0
        try:
            for packet in replay:
                assert packet.t<=replay.source_now()+1e-6
                assert packet.max_event_t<round(packet.t*1e6)
                if transport=='frames':
                    assert packet.frame is not None and packet.clean is None
                    assert hashlib.sha256(packet.frame.tobytes()).hexdigest()==packet.frame_sha256
                    assert hashlib.sha256(packet.preview_frame.tobytes()).hexdigest()==packet.preview_sha256
                consumed+=1
        finally:replay.close()
        assert replay.native_batches==65 and replay.produced==31
        result[transport]=dict(trace=replay.producer_frame_trace,consumed=consumed,
            native_events=replay.native_events,profile=replay.producer_profile,
            descriptor_mismatches=replay.descriptor_mismatches,busy_skips=replay.frame_pool_busy_skipped)
        (out/(transport+'.json')).write_text(json.dumps(result[transport],indent=2),encoding='utf-8')
    assert result['events']['trace']==result['frames']['trace']
    assert result['events']['native_events']==result['frames']['native_events']==16981921
    summary=dict(detector_images_exact=31,preview5ms_images_exact=31,native_events=16981921,
        native_batches=65,no_future_packet=True,comparison='Every produced frame, not only consumed intersection',
        trace_instrumentation_is_not_realtime_benchmark=True)
    (out/'equivalence.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
