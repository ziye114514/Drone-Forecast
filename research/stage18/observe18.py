"""RAW-only incremental observer: current measurements, never coast as detections."""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np
from frontend18 import IncrementalFrontend,TrackPublisher,PARAMS,ROOT
from event_drone_forecast.raw_io import open_raw,raw_chunks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--publisher',choices=['strict','track'],default='strict');p.add_argument('--preview',action='store_true')
    a=p.parse_args()
    if min(a.start,a.duration)<0:p.error('start/duration must be nonnegative')
    if round(a.start*1e6)%20000 or round(a.duration*1e6)%20000:p.error('start/duration must be multiples of20ms')
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('frontend18.py'),ROOT/'research/stage17/frontend.py',ROOT/'research/stage9/recovery.py',ROOT/'research/stage8/polarity.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]
    fingerprints={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (output/'protocol.json').write_text(json.dumps(dict(arguments=vars(a),parameters=PARAMS,source_hashes=fingerprints,
        input='One RAW only. No labels or reference predictions in inference.',warmup='Up to100ms before requested start, counted in wall.',
        scope='Current image-position observations. No mechanical RPM or future predictions.'),indent=2),encoding='utf-8')
    began=time.perf_counter();reader=open_raw(a.raw);h,w=reader.get_size();model=IncrementalFrontend(w,h);gate=TrackPublisher()
    start=max(0.,a.start-.1);duration=a.duration+(a.start-start) if a.duration else 0.
    ticks=detected=0;times=[];video=None;last_print=began
    if a.preview:
        video=cv2.VideoWriter(str(output/'observations.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50.,(w,h))
        if not video.isOpened():raise RuntimeError('Cannot open MP4 writer')
    try:
        with (output/'observations.jsonl').open('x',encoding='utf-8') as stream:
            for t,e in raw_chunks(reader,start,duration,PARAMS['step_us']):
                # A final partial slice at EOF is not a valid20ms input.
                if round(t*1e6)%20000:break
                q=model.advance(e,t,t-start>=.1-1e-8);now=time.perf_counter();tracked=gate.apply(q['proposal'],t)
                publisher_ms=(time.perf_counter()-now)*1000
                o=tracked['observation'] if a.publisher=='track' else q['observation']
                if t<=a.start+1e-8:continue
                ticks+=1;detected+=o['detected'];times.append(q['timing_ms']['total']+publisher_ms)
                stream.write(json.dumps(dict(t=t,observation=o,track_status=tracked['status'],timing_ms=q['timing_ms'],publisher_ms=publisher_ms),allow_nan=False)+'\n')
                if video is not None:
                    pix=e['y'].astype(np.int32)*w+e['x'];count=np.bincount(pix,minlength=h*w).reshape(h,w)
                    grey=np.uint8(255-230*np.clip(np.log1p(count)/np.log(8),0,1));frame=cv2.cvtColor(grey,cv2.COLOR_GRAY2BGR)
                    if o['detected']:
                        b=np.rint(o['bbox']).astype(int);cv2.rectangle(frame,tuple(b[:2]),tuple(b[2:]),(30,130,230),2)
                        cv2.drawMarker(frame,tuple(np.rint(o['center']).astype(int)),(30,130,230),cv2.MARKER_CROSS,16,2)
                    cv2.putText(frame,f"{t:.2f}s | incremental {a.publisher} | published={o['detected']} | current measurement only",(20,h-25),cv2.FONT_HERSHEY_SIMPLEX,.65,(50,70,100),2)
                    video.write(frame)
                if time.perf_counter()-last_print>15:print(f't={t:.2f}s',flush=True);last_print=time.perf_counter()
    finally:
        if video is not None:video.release()
    summary=dict(frames=ticks,detected=detected,wall_including_open_seek_decode_write_preview_s=time.perf_counter()-began,
        frontend_median_ms=float(np.median(times)) if times else None,frontend_p95_ms=float(np.percentile(times,95)) if times else None,
        source_duration_s=a.duration or None,preview=a.preview,source_unchanged=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==value for f,value in fingerprints.items()),
        scope='No future predictions. Video50fps is source timeline, not processing throughput.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
