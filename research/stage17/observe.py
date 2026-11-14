"""RAW-only stage17 observer. Current observations, no future predictions or RPM."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import cv2,numpy as np
from frontend import Frontend,PARAMS,ROOT
from event_drone_forecast.raw_io import open_raw,raw_chunks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--policy',choices=['global','roi'],default='roi');p.add_argument('--preview',action='store_true')
    a=p.parse_args()
    if a.start<0 or a.duration<0:p.error('start/duration must be nonnegative')
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False)
    source={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [
        Path(__file__),Path(__file__).with_name('frontend.py'),ROOT/'research/stage9/recovery.py',
        ROOT/'research/stage8/polarity.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]}
    (output/'protocol.json').write_text(json.dumps(dict(raw=str(Path(a.raw).resolve()),arguments=vars(a),parameters=PARAMS,
        source_hashes=source,labels_in_inference=False,scope='Current geometric observations only; no mechanical RPM or future trajectory.',
        warmup='At most100ms prior to start; causal and counted in wall time.'),indent=2),encoding='utf-8')
    began=time.perf_counter();reader=open_raw(a.raw);h,w=reader.get_size();model=Frontend(w,h,a.policy)
    start=max(0.,a.start-.1);duration=a.duration+(a.start-start) if a.duration else 0.
    history=collections.deque();ticks=detected=0;timings=[];video=None;last_print=began
    if a.preview:
        video=cv2.VideoWriter(str(output/'observations.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50.,(w,h))
        if not video.isOpened():raise RuntimeError('Cannot open MP4 writer')
    try:
        with (output/'observations.jsonl').open('x',encoding='utf-8') as f:
            for t,e in raw_chunks(reader,start,duration,PARAMS['step_us']):
                history.append((t,e))
                while history and history[0][0]<=t-.1+1e-8:history.popleft()
                warming=t-start<.1-1e-8
                if warming:
                    q=dict(observation=dict(t=t,detected=False,rotors=[]),search='warming',candidates=[])
                else:
                    q=model.step(np.concatenate([e for _,e in history]),e,t);timings.append(q['timing_ms']['total'])
                if t<=a.start+1e-8:continue
                ticks+=1;detected+=q['observation']['detected'];f.write(json.dumps(dict(t=t,**q),allow_nan=False)+'\n')
                if video is not None:
                    pixels=e['y'].astype(np.int32)*w+e['x'];count=np.bincount(pixels,minlength=h*w).reshape(h,w)
                    grey=np.uint8(255-230*np.clip(np.log1p(count)/np.log(8),0,1));frame=cv2.cvtColor(grey,cv2.COLOR_GRAY2BGR)
                    for r in q['candidates']:
                        cv2.ellipse(frame,tuple(np.rint([r['cx'],r['cy']]).astype(int)),
                            tuple(np.maximum(1,np.rint([r['major']/2,r['minor']/2])).astype(int)),r['angle'],0,360,(140,150,20),2)
                    o=q['observation']
                    if o['detected']:
                        box=np.rint(o['bbox']).astype(int);cv2.rectangle(frame,tuple(box[:2]),tuple(box[2:]),(20,100,220),2)
                    cv2.putText(frame,f"t={t:.2f}s | {q['search']} | published={o['detected']} | detection only",(20,h-25),cv2.FONT_HERSHEY_SIMPLEX,.7,(60,70,90),2)
                    video.write(frame)
                if time.perf_counter()-last_print>15:print(f't={t:.2f}s',flush=True);last_print=time.perf_counter()
    finally:
        if video is not None:video.release()
    summary=dict(frames=ticks,detected=detected,wall_including_open_seek_decode_write_s=time.perf_counter()-began,
        source_duration_s=a.duration or None,median_frontend_ms=float(np.median(timings)) if timings else None,
        p95_frontend_ms=float(np.percentile(timings,95)) if timings else None,
        source_unchanged=all(hashlib.sha256((ROOT/q).read_bytes()).hexdigest()==v for q,v in source.items()),
        preview=a.preview,scope='Current observations only. Video50fps reflects RAW timestamp spacing, not processing throughput.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
