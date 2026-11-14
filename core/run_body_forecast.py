"""RAW-only local whole-UAV/rotor fusion and image trajectory research runner.

SDK311 decodes causal events, a persistent installed Python313 GPU process
detects each current ON/OFF frame, six unchanged history models forecast.
No image manifests, cached detections, evaluation labels or future input.
"""
from pathlib import Path
import argparse,base64,hashlib,importlib.util,json,subprocess,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage21'),str(ROOT/'research/stage23')]
from windows21 import Observer
from robust23 import Predictor,MODELS
from event_drone_forecast.raw_io import open_raw,raw_chunks
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_core import BaseFrameGenerationAlgorithm
METHODS=['geometry','body_only','geometry_first','body_first']

def json_text(x):return json.dumps(x,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
def fingerprints():
    paths=[Path(__file__),ROOT/'research/stage42/worker42.py',ROOT/'research/stage21/windows21.py',ROOT/'research/stage23/robust23.py',ROOT/'research/stage22/predict22.py',ROOT/'research/stage14/motion.py',ROOT/'research/stage18/frontend18.py',ROOT/'research/stage17/frontend.py',ROOT/'research/stage9/recovery.py',ROOT/'research/stage8/polarity.py',*(ROOT/'src/event_drone_forecast').glob('*.py')]
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def run(raw,output,start=0.,duration=0.,primary='body_first',preview=False,infer_python='C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe'):
    raw=Path(raw).resolve();output=Path(output).resolve();protected=Path('D:/yolov10-main').resolve();assert not raw.is_relative_to(protected) and not output.is_relative_to(protected)
    assert start>=0 and duration>=0 and round(start*1e6)%20000==0 and round(duration*1e6)%20000==0 and primary in METHODS
    output.mkdir(parents=True,exist_ok=False);sources=fingerprints();began=time.perf_counter();video=None;log=(output/'gpu_worker.log').open('x',encoding='utf-8');worker=None
    reader=open_raw(raw);h,w=reader.get_size();assert (h,w)==(720,1280);front=Observer(w,h,'native20');noise=ActivityNoiseFilterAlgorithm(w,h,10000);buf=noise.get_empty_output_buffer();models={m:Predictor() for m in METHODS};count={m:0 for m in METHODS};valid={m:0 for m in METHODS};timings=[];n=0
    try:
        worker=subprocess.Popen([str(infer_python),str(ROOT/'research/stage42/worker42.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        ready=json.loads(worker.stdout.readline());assert ready['ready'];protocol=dict(raw=str(raw),raw_size_bytes=raw.stat().st_size,start=start,duration=duration,step_us=20000,primary=primary,source_hashes=sources,worker=ready,
            inputs='RAW only plus certified local learned weight. No GT/cache/manifest/future detections. Per-frame local PNG pipe, independent eval mode batch1.',
            scope='Current visible whole-body box + original rotor geometry; predictions use six causal history models only. No learned pose or thrust implied by semantic box. Offline source timestamps; actual processing latency measured.')
        (output/'protocol.json').write_text(json_text(protocol),encoding='utf-8')
        if preview:
            video=cv2.VideoWriter(str(output/'forecasts.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50.,(w,h));assert video.isOpened()
        begin=max(0.,start-.1);length=duration+start-begin if duration else 0.
        with (output/'observations_forecasts.jsonl').open('x',encoding='utf-8') as f:
            for t,e in raw_chunks(reader,begin,length,20000):
                tick=time.perf_counter();geometry=front.step(e,t)['observation'] if t-begin>=.1-1e-8 else dict(t=t,detected=False,rotors=[])
                noise.process_events(e,buf);clean=buf.numpy().copy()
                if t<=start+1e-8:continue
                frame=np.empty((h,w,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(clean,frame);ok,png=cv2.imencode('.png',frame);assert ok
                worker.stdin.write(json.dumps(dict(id=n,png=base64.b64encode(png.tobytes()).decode('ascii')))+'\n');worker.stdin.flush();line=worker.stdout.readline()
                if not line:raise RuntimeError('GPU worker stopped; see gpu_worker.log')
                result=json.loads(line);assert result['id']==n
                if result['boxes']:
                    best=max(result['boxes'],key=lambda b:b['confidence']);b=best['bbox'];body=dict(t=t,detected=True,center=[(b[0]+b[2])/2,(b[1]+b[3])/2],bbox=b,rotors=[],confidence=best['confidence'],source='body')
                else:body=dict(t=t,detected=False,center=None,rotors=[],source='body')
                observations=dict(geometry=geometry,body_only=body,geometry_first=geometry if geometry['detected'] else body,body_first=body if body['detected'] else geometry);forecasts={}
                for method,o in observations.items():
                    forecasts[method],_=models[method].step(t,o['center'] if o['detected'] else None);count[method]+=o['detected'];valid[method]+=forecasts[method]['valid']
                elapsed=(time.perf_counter()-tick)*1000;timings.append(elapsed);row=dict(t=t,event_count=len(e),filtered_event_count=len(clean),observations=observations,forecasts=forecasts,body_boxes=result['boxes'],gpu_inference_ms=result['inference_ms'],processing_ms=elapsed)
                f.write(json_text(row)+'\n');n+=1
                if video is not None:
                    for method,color in [('geometry',(220,160,70)),(primary,(30,170,255))]:
                        o=observations[method];q=forecasts[method]
                        if o['detected']:
                            box=np.rint(o['bbox']).astype(int);cv2.rectangle(frame,tuple(box[:2]),tuple(box[2:]),color,2)
                        if q['valid']:
                            points=[q['coefficients']['cv']['position']]+[q['predictions']['cv'][str(h)] for h in [.1,.2,.4]]
                            for a,b in zip(points,points[1:]):cv2.line(frame,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),color,2)
                    cv2.putText(frame,f'{t:.2f}s | blue=rotors CV orange={primary} CV | {elapsed:.1f}ms',(16,h-20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);video.write(frame)
        summary=dict(frames=n,published=count,valid_forecasts=valid,processing_median_ms=float(np.median(timings)),processing_p95_ms=float(np.percentile(timings,95)),wall_including_model_load_decode_video_s=time.perf_counter()-began,source_hashes_unchanged=fingerprints()==sources,labels_read=False,worker_batch=1)
        (output/'completion.json').write_text(json_text(summary),encoding='utf-8');print(json_text(summary),flush=True);return summary
    finally:
        if video is not None:video.release()
        if worker is not None:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
                try:worker.wait(timeout=10)
                except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=5)
        log.close()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.);p.add_argument('--primary',choices=METHODS,default='body_first');p.add_argument('--preview',action='store_true');a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.primary,a.preview)
