"""RAW-only body20 + isolated rotor10 + fixed-reference delayed physical learning."""
from pathlib import Path
import argparse,copy,hashlib,json,os,subprocess,sys,time
from multiprocessing import shared_memory
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage21'),str(ROOT/'research/stage23'),str(ROOT/'research/stage51'),str(ROOT/'research/stage52'),str(ROOT/'research/stage29')]
from windows21 import Observer
from robust23 import Predictor as HistoryPredictor,MODELS as HISTORY
from reference51 import Managed,associate,observation
from feedback52 import Managed52,update_source
from select29 import Selector,MODELS
from event_drone_forecast.raw_io import open_raw,raw_chunks
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_core import BaseFrameGenerationAlgorithm
METHODS=['reset_all','stable_reference','delayed_feedback']

def json_text(x):return json.dumps(x,ensure_ascii=False,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())
def fingerprints():
    paths=[Path(__file__),ROOT/'research/stage53/rotor_worker53.py',ROOT/'research/stage47/worker47.py',ROOT/'research/stage45/graph45.py',ROOT/'research/stage44/common44.py',ROOT/'research/stage41/common41.py',ROOT/'research/stage40/common40.py',ROOT/'research/stage37/common37.py']
    for name in ['stage21/windows21.py','stage23/robust23.py','stage22/predict22.py','stage14/motion.py','stage18/frontend18.py','stage17/frontend.py','stage9/recovery.py','stage8/polarity.py','stage51/reference51.py','stage52/feedback52.py','stage48/compose48.py','stage28/axis28.py','stage27/physics27.py','stage29/select29.py','stage33/slots33.py']:paths.append(ROOT/'research'/name)
    paths.extend((ROOT/'src/event_drone_forecast').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def send(worker,message):worker.stdin.write(json_text(message)+'\n');worker.stdin.flush()
def receive(worker,kind):
    line=worker.stdout.readline()
    if not line:raise RuntimeError(kind+' worker stopped; inspect its log')
    return json.loads(line)
def stop(worker):
    if worker is None:return
    if worker.poll() is None:
        try:send(worker,dict(stop=True))
        except (BrokenPipeError,OSError):pass
        try:worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:worker.wait(timeout=5)
            except subprocess.TimeoutExpired:worker.kill();worker.wait(timeout=5)
    for stream in [worker.stdin,worker.stdout]:
        if stream is not None:stream.close()

def run(raw,output,start=0.,duration=0.,precision=32,preview=False,verify_frames=False,infer_python='C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe'):
    raw=Path(raw).resolve();output=Path(output).resolve();assert raw.suffix.lower()=='.raw' and raw.is_file()
    assert start>=0 and duration>=0 and round(start*1e6)%20000==0 and round(duration*1e6)%20000==0 and precision in [16,32]
    assert not raw.is_relative_to(Path('D:/yolov10-main')) and not output.is_relative_to(Path('D:/yolov10-main'))
    output.mkdir(parents=True,exist_ok=False)
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode=args[1] if len(args)>1 else None;writing=isinstance(mode,str) and any(x in mode for x in ['w','a','x','+'])
            if not writing and p.is_relative_to(ROOT.parent/'output') and not p.is_relative_to(output) and p.suffix.lower() in ['.json','.jsonl','.png','.mp4']:
                raise RuntimeError('RAW entry cache/truth input forbidden '+str(p))
            if p.is_relative_to(ROOT.parent/'output/raw_only_project/evaluation_data'):raise RuntimeError('Truth input forbidden')
    sys.addaudithook(guard);began=time.perf_counter();hashes=fingerprints();reader=open_raw(raw);h,w=reader.get_size();assert (h,w)==(720,1280)
    front=Observer(w,h,'native20');noise=ActivityNoiseFilterAlgorithm(w,h,10000);buffer=noise.get_empty_output_buffer();history=HistoryPredictor()
    models={m:Managed52() if m=='delayed_feedback' else Managed(m) for m in METHODS};selectors={m:Selector() for m in METHODS}
    memory=None;frame=None;video=None;rotor=None;gpu=None;logs=[];stats={m:dict(physics_active=0,actual_updates={name:0 for name in models[m].predictor.models}) for m in METHODS};timings=[];worker_times=[];rotor_times=[];n=detected=valid=0;last_print=began;summary=None
    try:
        rlog=(output/'rotor_worker.log').open('x',encoding='utf-8');logs.append(rlog)
        rotor=subprocess.Popen([sys.executable,str(ROOT/'research/stage53/rotor_worker53.py'),'--raw',str(raw),'--output',str(output),'--start',str(start),'--duration',str(duration)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=rlog,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        rotor_ready=receive(rotor,'Rotor');assert rotor_ready['ready'] and (rotor_ready['height'],rotor_ready['width'])==(h,w)
        memory=shared_memory.SharedMemory(create=True,size=h*w*3);frame=np.ndarray((h,w,3),np.uint8,buffer=memory.buf)
        glog=(output/'gpu_worker.log').open('x',encoding='utf-8');logs.append(glog)
        gpu=subprocess.Popen([str(infer_python),str(ROOT/'research/stage47/worker47.py'),'--shared-memory',memory.name,'--precision',str(precision)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=glog,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        gpu_ready=receive(gpu,'GPU');assert gpu_ready['ready']
        protocol=dict(raw=str(raw),raw_size_bytes=raw.stat().st_size,raw_mtime_ns=raw.stat().st_mtime_ns,start=start,duration=duration,step_us=20000,rotor_step_us=10000,precision=precision,
            source_hashes=hashes,rotor_worker=rotor_ready,gpu_worker=gpu_ready,derived_update_sha256=hashlib.sha256(update_source().encode('utf-8')).hexdigest(),methods=METHODS,
            inputs='RAW plus fixed local detector weights only. Two independent bounded RAW readers: body20 and rotor10. No cached detections/sensing/manifests or evaluation truth. Both workers process only requested current cutoff.',
            scope='Actual ellipse/frequency features + body-supported position enter physical models. Effective pixel dynamics, not true mass/wind/RPM/3D truth. Old reset and stable controls saved alongside delayed feedback; all six historical coefficients exact to independent history filter on every tick.',
            timing='Processing includes current native20 geometry/noise/worker waits and physics, excludes parent next RAW block, JSON and preview. Workers may overlap. Wall includes both worker cold starts, duplicate RAW decoding and video.50fps video is source time, not processing throughput.')
        (output/'protocol.json').write_text(json_text(protocol),encoding='utf-8')
        if preview:video=cv2.VideoWriter(str(output/'forecasts.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50.,(w,h));assert video.isOpened()
        begin=max(0.,start-.1);length=duration+start-begin if duration else 0.
        with (output/'observations_forecasts.jsonl').open('x',encoding='utf-8') as stream:
            for t,e in raw_chunks(reader,begin,length,20000):
                tick=time.perf_counter();geometry=front.step(e,t)['observation'] if t-begin>=.1-1e-8 else dict(t=t,detected=False,rotors=[])
                noise.process_events(e,buffer);clean=buffer.numpy().copy()
                if t<=start+1e-8:continue
                BaseFrameGenerationAlgorithm.generate_frame(clean,frame);frame_sha=hashlib.sha256(frame.tobytes()).hexdigest() if verify_frames else None
                send(rotor,dict(id=n,cutoff=t));send(gpu,dict(id=n,verify_frame=verify_frames))
                result=receive(gpu,'GPU');rotor_result=receive(rotor,'Rotor');assert result['id']==rotor_result['id']==n and abs(rotor_result['cutoff']-t)<1e-8
                assert not verify_frames or result['frame_sha256']==frame_sha
                if result['boxes']:
                    best=max(result['boxes'],key=lambda b:b['confidence']);b=best['bbox'];body=dict(t=t,detected=True,center=[(b[0]+b[2])/2,(b[1]+b[3])/2],bbox=b,rotors=[],confidence=best['confidence'],source='body')
                else:body=dict(t=t,detected=False,center=None,rotors=[],source='body')
                position=body if body['detected'] else geometry;obs=observation(position,rotor_result['record']['observation']);sensed,audit=associate(rotor_result['record']['sensing'],position);epoch=rotor_result['record']['slot_audit']['calibration_epoch']
                original_history,_=history.step(t,position['center'] if position.get('detected') else None);forecasts={};diagnostics={}
                for method in METHODS:
                    q,d=models[method].step(t,obs,sensed,epoch);q=selectors[method].step(q,obs);q['association_diagnostic']=audit
                    assert q['valid']==original_history['valid']
                    for name in HISTORY:assert q['coefficients'].get(name)==original_history['coefficients'].get(name),(t,method,name)
                    forecasts[method]=q;diagnostics[method]=d;stats[method]['physics_active']+=q['feature_diagnostic']['active']
                    for name,value in d['actual_update_increments'].items():stats[method]['actual_updates'][name]+=value
                elapsed=(time.perf_counter()-tick)*1000;timings.append(elapsed);worker_times.append(result['inference_ms']);rotor_times.append(rotor_result['request_wall_ms']);n+=1;detected+=bool(position.get('detected'));valid+=original_history['valid']
                row=dict(t=t,event_count=len(e),filtered_event_count=len(clean),geometry20=geometry,body_observation=body,observation=obs,sensing=sensed,slot_audit=rotor_result['record']['slot_audit'],
                    body_boxes=result['boxes'],frame_sha256=frame_sha,frontends=forecasts,lifecycle=diagnostics,processing_ms=elapsed,gpu_inference_ms=result['inference_ms'],rotor_request_ms=rotor_result['request_wall_ms'])
                stream.write(json_text(row)+'\n')
                if video is not None:
                    display=frame.copy()
                    if position.get('detected'):b=np.rint(position['bbox']).astype(int);cv2.rectangle(display,tuple(b[:2]),tuple(b[2:]),(30,170,255),2)
                    for r in sensed['rotors']:
                        if r.get('observed_now'):cv2.ellipse(display,((r['cx'],r['cy']),(max(r['major'],1.),max(r['minor'],1.)),r['angle']),(180,90,210),1)
                    q=forecasts['delayed_feedback']
                    if q['valid']:
                        for model,color in [('trusted_acc',(60,190,70)),('adaptive_axis_freq',(190,80,210))]:
                            pts=[q['coefficients'][model]['position']]+[q['predictions'][model][str(horizon)] for horizon in [.1,.2,.4]]
                            for a,b in zip(pts,pts[1:]):cv2.line(display,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),color,2)
                    cv2.putText(display,f'{t:.2f}s | green=history purple=axis+frequency | active={q["feature_diagnostic"]["active"]} | updates={stats["delayed_feedback"]["actual_updates"]["adaptive_axis_freq"]}',(12,h-20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1);video.write(display)
                if time.perf_counter()-last_print>15:print(json_text(dict(t=t,frames=n,detected=detected,valid=valid,physics=stats['delayed_feedback'])),flush=True);last_print=time.perf_counter()
        summary=dict(frames=n,published=detected,valid_forecasts=valid,methods=stats,processing_median_ms=float(np.median(timings)),processing_p95_ms=float(np.percentile(timings,95)),fraction_over20ms=float(np.mean(np.asarray(timings)>20)),
            gpu_inference_median_ms=float(np.median(worker_times)),rotor_request_median_ms=float(np.median(rotor_times)),source_hashes_unchanged=hashes==fingerprints(),labels_read=False,history_all_six_exact=True)
    finally:
        if video is not None:video.release()
        stop(gpu);stop(rotor)
        frame=None
        if memory is not None:memory.close();memory.unlink()
        for log in logs:log.close()
    summary['wall_including_cold_decode_video_s']=time.perf_counter()-began
    (output/'completion.json').write_text(json_text(summary),encoding='utf-8');print(json_text(summary),flush=True);return summary
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.);p.add_argument('--precision',type=int,choices=[16,32],default=32);p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true');a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.precision,a.preview,a.verify_frames)
