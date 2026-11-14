"""Lean RAW-only optimized135 motion prediction.

One RAW reader, one shared event-count pair, unchanged local CUDA detector.
Two guarded geometry models and only optimized/matched-body/CV forecasts;
no old72-way research controls or physical-force branches in this entry.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
from multiprocessing import shared_memory
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage138'),str(ROOT/'research/stage137'),str(ROOT/'research/stage136'),str(ROOT/'research/stage135'),str(ROOT/'research/stage129'),str(ROOT/'research/stage124')]
from predictor135 import Predictor,forecast
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
    paths=[Path(__file__).resolve(),ROOT/'research/stage138/objective138.py',ROOT/'research/stage138/tiles138.py',ROOT/'research/stage137/fast137.py',ROOT/'research/stage136/fast_geometry136.py',ROOT/'research/stage135/predictor135.py',ROOT/'research/stage135/publication135.py',ROOT/'research/stage132/fast132.py',ROOT/'research/stage130/motion130.py',ROOT/'research/stage23/robust23.py',ROOT/'research/stage129/boundary129.py',ROOT/'research/stage124/occupancy124.py',ROOT/'research/stage47/worker47.py']
    paths += [p for d in [8,9,55,58,61,104,107,110,113,117,119,122] for p in (ROOT/f'research/stage{d}').glob('*.py')]
    paths += list((ROOT/'src/event_drone_forecast').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve();mode=args[1] if len(args)>1 else None;write=isinstance(mode,str) and any(v in mode for v in 'wax+')
        if not write and p.is_relative_to(ROOT.parent/'output') and p.suffix.lower() in ['.json','.jsonl','.png','.mp4','.npz']:
            raise RuntimeError('No output cache or truth input in optimized RAW entry: '+str(p))
def receive(p):
    line=p.stdout.readline()
    if not line:raise RuntimeError('GPU worker stopped; inspect gpu_worker.log')
    return json.loads(line)
def run(raw,output,start,duration,preview,verify_frames,geometry_backend='shared'):
    raw=Path(raw).resolve();out=Path(output).resolve()
    for p in [Path('D:/yolov10-main'),Path('E:/dataset/software/yolov10-main')]:
        if raw.is_relative_to(p) or out.is_relative_to(p):raise ValueError('Protected project')
    if out.exists():raise FileExistsError(out)
    if start<0 or duration<=0 or round(start*1e6)%20000 or round(duration*1e6)%20000:raise ValueError('Positive duration/start must align to20ms')
    if not raw.is_file() or raw.suffix.lower()!='.raw':raise ValueError('Expected RAW')
    out.mkdir(parents=True,exist_ok=False);sys.addaudithook(guard);hashes=fingerprints();began=time.perf_counter();reader=open_raw(raw);h,w=reader.get_size();assert (h,w)==(720,1280)
    if geometry_backend=='shared':install_shared_objective();fronts=tile_fronts(w,h)
    else:fronts=fast_geometry_setup(w,h) if geometry_backend=='fast' else dict(count=MergeRecovery(w,h,'merge_safe'),occupancy=SupportFrontend(w,h,'occupancy_roi'))
    models={name:PartialModel('affine3') for name in fronts};predictor=Predictor()
    noise=ActivityNoiseFilterAlgorithm(w,h,10000);buffer=noise.get_empty_output_buffer();memory=None;image=None;gpu=None;video=None;log=None;n=0;publisher=Publication();cost=[];geometry_cost=[];nn_cost=[];render_cost=[];frontend_cost=[];model_cost=[]
    try:
        memory=shared_memory.SharedMemory(create=True,size=h*w*3);image=np.ndarray((h,w,3),np.uint8,buffer=memory.buf);log=(out/'gpu_worker.log').open('x',encoding='utf-8')
        infer=Path('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe');assert infer.is_file()
        gpu=subprocess.Popen([str(infer),str(ROOT/'research/stage47/worker47.py'),'--shared-memory',memory.name,'--precision','32'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        ready=receive(gpu);assert ready['ready']
        (out/'protocol.json').write_text(js(dict(stage=138,predictor_stage=135,geometry_backend=geometry_backend,counts_overlap_GPU=geometry_backend=='shared',entry='run_optimized_forecast.py',raw=str(raw),start=start,duration=duration,raw_size=raw.stat().st_size,raw_mtime_ns=raw.stat().st_mtime_ns,
            input='RAW and fixed pre-existing detector weights only; no old output/cache/label input.',sources=hashes,gpu_worker=ready,geometry='Current20ms native normalized counts; count and occupancy share one prepared ON/OFF pair; existing FOV guard and independent affine3 models.',
            forecasts=['optimized_geometry','matched_body','body_cv'],research_controls_removed=True,default_promoted=False)),'utf-8')
        if preview:video=cv2.VideoWriter(str(out/'predictions.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50,(w+400,h));assert video.isOpened()
        begin=max(0.,start-.1)
        with (out/'predictions.jsonl').open('x',encoding='utf-8') as stream:
            for t,e in raw_chunks(reader,begin,duration+start-begin,20000):
                tick=time.perf_counter();noise.process_events(e,buffer);clean=buffer.numpy().copy()
                geometry_ready=t-begin>=.1-1e-8
                if t<=start+1e-8:
                    if geometry_ready:
                        prepared=counts(e,w,h);prepared['on']/=2;prepared['off']/=2
                        fronts['count'].detect_prepared(e,t,prepared);fronts['occupancy'].set_hint(e,None,t);fronts['occupancy'].detect_prepared(e,t,prepared)
                    continue
                if geometry_backend!='shared':prepared=counts(e,w,h);prepared['on']/=2;prepared['off']/=2
                BaseFrameGenerationAlgorithm.generate_frame(clean,image);digest=hashlib.sha256(image.tobytes()).hexdigest() if verify_frames else None
                gpu.stdin.write(json.dumps(dict(id=n,verify_frame=verify_frames))+'\n');gpu.stdin.flush()
                if geometry_backend=='shared':prepared=counts(e,w,h);prepared['on']/=2;prepared['off']/=2
                nn=receive(gpu);assert nn['id']==n and (not verify_frames or nn['frame_sha256']==digest)
                if nn['boxes']:
                    b=max(nn['boxes'],key=lambda b:b['confidence']);box=b['bbox'];body=dict(t=t,detected=True,center=[(box[0]+box[2])/2,(box[1]+box[3])/2],bbox=box,rotors=[],confidence=b['confidence'],source='body')
                else:body=dict(t=t,detected=False,center=None,rotors=[],source='body')
                gtick=time.perf_counter();fronts['occupancy'].set_hint(e,body,t);observations={};current={};boundary={};diagnostics={};fcost=0.;mcost=0.
                for name,front in fronts.items():
                    ftick=time.perf_counter();obs=front.detect_prepared(e,t,prepared)[0] if geometry_ready else dict(t=t,detected=False,rotors=[]);fcost+=(time.perf_counter()-ftick)*1000
                    mtick=time.perf_counter();rotors,audit=bounded(obs.get('rotors',[]),w,h);baseline=baseline_for(models[name],t,rotors,body);_,d=models[name].step(t,body,rotors,dict(valid=False),baseline);mcost+=(time.perf_counter()-mtick)*1000
                    observations[name]=obs;boundary[name]=audit;current[name]=rotors;diagnostics[name]=d
                gcost=(time.perf_counter()-gtick)*1000;ptick=time.perf_counter();result=predictor.step(t,body,diagnostics);pcost=(time.perf_counter()-ptick)*1000
                valid=publisher.step(t,body['detected'])
                coefficients=dict(result['coefficients']);cv=coefficients['matched_body']
                if cv is not None:cv=dict(cv,acceleration=[0.,0.])
                coefficients['body_cv']=cv;predictions={name:{str(dt):forecast(c,dt) for dt in [.1,.2,.4]} for name,c in coefficients.items() if c is not None}
                elapsed=(time.perf_counter()-tick)*1000;cost.append(elapsed);geometry_cost.append(gcost);nn_cost.append(nn['inference_ms']);frontend_cost.append(fcost);model_cost.append(mcost);render_tick=time.perf_counter()
                record=dict(t=t,body=body,models=diagnostics,geometry=observations,boundary=boundary,tiling={name:getattr(front,'tile_diagnostic',None) for name,front in fronts.items()},predictor=result,coefficients=coefficients,predictions=predictions,valid=valid,
                    event_count=len(e),max_event_t=int(e['t'].max()) if len(e) else None,frame_sha256=digest,processing_ms=elapsed,geometry_ms=gcost,frontend_ms=fcost,model_ms=mcost,predictor_ms=pcost,NN_ms=nn['inference_ms'])
                stream.write(js(record)+'\n')
                if video is not None:
                    im=np.full((h,w+400,3),248,np.uint8);im[:,:w]=image
                    for name,col in [('body_cv',(130,130,130)),('optimized_geometry',(40,170,20))]:
                        if valid and name in predictions:
                            points=np.rint([coefficients[name]['position']]+[predictions[name][str(dt)] for dt in [.1,.2,.4]]).astype(np.int32);cv2.polylines(im,[points],False,col,2);cv2.circle(im,tuple(points[-1]),4,col,2)
                    lines=[f'RAW t={t:.2f}s','Green: optimized0.4s forecast','Gray: body CV0.4s','Relative rotor motion -> velocity','Robust acceleration -> rollout',f'Structure: {result["structure_supported"]}',f'Geometric motion: {result["current_geometry_velocity_source"]}',f'Process: {elapsed:.1f}ms','50fps playback, not realtime']
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
        completion=dict(frames=n,raw_only=True,labels_read=False,output_caches_read=False,sources_unchanged=hashes==fingerprints(),processing_median_ms=float(np.median(cost)),geometry_median_ms=float(np.median(geometry_cost)),gpu_median_ms=float(np.median(nn_cost)),serialization_preview_median_ms=float(np.median(render_cost)),wall_including_cold_start_s=time.perf_counter()-began,
            only_three_forecasts=True,old_research_controls_not_executed=True,geometry_backend=geometry_backend,frontend_median_ms=float(np.median(frontend_cost)),model_median_ms=float(np.median(model_cost)))
        (out/'completion.json').write_text(js(completion),'utf-8');print(js(completion),flush=True)
    finally:
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
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=2.);p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true');p.add_argument('--geometry-backend',choices=['shared','fast','legacy'],default='shared');a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames,a.geometry_backend)
