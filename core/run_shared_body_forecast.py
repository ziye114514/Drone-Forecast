"""RAW-only CUDA graph/shared-memory path; unchanged42 geometry/history arithmetic."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,types,traceback
from multiprocessing import shared_memory
ROOT=Path(__file__).resolve().parents[1]

def build(precision,body_only,verify_frames):
    path=ROOT/'core/run_body_forecast.py';source=path.read_text(encoding='utf-8');owned_memory=[];owned_workers=[]
    changes=[
      ("str(ROOT/'research/stage42/worker42.py')","str(ROOT/'research/stage47/worker47.py'),'--shared-memory',shared.name,'--precision',str(shared_precision)"),
      ("front=Observer(w,h,'native20')","shared=allocate_shared(720*1280*3);shared_frame=np.ndarray((720,1280,3),np.uint8,buffer=shared.buf);front=Observer(w,h,'native20')"),
      ("geometry=front.step(e,t)['observation'] if t-begin>=.1-1e-8", "geometry=front.step(e,t)['observation'] if geometry_enabled and t-begin>=.1-1e-8"),
      ("frame=np.empty((h,w,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(clean,frame);ok,png=cv2.imencode('.png',frame);assert ok", "frame=shared_frame;BaseFrameGenerationAlgorithm.generate_frame(clean,frame);frame_sha=hashlib.sha256(frame.tobytes()).hexdigest() if verify_frame_pixels else None"),
      ("dict(id=n,png=base64.b64encode(png.tobytes()).decode('ascii'))", "dict(id=n,verify_frame=verify_frame_pixels)"),
      ("result=json.loads(line);assert result['id']==n", "result=json.loads(line);assert result['id']==n;assert not verify_frame_pixels or result['frame_sha256']==frame_sha"),
      ("for method,o in observations.items():", "for method,o in [(m,observations[m]) for m in METHODS]:"),
      ("for method,color in [('geometry',(220,160,70)),(primary,(30,170,255))]:\n                        o=", "for method,color in [('geometry',(220,160,70)),(primary,(30,170,255))]:\n                        if method not in forecasts:continue\n                        o="),
      ("gpu_inference_ms=result['inference_ms'],processing_ms=elapsed", "gpu_inference_ms=result['inference_ms'],processing_ms=elapsed,frame_sha256=frame_sha"),
      ("step_us=20000,primary=primary,source_hashes=sources", "step_us=20000,primary=primary,precision=shared_precision,geometry_enabled=geometry_enabled,primary_only=single_body_branch,verify_frame_pixels=verify_frame_pixels,source_hashes=sources"),
      ("Per-frame local PNG pipe, independent eval mode batch1.","Owned shared-memory native BGR image, synchronous current id/ack, independent CUDA graph batch1. No PNG transport."),
      ("scope='Current visible whole-body box + original rotor geometry; predictions use six causal history models only.","scope='Current visible whole-body box + optional original rotor geometry; predictions use six causal history models only.")]
    for old,new in changes:assert source.count(old)==1,old;source=source.replace(old,new)
    namespace={'__name__':'shared_body42','__file__':str(path),'geometry_enabled':not body_only,'shared_precision':precision,'verify_frame_pixels':verify_frames,'single_body_branch':body_only}
    def allocate(size):
        memory=shared_memory.SharedMemory(create=True,size=size);owned_memory.append(memory);return memory
    namespace['allocate_shared']=allocate;exec(compile(source,str(path),'exec'),namespace)
    original_fingerprints=namespace['fingerprints']
    def fingerprints():
        result=original_fingerprints()
        for p in [Path(__file__),ROOT/'research/stage47/worker47.py',ROOT/'research/stage45/graph45.py',ROOT/'research/stage44/common44.py',ROOT/'research/stage41/common41.py',ROOT/'research/stage40/common40.py',ROOT/'research/stage37/common37.py']:
            result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace['fingerprints']=fingerprints
    if body_only:namespace['METHODS']=['body_only']
    def spawn(*args,**kwargs):
        process=subprocess.Popen(*args,**kwargs);owned_workers.append(process);return process
    namespace['subprocess']=types.SimpleNamespace(Popen=spawn,CREATE_NO_WINDOW=getattr(subprocess,'CREATE_NO_WINDOW',0),TimeoutExpired=subprocess.TimeoutExpired)
    return namespace,owned_memory,owned_workers

def run(raw,output,start=0.,duration=0.,precision=32,body_only=False,verify_frames=False,preview=False):
    assert precision in [16,32];pipeline,memory,workers=build(precision,body_only,verify_frames)
    try:return pipeline['run'](raw,output,start,duration,primary='body_only' if body_only else 'body_first',preview=preview)
    except BaseException as error:
        traceback.clear_frames(error.__traceback__);raise
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                try:worker.wait(timeout=5)
                except subprocess.TimeoutExpired:worker.kill();worker.wait(timeout=5)
        pipeline.clear()
        for item in memory:
            item.close();item.unlink()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.);p.add_argument('--precision',type=int,choices=[16,32],default=32);p.add_argument('--body-only',action='store_true');p.add_argument('--verify-frames',action='store_true');p.add_argument('--preview',action='store_true');a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.precision,a.body_only,a.verify_frames,a.preview)
