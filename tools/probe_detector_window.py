"""Fixed20/60ms detector input probe; one model, same native cutoffs."""
from pathlib import Path
from collections import Counter,deque
from multiprocessing import shared_memory
import hashlib,json,subprocess,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path[:0]=[str(ROOT/'core'),str(ROOT)]
from live_decode import native_batches
from live_detector_window import DetectorWindow
from live_stream import RateGate
from forecast_runtime import resolve
from run_live_forecast import receive
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
OUT=WORK/'output/detector_window_v2_20261026';OUT.mkdir(exist_ok=False)
(OUT/'source_snapshot').mkdir()
for name in ['live_detector_window.py','live_worker.py','tools/probe_detector_window.py']:
    dest=OUT/'source_snapshot'/name;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes((ROOT/name).read_bytes())
runtime=resolve();memory=shared_memory.SharedMemory(create=True,size=720*1280*3)
image=np.ndarray((720,1280,3),np.uint8,buffer=memory.buf)
log=(OUT/'gpu_worker.log').open('x',encoding='utf-8');gpu=None;counter=0
design=dict(windows_ms=[20,60],selection='Preserve20ms detections; query60ms only if20ms has none.',
    gate='More present detections on train2; no new empty-scene detections. Inspect center error before integrating.',
    parameter_sweep=False,grayscale_reconstruction=False,past_native_events_only=True,labels_fed=False)
(OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
results={}
try:
    gpu=subprocess.Popen([runtime['gpu_python'],str(ROOT/'core/live_worker.py'),'--shared-memory',memory.name,
        '--weights',runtime['detector_weights'],'--expected-sha256',runtime['detector_sha256'],
        '--cache-dir',runtime['cache_dir']],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,
        encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert receive(gpu,log)['ready']
    image.fill(127);gpu.stdin.write(json.dumps(dict(id=counter,warmup=True))+'\n');gpu.stdin.flush()
    assert receive(gpu,log)['id']==counter;counter+=1
    for case,start,duration,raw in [('train2',40.,4.,'E:/dataset/train/2/aligned/event.raw'),
          ('side',110.,1.2,'E:/dataset/train/6/media/event.raw'),('negative',0.,1.,'E:/dataset/train/5/media/event.raw')]:
        noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer()
        windows={ms:DetectorWindow(1280,720,ms) for ms in [20,60]};gate=RateGate(start,25.)
        rows=[];n_slices=n_events=0;costs=[];new=0;past=deque(maxlen=3);boundary_differences=0
        for t,e in native_batches(raw,max(0.,start-.1),duration+min(.1,start),build_index=False):
            n_slices+=1;n_events+=len(e);noise.process_events(e,buffer);clean=buffer.numpy();past.append(clean.copy())
            for engine in windows.values():engine.feed(clean,t)
            if t<=start+1e-8 or not gate.take(t):continue
            row=dict(t=t,windows={});frames={}
            for ms,engine in windows.items():
                stage=time.perf_counter();frames[ms]=engine.generate(t).copy();costs.append((time.perf_counter()-stage)*1000)
                # Installed SDK uses an exclusive lower boundary. Explicitly
                # check that convention, then preserve the original20ms image.
                source=clean if ms==20 else np.concatenate(list(past))
                inside=source[source['t']>round(t*1e6)-ms*1000]
                BaseFrameGenerationAlgorithm.generate_frame(inside,image)
                assert np.array_equal(image,frames[ms]),('SDK_window_mismatch',case,t,ms)
                if ms==20:
                    BaseFrameGenerationAlgorithm.generate_frame(clean,image)
                    boundary_differences+=int(np.sum(np.any(image!=frames[20],axis=2)))
                    frames[20]=image.copy()
                np.copyto(image,frames[ms]);digest=hashlib.sha256(image.tobytes()).hexdigest()
                gpu.stdin.write(json.dumps(dict(id=counter,t=t,verify_frame=True))+'\n');gpu.stdin.flush()
                d=receive(gpu,log);assert d['id']==counter and d['frame_sha256']==digest;counter+=1
                row['windows'][str(ms)]=dict(boxes=d['boxes'],sha256=digest,inference_ms=d['inference_ms'])
            fallback=not row['windows']['20']['boxes'] and bool(row['windows']['60']['boxes'])
            row['fallback_added']=fallback;new+=fallback
            if fallback and new<=3:
                display=np.hstack([frames[20],frames[60]])
                cv2.imwrite(str(OUT/f'{case}_fallback_{new}.png'),display)
            rows.append(row)
        result=dict(native_slices=n_slices,native_events=n_events,ticks=len(rows),
            detected20=sum(bool(r['windows']['20']['boxes']) for r in rows),
            detected60=sum(bool(r['windows']['60']['boxes']) for r in rows),
            fallback_added=new,all_native20_frames_exact=True,SDK_window_boundary='exclusive_lower',
            SDK20_boundary_pixel_differences_not_used=boundary_differences,
            frame_generation_median_ms=float(np.median(costs)),rows=rows)
        (OUT/(case+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        results[case]={k:v for k,v in result.items() if k!='rows'}
        print(json.dumps(dict(case=case,**results[case])),flush=True)
    (OUT/'summary.json').write_text(json.dumps(dict(design=design,results=results,
        model_sha256=runtime['detector_sha256'],scope='Offline same-cutoff input probe; not realtime or forecast improvement',
        default_promoted=False),indent=2),encoding='utf-8')
finally:
    if gpu is not None:
        if gpu.poll() is None:
            gpu.stdin.close()
            try:gpu.wait(timeout=20)
            except subprocess.TimeoutExpired:gpu.terminate();gpu.wait(timeout=5)
    image=None;memory.close();memory.unlink();log.close()
