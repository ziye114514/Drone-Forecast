"""Same native images and CUDA worker: prepare support before/after receive."""
from pathlib import Path
from multiprocessing import shared_memory
import hashlib,json,subprocess,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from forecast_runtime import resolve
from live_local_flow import frame_support
from run_live_forecast import receive
from metavision_sdk_core import BaseFrameGenerationAlgorithm
OUT=WORK/'output/support_overlap_20261026';OUT.mkdir(exist_ok=False)
images=sorted((WORK/'output/flip_consistency_20261026/native_probe').glob('*.png'));assert len(images)==9
manifest=json.loads((ROOT/'configs/live_tracks.json').read_text('utf-8'));recipe=manifest['tracks']['accumulated_v5_fast_ci']
runtime=resolve(detector_weights=str(ROOT/recipe['model']),detector_sha256=recipe['sha256'])
design=dict(mechanism='Generate the same current support after GPU request flush and before receive, instead of after receive.',
    native_images=[str(p) for p in images],paired_repetitions=3,order='Alternate by image index and repetition',
    no_RAW_decode=True,no_labels=True,model_sha256=runtime['detector_sha256'],whole_pipeline_speed_not_claimed=True)
(OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
memory=shared_memory.SharedMemory(create=True,size=720*1280*3);image=np.ndarray((720,1280,3),np.uint8,buffer=memory.buf)
log=(OUT/'gpu_worker.log').open('x',encoding='utf-8');gpu=None;counter=0;rows=[]
try:
    gpu=subprocess.Popen([runtime['gpu_python'],str(ROOT/'core/live_worker.py'),'--shared-memory',memory.name,
        '--weights',runtime['detector_weights'],'--expected-sha256',runtime['detector_sha256'],'--cache-dir',runtime['cache_dir']],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert receive(gpu,log)['ready'];image.fill(127)
    gpu.stdin.write(json.dumps(dict(id=counter,warmup=True))+'\n');gpu.stdin.flush();assert receive(gpu,log)['id']==counter;counter+=1
    for repetition in range(3):
        for index,path in enumerate(images):
            np.copyto(image,cv2.imread(str(path)));sha=hashlib.sha256(image.tobytes()).hexdigest();baseline=None
            order=['serial','gpu-overlap'] if (repetition+index)%2==0 else ['gpu-overlap','serial']
            for mode in order:
                start=time.perf_counter();gpu.stdin.write(json.dumps(dict(id=counter,verify_frame=True))+'\n');gpu.stdin.flush()
                support=None
                if mode=='gpu-overlap':support=frame_support(image,BaseFrameGenerationAlgorithm.bg_color_default()[0])
                nn=receive(gpu,log);assert nn['id']==counter and nn['frame_sha256']==sha;counter+=1
                if support is None:support=frame_support(image,BaseFrameGenerationAlgorithm.bg_color_default()[0])
                elapsed=(time.perf_counter()-start)*1000
                assert hashlib.sha256(image.tobytes()).hexdigest()==sha
                matched=dict(boxes=nn['boxes'],support_sha256=hashlib.sha256(support.tobytes()).hexdigest())
                if baseline is None:baseline=matched
                else:assert baseline==matched
                rows.append(dict(image=path.name,repetition=repetition,mode=mode,stage_ms=elapsed,
                    NN_ms=nn['inference_ms'],frame_sha256=sha,support_sha256=matched['support_sha256']))
    summary={mode:dict(n=len(v),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)))
        for mode in ['serial','gpu-overlap'] for v in [[r['stage_ms'] for r in rows if r['mode']==mode]]}
    report=dict(summary=summary,all_NN_boxes_and_support_hashes_exact=True,read_only_current_image=True,
        model_sha256=runtime['detector_sha256'],RAW_or_labels_read=False,scope='Matched-image GPU request plus support preparation only; not end-to-end RAW latency.',rows=rows)
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
finally:
    if gpu is not None and gpu.poll() is None:
        gpu.stdin.close()
        try:gpu.wait(timeout=20)
        except subprocess.TimeoutExpired:gpu.terminate();gpu.wait(timeout=5)
    image=None;memory.close();memory.unlink();log.close()
