"""Fresh native input with selected adapted model; no labels used for inference."""
from pathlib import Path
from multiprocessing import shared_memory
import argparse,hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT)]
from forecast_runtime import resolve
from live_decode import native_batches
from run_live_forecast import receive
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
parser=argparse.ArgumentParser();parser.add_argument('--model-dir',type=Path);parser.add_argument('--no-roi',action='store_true');args=parser.parse_args()
PARENT=args.model_dir.resolve() if args.model_dir else WORK/'output/roi_augmented_detector_20261026'
export=json.loads((PARENT/'export.json').read_text('utf-8'))
OUT=PARENT/'forecast_probe';OUT.mkdir(exist_ok=False)
runtime=resolve(detector_weights=export['model'],detector_sha256=export['sha256'])
(OUT/'plan.json').write_text(json.dumps(dict(
    candidate=export.get('kind','Single validation-selected fine-tune')+'; full image and current detected-ROI center variants.',
    gate='Both .4s means improve on all original scoreable valid publication targets; neither .1s worsens more than1px. No parameter tuning.',
    refiner_selection='Unchanged fixed current IoU/center/confidence rule',
    latency='Frozen original clocks ignore additional inference cost; positive results require paced RAW test.',
    labels_input=False,weights_sha256=export['sha256']),indent=2),encoding='utf-8')
memory=shared_memory.SharedMemory(create=True,size=720*1280*3);image=np.ndarray((720,1280,3),np.uint8,buffer=memory.buf)
log=(OUT/'gpu_worker.log').open('x',encoding='utf-8');gpu=None;counter=0;results={}
try:
    gpu=subprocess.Popen([runtime['gpu_python'],str(ROOT/'core/live_worker.py'),'--shared-memory',memory.name,
        '--weights',runtime['detector_weights'],'--expected-sha256',runtime['detector_sha256'],'--cache-dir',runtime['cache_dir']],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert receive(gpu,log)['ready'];image.fill(127)
    gpu.stdin.write(json.dumps(dict(id=counter,warmup=True))+'\n');gpu.stdin.flush();assert receive(gpu,log)['id']==counter;counter+=1
    for case,start,raw in [('side_v2',110.,'E:/dataset/train/6/media/event.raw'),('train2',40.,'E:/dataset/train/2/aligned/event.raw')]:
        old=WORK/'output/live_scale_rollout_RAW_20261026'/case/'tracking.jsonl'
        saved={round(r['t'],6):r for r in [json.loads(s) for s in old.read_text('utf-8').splitlines()]}
        noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer();rows=[];total=0;events=0
        for t,e in native_batches(raw,start-.1,4.1,build_index=False):
            total+=1;events+=len(e);noise.process_events(e,buffer);r=saved.get(round(t,6))
            if r is None:continue
            BaseFrameGenerationAlgorithm.generate_frame(buffer.numpy(),image);sha=hashlib.sha256(image.tobytes()).hexdigest()
            assert sha==r['detector_frame_sha256']
            gpu.stdin.write(json.dumps(dict(id=counter,t=t,verify_frame=True,refine_roi=not args.no_roi))+'\n');gpu.stdin.flush()
            d=receive(gpu,log);assert d['id']==counter and d['frame_sha256']==sha;counter+=1
            boxes=d['boxes']
            if boxes:
                best=max(boxes,key=lambda b:b['confidence']);b=np.asarray(best['bbox'])
                body=dict(t=t,detected=True,bbox=b.tolist(),center=((b[:2]+b[2:])/2).tolist(),confidence=best['confidence'])
            else:body=dict(t=t,detected=False,center=None)
            rows.append(dict(t=t,original_body=r['body'],adapted_body=body,refined_body=d.get('refined_body'),
                refinement=d.get('refinement'),refinement_ms=d.get('refinement_ms'),inference_ms=d['inference_ms'],frame_sha256=sha))
        report=dict(native_slices=total,native_events=events,matched_current_ticks=len(rows),ROI_requested=not args.no_roi,
            full_detections=sum(r['adapted_body']['detected'] for r in rows),
            refined_centers=sum(r['refined_body'] is not None for r in rows),
            original_detections=sum(r['original_body']['detected'] for r in rows),
            all_original_input_frames_exact=True,rows=rows)
        (OUT/(case+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        results[case]={k:v for k,v in report.items() if k!='rows'};print(json.dumps({case:results[case]}),flush=True)
    (OUT/'summary.json').write_text(json.dumps(dict(results=results,labels_read_during_inference=False,
        scope='Saved cutoff cohort, fresh native frames and new detector; no paced latency claim',default_promoted=False),indent=2),encoding='utf-8')
finally:
    if gpu is not None and gpu.poll() is None:
        gpu.stdin.close()
        try:gpu.wait(timeout=20)
        except subprocess.TimeoutExpired:gpu.terminate();gpu.wait(timeout=5)
    image=None;memory.close();memory.unlink();log.close()
