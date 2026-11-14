"""Same native20ms images, current ROI center versus current full-frame center."""
from pathlib import Path
from multiprocessing import shared_memory
from collections import Counter
import hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path[:0]=[str(ROOT/'core'),str(ROOT)]
from forecast_runtime import resolve
from live_decode import native_batches
from run_live_forecast import receive
from live_roi_refine import DESIGN
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
OUT=WORK/'output/roi_refine_20261026';OUT.mkdir(exist_ok=False);(OUT/'source_snapshot').mkdir()
for name in ['live_roi_refine.py','live_worker.py','tools/probe_roi_refine.py']:
    dest=OUT/'source_snapshot'/name;dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes((ROOT/name).read_bytes())
(OUT/'design.json').write_text(json.dumps(dict(**DESIGN,
    gate='Both same-valid-origin .4s mean errors decrease; .1s neither worsens more than1px. Center replacement only; original bbox/eligibility inputs retained.'),indent=2),encoding='utf-8')
runtime=resolve();memory=shared_memory.SharedMemory(create=True,size=720*1280*3);image=np.ndarray((720,1280,3),np.uint8,buffer=memory.buf)
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
            BaseFrameGenerationAlgorithm.generate_frame(buffer.numpy(),image)
            sha=hashlib.sha256(image.tobytes()).hexdigest();assert sha==r['detector_frame_sha256']
            gpu.stdin.write(json.dumps(dict(id=counter,t=t,verify_frame=True,refine_roi=True))+'\n');gpu.stdin.flush()
            d=receive(gpu,log);assert d['id']==counter and d['frame_sha256']==sha;counter+=1
            assert d['boxes']==r['detector_boxes'],('original_detection_changed',case,t)
            rows.append(dict(t=t,original_body=r['body'],refined_body=d['refined_body'],refinement=d['refinement'],refinement_ms=d['refinement_ms']))
        report=dict(native_slices=total,native_events=events,matched_current_ticks=len(rows),
            refined_centers=sum(r['refined_body'] is not None for r in rows),
            reasons=dict(Counter(r['refinement']['reason'] for r in rows)),
            refinement_median_ms=float(np.median([r['refinement_ms'] for r in rows])),
            all_original_frame_hashes_and_NN_boxes_exact=True,rows=rows)
        (OUT/(case+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
        results[case]={k:v for k,v in report.items() if k!='rows'};print(json.dumps({case:results[case]}),flush=True)
    (OUT/'summary.json').write_text(json.dumps(dict(results=results,scope='Native-event observer with frozen timestamps; not paced end-to-end deployment.',
        labels_read_during_probe=False,model_unchanged=True,default_promoted=False),indent=2),encoding='utf-8')
finally:
    if gpu is not None and gpu.poll() is None:
        gpu.stdin.close()
        try:gpu.wait(timeout=20)
        except subprocess.TimeoutExpired:gpu.terminate();gpu.wait(timeout=5)
    image=None;memory.close();memory.unlink();log.close()
