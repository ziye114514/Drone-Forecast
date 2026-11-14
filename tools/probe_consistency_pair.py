"""Two trained detectors at the accumulated run's exact native-frame cutoffs."""
from pathlib import Path
from multiprocessing import shared_memory
import hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_decode import native_batches
from forecast_runtime import resolve
from run_live_forecast import receive
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
OUT=WORK/'output/flip_consistency_20261026/native_probe';OUT.mkdir(exist_ok=False)
(OUT/'source_probe.py').write_bytes(Path(__file__).read_bytes())
exports={n:json.loads((OUT.parent/n/'export.json').read_text('utf-8')) for n in ['control','consistency']}
runtime=resolve(detector_weights=exports['control']['model'],detector_sha256=exports['control']['sha256'])
memory=shared_memory.SharedMemory(create=True,size=720*1280*3);image=np.ndarray((720,1280,3),np.uint8,buffer=memory.buf)
log=(OUT/'gpu_worker.log').open('x',encoding='utf-8');gpu=None;counter=0;summary={}

def body(boxes,t):
    if not boxes:return dict(t=t,detected=False,center=None)
    selected=max(boxes,key=lambda r:r['confidence']);b=np.asarray(selected['bbox'])
    return dict(t=t,detected=True,bbox=b.tolist(),confidence=selected['confidence'],center=((b[:2]+b[2:])/2).tolist())

try:
    gpu=subprocess.Popen([runtime['gpu_python'],str(ROOT/'core/live_worker.py'),'--shared-memory',memory.name,
        '--weights',runtime['detector_weights'],'--expected-sha256',runtime['detector_sha256'],'--cache-dir',runtime['cache_dir'],
        '--center-ensemble-weights',exports['consistency']['model'],'--center-ensemble-sha256',exports['consistency']['sha256']],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert receive(gpu,log)['ready'];image.fill(127)
    gpu.stdin.write(json.dumps(dict(id=counter,warmup=True))+'\n');gpu.stdin.flush();assert receive(gpu,log)['id']==counter;counter+=1
    cases=dict(side=WORK/'output/single_model_optimized_RAW_20261026/side',
        train2=WORK/'output/single_model_optimized_RAW_20261026/train2',test3=WORK/'output/heldout_motion_RAW_20261026')
    for case,run in cases.items():
        protocol=json.loads((run/'protocol.json').read_text('utf-8'))
        saved={round(r['t'],6):r for r in map(json.loads,(run/'tracking.jsonl').read_text('utf-8').splitlines())}
        noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer()
        begin=max(0.,protocol['start']-.1);rows=[];native_count=native_events=0
        snapshots=set(sorted(saved)[k] for k in [0,len(saved)//2,len(saved)-1])
        for t,e in native_batches(protocol['raw'],begin,protocol['duration']+protocol['start']-begin,build_index=False):
            native_count+=1;native_events+=len(e);noise.process_events(e,buffer);r=saved.get(round(t,6))
            if r is None:continue
            BaseFrameGenerationAlgorithm.generate_frame(buffer.numpy(),image);sha=hashlib.sha256(image.tobytes()).hexdigest()
            assert sha==r['detector_frame_sha256']
            gpu.stdin.write(json.dumps(dict(id=counter,t=t,verify_frame=True))+'\n');gpu.stdin.flush()
            d=receive(gpu,log);assert d['id']==counter and d['frame_sha256']==sha;counter+=1
            rows.append(dict(t=t,frame_sha256=sha,control=body(d['boxes'],t),consistency=body(d['ensemble_boxes'],t)))
            if round(t,6) in snapshots:
                import cv2
                cv2.imwrite(str(OUT/f'{case}_{t:.6f}.png'),image)
        assert len(rows)==len(saved)
        (OUT/(case+'.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
        summary[case]=dict(matched_current_frames=len(rows),native_slices=native_count,native_events=native_events,
            detected={n:sum(row[n]['detected'] for row in rows) for n in exports})
        print(json.dumps({case:summary[case]}),flush=True)
    report=dict(results=summary,weights={n:e['sha256'] for n,e in exports.items()},all_current_frame_hashes_exact=True,
        labels_read=False,scope='Offline matched-frame pair; two models for measurement only. Deployment uses a single exported model.')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
finally:
    if gpu is not None and gpu.poll() is None:
        gpu.stdin.close()
        try:gpu.wait(timeout=20)
        except subprocess.TimeoutExpired:gpu.terminate();gpu.wait(timeout=5)
    image=None;memory.close();memory.unlink();log.close()
