"""Offline detector at exactly a completed run's accepted cutoffs."""
from pathlib import Path
from multiprocessing import shared_memory
import argparse,hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_decode import native_batches
from forecast_runtime import resolve
from run_live_forecast import receive
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--model-dir',type=Path);parser.add_argument('--reference-name',default='original_reference')
parser.add_argument('--only-times',type=float,nargs='+');args=parser.parse_args()
if Path(args.reference_name).name!=args.reference_name:raise ValueError('Reference name must be one directory name')
RUN=args.run_dir.resolve();OUT=RUN/args.reference_name;OUT.mkdir(exist_ok=False)
(OUT/'source_probe.py').write_bytes(Path(__file__).read_bytes())
protocol=json.loads((RUN/'protocol.json').read_text('utf-8'))
saved={round(r['t'],6):r for r in [json.loads(l) for l in (RUN/'tracking.jsonl').read_text('utf-8').splitlines()]}
if args.only_times is not None:
    chosen={round(t,6) for t in args.only_times};assert chosen<=saved.keys()
    saved={t:r for t,r in saved.items() if t in chosen}
overrides={}
if args.model_dir is not None:
    export=json.loads((args.model_dir.resolve()/'export.json').read_text('utf-8'))
    overrides=dict(detector_weights=export['model'],detector_sha256=export['sha256'])
runtime=resolve(**overrides);memory=shared_memory.SharedMemory(create=True,size=720*1280*3);image=np.ndarray((720,1280,3),np.uint8,buffer=memory.buf)
log=(OUT/'gpu_worker.log').open('x',encoding='utf-8');gpu=None;counter=0;rows=[];native_count=native_events=0
try:
    gpu=subprocess.Popen([runtime['gpu_python'],str(ROOT/'core/live_worker.py'),'--shared-memory',memory.name,
        '--weights',runtime['detector_weights'],'--expected-sha256',runtime['detector_sha256'],'--cache-dir',runtime['cache_dir']],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert receive(gpu,log)['ready'];image.fill(127)
    gpu.stdin.write(json.dumps(dict(id=counter,warmup=True))+'\n');gpu.stdin.flush();assert receive(gpu,log)['id']==counter;counter+=1
    noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer()
    begin=max(0.,protocol['start']-.1)
    for t,e in native_batches(protocol['raw'],begin,protocol['duration']+protocol['start']-begin,build_index=False):
        native_count+=1;native_events+=len(e)
        noise.process_events(e,buffer);r=saved.get(round(t,6))
        if r is None:continue
        BaseFrameGenerationAlgorithm.generate_frame(buffer.numpy(),image);sha=hashlib.sha256(image.tobytes()).hexdigest()
        assert sha==r['detector_frame_sha256']
        gpu.stdin.write(json.dumps(dict(id=counter,t=t,verify_frame=True))+'\n');gpu.stdin.flush()
        d=receive(gpu,log);assert d['id']==counter and d['frame_sha256']==sha;counter+=1
        if d['boxes']:
            b=max(d['boxes'],key=lambda b:b['confidence']);box=np.asarray(b['bbox'])
            body=dict(t=t,detected=True,bbox=box.tolist(),confidence=b['confidence'],center=((box[:2]+box[2:])/2).tolist())
        else:body=dict(t=t,detected=False,center=None)
        rows.append(dict(t=t,body=body,frame_sha256=sha))
    assert len(rows)==len(saved)
    (OUT/'observations.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    report=dict(matched_current_frames=len(rows),all_native_input_hashes_exact=True,
        native_slices=native_count,native_events=native_events,
        original_weight_sha256=runtime['detector_sha256'] if args.model_dir is None else None,
        detector_weight_sha256=runtime['detector_sha256'],labels_read=False,
        scope='Offline detector-only reference; not independently timed production baseline.')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
finally:
    if gpu is not None and gpu.poll() is None:
        gpu.stdin.close()
        try:gpu.wait(timeout=20)
        except subprocess.TimeoutExpired:gpu.terminate();gpu.wait(timeout=5)
    image=None;memory.close();memory.unlink();log.close()
