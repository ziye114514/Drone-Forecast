"""Interleaved three-window device comparison; not end-to-end FPS evidence."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from rotor_cuda_device import DeviceRotorFront


def main(output):
    import torch
    engine=DeviceRotorFront();base=ROOT.parent/'output/rotor_pipeline_20261101/cache'
    windows=[]
    for clip,i in [('clear',8),('side',8),('empty',0)]:
        with np.load(base/clip/f'{i:04d}.npz') as z:
            windows.append((clip,z['events'].copy(),float(z['t'])))
    engine.process(windows[0][1],windows[0][2]);torch.cuda.synchronize()
    records=[]
    for clip,events,t in windows:
        # Warm each shape in both paths. Empty-window lazy setup is otherwise
        # charged only to the first arm and dominates the apparent saving.
        for fused in [False, True]:
            engine.fused_sparse_copy=fused;engine.process(events,t)
        torch.cuda.synchronize()
        for repeat in range(3):
            saved={}
            for fused in ([False,True] if repeat%2==0 else [True,False]):
                engine.fused_sparse_copy=fused;torch.cuda.synchronize();start=time.perf_counter()
                value=engine.process(events,t);torch.cuda.synchronize()
                saved[fused]=value
                records.append(dict(clip=clip,repeat=repeat,fused=fused,ms=(time.perf_counter()-start)*1000))
            assert saved[True][3] == saved[False][3]
            for a,b in zip(saved[True][:3],saved[False][:3]):np.testing.assert_array_equal(a,b)
    result=dict(records=records,exact_sparse_outputs_equal=True,interleaved=True,
        includes_same_device_compute=True,includes_IPC_geometry_decode=False,
        actual_30Hz_or_pipeline_gain_proven=False,
        shape_warmup_each_arm=True,
        source_sha256=hashlib.sha256((ROOT/'core/rotor_cuda_device.py').read_bytes()).hexdigest())
    result['per_clip']={clip:{str(fused):float(np.median([r['ms'] for r in records if r['clip']==clip and r['fused']==fused]))
        for fused in [False,True]} for clip,_,_ in windows}
    for fused in [False,True]:
        a=[r['ms'] for r in records if r['fused']==fused]
        result['fused' if fused else 'three_copy']=dict(mean_ms=float(np.mean(a)),median_ms=float(np.median(a)))
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
