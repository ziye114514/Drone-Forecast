"""Paired nine saved native windows, no full pipeline speed claim."""
from pathlib import Path
import sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from rotor_cuda_device import DeviceRotorFront
from rotor_nvrtc import FusedHarmonics


def main(output):
    import torch
    torch.set_num_threads(2)
    engine=DeviceRotorFront(native_preprocessing=True)
    kernels={v:FusedHarmonics(warp_reduction=v) for v in [False,True]}
    base=ROOT.parent/'output/rotor_pipeline_20261101/cache';records=[];max_score_diff=0.
    for clip,indices in [('clear',[0,8,14,23]),('side',[0,8,14,23]),('empty',[0])]:
        for index in indices:
            with np.load(base/clip/f'{index:04d}.npz') as z:e=z['events'].copy();t=float(z['t'])
            for warp in [False,True]:engine.fused_kernel=kernels[warp];engine.process(e,t)
            outputs={}
            for warp in [False,True,True,False]:
                engine.fused_kernel=kernels[warp];torch.cuda.synchronize();tick=time.perf_counter()
                outputs[warp]=engine.process(e,t);torch.cuda.synchronize()
                records.append(dict(clip=clip,index=index,warp=warp,ms=(time.perf_counter()-tick)*1000))
            a,b=outputs[False],outputs[True]
            assert np.array_equal(a[0],b[0]) and np.array_equal(a[2],b[2])
            assert np.array_equal(a[1][:,[0,1,3,4,5]],b[1][:,[0,1,3,4,5]])
            max_score_diff=max(max_score_diff,float(np.max(abs(a[1][:,2]-b[1][:,2]),initial=0)))
    result=dict(records=records,max_score_difference=max_score_diff,
        native_and_selected_frequency_fields_equal=True,device_only=True,
        medians={c:{str(v):float(np.median([r['ms'] for r in records if r['clip']==c and r['warp']==v]))
            for v in [False,True]} for c in ['clear','side','empty']})
    Path(output).write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main(sys.argv[1])
