"""Fixed native Gaussian/morphology kernels vs installed Torch operations."""
from pathlib import Path
import sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from rotor_cuda_device import DeviceRotorFront


if __name__=='__main__':
    import torch
    torch.set_num_threads(2);engine=DeviceRotorFront(native_preprocessing=True);records=[]
    base=ROOT.parent/'output/rotor_pipeline_20261101/cache';differences=[]
    for clip,indices in [('clear',[0,8,14,23]),('side',[0,8,14,23]),('empty',[0])]:
        for index in indices:
            with np.load(base/clip/f'{index:04d}.npz') as z:e=z['events'].copy();t=float(z['t'])
            outputs={}
            for native in [False,True]:engine.native_spatial=native;engine.process(e,t)
            for native in [False,True,True,False]:
                engine.native_spatial=native;torch.cuda.synchronize();tick=time.perf_counter()
                outputs[native]=engine.process(e,t);torch.cuda.synchronize()
                records.append(dict(clip=clip,index=index,native=native,ms=(time.perf_counter()-tick)*1000))
            a,b=outputs[False],outputs[True]
            assert np.array_equal(a[0],b[0])
            assert np.array_equal(a[1][:,[0,1,3,4,5]],b[1][:,[0,1,3,4,5]])
            equal=np.array_equal(a[2],b[2]);differences.append(dict(clip=clip,index=index,native_equal=equal,counts=[len(a[2]),len(b[2])]))
            print(clip,index,equal,flush=True)
    result=dict(records=records,differences=differences,all_native_equal=all(r['native_equal'] for r in differences),
        medians={c:{str(v):float(np.median([r['ms'] for r in records if r['clip']==c and r['native']==v])) for v in [False,True]} for c in ['clear','side','empty']},
        full_pipeline_FPS_measured=False)
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps({k:v for k,v in result.items() if k!='records'}))
