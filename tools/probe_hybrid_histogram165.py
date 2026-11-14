"""Paired exact histogram/spectrum check on previously saved native windows."""
from pathlib import Path
import sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from rotor_cuda_device import DeviceRotorFront

if __name__=='__main__':
    import torch
    torch.set_num_threads(2)
    engine=DeviceRotorFront(native_preprocessing=True,native_spatial=True)
    base=ROOT.parent/'output/rotor_pipeline_20261101/cache'
    records=[];checks=[]
    for clip,indices in [('clear',[0,8,14,23]),('side',[0,8,14,23]),('empty',[0])]:
        for index in indices:
            with np.load(base/clip/f'{index:04d}.npz') as z:e=z['events'].copy();t=float(z['t'])
            outputs={}
            for hybrid in [False,True]:engine.hybrid_histogram=hybrid;engine.process(e,t)
            for hybrid in [False,True,True,False]:
                engine.hybrid_histogram=hybrid;torch.cuda.synchronize();tick=time.perf_counter()
                outputs[hybrid]=engine.process(e,t);torch.cuda.synchronize()
                records.append(dict(clip=clip,index=index,hybrid=hybrid,events=len(e),ms=(time.perf_counter()-tick)*1000))
            a,b=outputs[False],outputs[True]
            counts=np.array_equal(a[0][:,[0,1,2,4,5,6]],b[0][:,[0,1,2,4,5,6]])
            period=np.array_equal(a[1][:,[0,1,3,4,5]],b[1][:,[0,1,3,4,5]])
            native=np.array_equal(a[2],b[2])
            check=dict(clip=clip,index=index,events=len(e),counts=counts,period=period,native=native,
                density_max_diff=float(np.max(np.abs(a[0][:,3]-b[0][:,3]))) if len(a[0]) else 0.,backend=b[3]['parameters']['histogram_backend'])
            checks.append(check);print(json.dumps(check),flush=True)
    result=dict(checks=checks,records=records,all_equal=all(c['counts'] and c['period'] and c['native'] for c in checks),
        medians={c:{str(v):float(np.median([r['ms'] for r in records if r['clip']==c and r['hybrid']==v])) for v in [False,True]} for c in ['clear','side','empty']})
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result['medians']))
