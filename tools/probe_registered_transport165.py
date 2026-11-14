"""Bounded registered native input vs pageable IPC source; no event changes."""
from pathlib import Path
from multiprocessing import shared_memory
import ctypes as C
import sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from rotor_cuda_device import DeviceRotorFront
from rotor_cuda_worker import DTYPE,CAPACITY

if __name__=='__main__':
    import torch
    torch.set_num_threads(2);engine=DeviceRotorFront(native_preprocessing=True,native_spatial=True)
    memory=shared_memory.SharedMemory(create=True,size=CAPACITY*16)
    events=np.ndarray(CAPACITY,DTYPE,buffer=memory.buf);events['p'].fill(0)
    driver=C.WinDLL('nvcuda.dll');registered=False;records=[]
    pointer=C.c_void_p(events.ctypes.data)
    def registration(flag):
        global registered
        if flag==registered:return
        torch.cuda.synchronize()
        code=driver.cuMemHostRegister_v2(pointer,C.c_size_t(memory.size),C.c_uint(0)) if flag else driver.cuMemHostUnregister(pointer)
        if code:raise RuntimeError(f'CUDA host registration {flag} error{code}; no silent fallback')
        registered=flag
    try:
        for clip,index in [('side',0),('clear',8),('empty',0)]:
            with np.load(ROOT.parent/f'output/rotor_pipeline_20261101/cache/{clip}/{index:04d}.npz') as z:e=z['events'].copy();t=float(z['t'])
            np.copyto(events[:len(e)].view(np.uint8),e.view(np.uint8))
            outputs={}
            for pin in [False,True]:registration(pin);engine.process(events[:len(e)],t)
            for pin in [False,True,True,False]:
                registration(pin);torch.cuda.synchronize();tick=time.perf_counter()
                outputs[pin]=engine.process(events[:len(e)],t);torch.cuda.synchronize()
                records.append(dict(clip=clip,registered=pin,events=len(e),ms=(time.perf_counter()-tick)*1000))
            assert np.array_equal(outputs[False][0],outputs[True][0])
            assert np.array_equal(outputs[False][1][:,[0,1,3,4,5]],outputs[True][1][:,[0,1,3,4,5]])
            assert np.array_equal(outputs[False][2],outputs[True][2])
        result=dict(registered_bytes=memory.size,records=records,all_evidence_equal=True,
            medians={c:{str(v):float(np.median([r['ms'] for r in records if r['clip']==c and r['registered']==v])) for v in [False,True]} for c in ['side','clear','empty']})
        Path(sys.argv[1]).write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
    finally:
        registration(False);events=None;memory.close();memory.unlink()
