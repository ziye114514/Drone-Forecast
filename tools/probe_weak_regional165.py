"""One weak-signal mechanism trial, no pixel frequency qualification relaxed."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from rotor_cuda_device import DeviceRotorFront
from rotor_cuda_worker import fill_dense_maps,P

if __name__=='__main__':
    import torch,cv2
    torch.set_num_threads(2);cv2.setNumThreads(2)
    engine=DeviceRotorFront(native_preprocessing=True,native_spatial=True,regional_seeds=True)
    directory=ROOT.parent/'output/native_plane165_20261104';cache=directory/'cache_train2'
    manifest=json.loads((cache/'manifest.json').read_text('utf-8'));records=[]
    cases=[('train2',i) for i in [0,20,23,36]]+[('empty',0),('clear',8),('side',8)]
    for clip,index in cases:
        if clip=='train2':
            item=manifest['frames'][index];e=np.load(cache/item['file']);t=item['t']
        else:
            with np.load(ROOT.parent/f'output/rotor_pipeline_20261101/cache/{clip}/{index:04d}.npz') as z:e=z['events'].copy();t=float(z['t'])
        strong={}
        for weak in [False,True]:
            engine.weak_regional=weak;tick=time.perf_counter();activity,period,native,meta=engine.process(e,t)
            w={k:np.zeros(P,dtype=bool if k in ('dense','selected') else np.uint8 if k=='mask' else np.float32) for k in ['on','off','local','frequency','evidence','on20','off20','dense','selected','mask']}
            diag=fill_dense_maps(activity,period,native,meta,w)
            strong[weak]=period[period[:,5]>0,:2]
            record=dict(clip=clip,index=index,weak=weak,eligible=meta['eligible_pixels'],strict_pixels=meta['accepted_pixels'],
                regional_pixels=meta['regional_seed_pixels'],groups=meta['regional_seed_groups'],
                native=diag['native_pixels'],elapsed_ms=(time.perf_counter()-tick)*1000)
            records.append(record);print(json.dumps(record),flush=True)
            if clip=='train2' and weak:np.savez_compressed(directory/f'weak_train2_{index:04d}.npz',on=w['on'].reshape(720,1280),off=w['off'].reshape(720,1280),native=w['selected'].reshape(720,1280))
        assert np.array_equal(strong[False],strong[True]),'Strong pixel definition changed'
    (directory/'weak_regional.json').write_text(json.dumps(records,indent=2),'utf-8')
