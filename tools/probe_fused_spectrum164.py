"""Native fused spectrum vs frozen causal Torch implementation."""
from pathlib import Path
import sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rotor_cuda_device import DeviceRotorFront


def main(output):
    import torch
    torch.set_num_threads(2);engine=DeviceRotorFront()
    base=ROOT.parent/'output/rotor_pipeline_20261101/cache';records=[]
    for clip,indices in [('clear',[0,8,14,23]),('side',[0,8,14,23]),('empty',[0])]:
        for index in indices:
            with np.load(base/clip/f'{index:04d}.npz') as z:e=z['events'].copy();t=float(z['t'])
            for fused in [False,True]:
                engine.fused_harmonics=fused;engine.native_preprocessing=fused;engine.process(e,t)
            outputs={}
            for fused in [False,True,True,False]:
                engine.fused_harmonics=fused;engine.native_preprocessing=fused;torch.cuda.synchronize();start=time.perf_counter()
                outputs[fused]=engine.process(e,t);torch.cuda.synchronize()
                records.append(dict(clip=clip,index=index,fused=fused,ms=1000*(time.perf_counter()-start)))
            a,b=outputs[False],outputs[True]
            assert np.array_equal(a[0],b[0][:,:5])
            end=round(t*1e6);recent=e[(e['t']>=end-20000)&(e['t']<end)]
            ix=recent['y'].astype(np.int64)*1280+recent['x']
            for p,col in [(1,5),(0,6)]:
                expected=np.bincount(ix[recent['p']==p],minlength=1280*720)
                np.testing.assert_array_equal(b[0][:,col],expected[b[0][:,0].astype(np.int64)])
            assert np.array_equal(a[2],b[2]),(clip,index,'native mask')
            assert np.array_equal(a[1][:,[0,1,3,4,5]],b[1][:,[0,1,3,4,5]]),(clip,index,'selected frequencies')
            print(clip,index,max(abs(a[1][:,2]-b[1][:,2]),default=0.),flush=True)
    r=dict(records=records,all_nine_sparse_activity_native_and_selected_frequencies_equal=True,
        medians_by_clip={c:{str(v):float(np.median([r['ms'] for r in records if r['clip']==c and r['fused']==v]))
            for v in [False,True]} for c in ['clear','side','empty']},no_full_pipeline_FPS_claim=True)
    Path(output).write_text(json.dumps(r,indent=2),'utf-8');print(json.dumps(r))


if __name__=='__main__':main(sys.argv[1])
