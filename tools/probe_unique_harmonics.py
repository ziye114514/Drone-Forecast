"""Same causal windows: shared harmonic coefficients vs the full comb bank."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rotor_cuda_device import DeviceRotorFront


def main(output):
    import torch
    torch.set_num_threads(2)
    engine=DeviceRotorFront();base=ROOT.parent/'output/rotor_pipeline_20261101/cache'
    records=[];differences=[]
    for clip,indices in [('clear',[0,8,14,23]),('side',[0,8,14,23]),('empty',[0])]:
        for index in indices:
            with np.load(base/clip/f'{index:04d}.npz') as z:e=z['events'].copy();t=float(z['t'])
            for option in [False,True]:
                engine.unique_harmonics=option;engine.process(e,t)
            for repeat in range(2):
                saved={}
                for option in ([False,True] if repeat==0 else [True,False]):
                    engine.unique_harmonics=option;torch.cuda.synchronize();start=time.perf_counter()
                    saved[option]=engine.process(e,t);torch.cuda.synchronize()
                    records.append(dict(clip=clip,index=index,repeat=repeat,unique=option,ms=1000*(time.perf_counter()-start)))
                a,b=saved[False],saved[True]
                assert np.array_equal(a[0],b[0])
                assert np.array_equal(a[2],b[2])
                assert np.array_equal(a[1][:,[0,1,3,4,5]],b[1][:,[0,1,3,4,5]])
                differences.append(float(np.max(np.abs(a[1][:,2]-b[1][:,2]))) if len(a[1]) else 0.)
    result=dict(records=records,activity_native_and_selected_frequencies_equal=True,
        maximum_spectral_score_difference=max(differences),
        old_frequency_harmonic_pairs=57*4,unique_coefficients=len(engine.spectral_f),
        means_ms={str(v):float(np.mean([r['ms'] for r in records if r['unique']==v])) for v in [False,True]},
        per_clip={c:{str(v):float(np.median([r['ms'] for r in records if r['unique']==v and r['clip']==c]))
            for v in [False,True]} for c in ['clear','side','empty']},
        full_pipeline_30FPS_proven=False)
    Path(output).write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
