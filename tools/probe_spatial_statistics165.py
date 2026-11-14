"""Native Fourier moments prototype, same periodic masks and frozen geometry."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from rotor_cuda_device import DeviceRotorFront
from rotor_spatial_statistics import spatial_statistics

if __name__=='__main__':
    import torch
    torch.set_num_threads(2);engine=DeviceRotorFront(native_preprocessing=True,native_spatial=True);records=[]
    folder=ROOT.parent/'output/native_plane165_20261104'
    geometry=list(map(json.loads,(ROOT.parent/'output/phase_speed163_20261101/clear_candidate_final/predictions.jsonl').read_text('utf-8').splitlines()))
    for index in [0,8,14,23]:
        with np.load(ROOT.parent/f'output/rotor_pipeline_20261101/cache/clear/{index:04d}.npz') as z:e=z['events'].copy();t=float(z['t'])
        outputs={};timing={}
        for phase in [False,True]:engine.phase_statistics=phase;engine.process(e,t)
        for phase in [False,True,True,False]:
            engine.phase_statistics=phase;tick=time.perf_counter();outputs[phase]=engine.process(e,t)
            timing.setdefault(str(phase),[]).append((time.perf_counter()-tick)*1000)
        a,b=outputs[False],outputs[True];assert np.array_equal(a[0],b[0]);assert np.array_equal(a[1][:,[0,1,3,4,5]],b[1][:,[0,1,3,4,5]]);assert np.array_equal(a[2],b[2])
        period=b[1];rotation=spatial_statistics(period[:,0].astype(np.int64),period[:,1],period[:,12:16],geometry[index]['geometry'].get('rotors',[]))
        records.append(dict(index=index,rotation=rotation,ms={k:float(np.median(v)) for k,v in timing.items()},original_evidence_equal=True,
            spectral_score_max_float_diff=float(np.max(np.abs(a[1][:,2]-b[1][:,2]))) if len(a[1]) else 0.))
        print(index,rotation['qualified_count'],flush=True)
    (folder/'spatial_statistics_pilot.json').write_text(json.dumps(records,indent=2),'utf-8');print(json.dumps(records))
