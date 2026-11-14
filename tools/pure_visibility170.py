"""Four reviewed cutoffs, same native cache/GPU front; no NN/coordinate labels."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_cuda_worker import CudaRotorFrontWorker
from rotor_only_front import RotorOnlyFront
from run_rotor_only import serial
OUT=ROOT.parent/'output/geometry170_20261106'

def window(cache,t):
    m=json.loads((cache/'manifest.json').read_text('utf-8'));end=round(t*1e6);begin=end-33333;parts=[]
    for i,r in enumerate(m['frames']):
        prior=m['spec']['start'] if i==0 else m['frames'][i-1]['t']
        if r['t']<begin/1e6 or prior>end/1e6:continue
        a=np.load(cache/r['file']);parts.append(a[(a['t']>=begin)&(a['t']<end)])
    return np.concatenate(parts)

if __name__=='__main__':
    cv2.setNumThreads(2);assert not (OUT/'pure_results.json').exists()
    files=['fast_structure_front.py','rotor_only_front.py','curvature_visibility.py','run_rotor_only.py']
    fingerprints={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True)
    result=[]
    try:
        for label,t in [('A',110.02),('B',111.02),('C',112.02),('D',114.02)]:
            events=window(ROOT.parent/'output/side_temporal164_20261101/cache_side_warm',t)
            prepared=worker.prepare_uncached(events,t);entry=dict(label=label,t=t,arms={})
            for legacy in (True,False):
                front=RotorOnlyFront(worker,fast_geometry=True,legacy_visibility=legacy);front.provide_phase_events=False
                worker.cached=(t,prepared);tick=time.perf_counter();frame=front.step(events[:0],t)
                arm=dict(body=frame['body'],structure=frame['structure'],geometry=frame['geometry'],ms=1000*(time.perf_counter()-tick),
                         NN_used=False,geometry_uses_seeded_native_outline=not legacy)
                entry['arms']['legacy' if legacy else 'fixed']=arm
                np.savez_compressed(OUT/f'pure_{label}_{legacy}.npz',on=frame['on'],off=frame['off'],native=frame['native_mask'],geometry_native=frame['geometry_native_mask'])
                front.previous=None;front.geometry.count_current=None;frame=None;front=None
            result.append(entry);print(label,[(k,a['structure']['mode'],len(a['geometry'].get('rotors',[])),a['structure'].get('view_evidence')) for k,a in entry['arms'].items()],flush=True)
    finally:prepared=None;worker.cached=None;worker.close()
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in fingerprints.items())
    (OUT/'pure_results.json').write_text(json.dumps(dict(records=result,source_sha256=fingerprints,raw_or_live_FPS_measured=False,independent_new_dataset=False),default=serial,indent=2),'utf-8')
