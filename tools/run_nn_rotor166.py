"""Fixed NN tapes -> native rotor evidence once -> four controlled forecasters."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,time,json,hashlib,copy
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from live_decode import join_native
from rotor_cuda_worker import CudaRotorFrontWorker
from nn_rotor_fusion import NativeNNRotorFeatures,NNRotorFusion,ARMS,unavailable
from run_rotor_only import serial,SOURCE_FILES

BASE=ROOT.parent/'output';OUT=BASE/'nn_rotor166_20261105'
SPECS=[('side',BASE/'stability_audit_20261031/RAW_side_budget8s',BASE/'side_temporal164_20261101/cache_side_warm',114.4),
    ('train2',BASE/'stability_audit_20261031/RAW_other_rollingGC6s',BASE/'native_plane165_20261104/cache_train2_inside',44.7),
    ('clear',BASE/'stability_audit_20261031/RAW_clear_final_clean4s',OUT/'cache_clear',30.)]


class NativeCache:
    def __init__(self,path):
        manifest=json.loads((path/'manifest.json').read_text('utf-8'));self.packets=[];amount=0
        for item in manifest['frames']:
            if manifest.get('storage')=='npy':e=np.load(path/item['file'])
            else:
                with np.load(path/f'{item["index"]:04d}.npz') as z:e=z['events'].copy()
            amount+=e.nbytes
            if amount>1536*1024**2:raise ValueError('Bounded per-clip native cache exceeded1.5GiB')
            self.packets.append(e)
        self.ends=np.rint(np.array([r['t'] for r in manifest['frames']])*1e6).astype(np.int64)
        self.begin=round(manifest['spec']['start']*1e6);self.amount=amount

    def window(self,t):
        end=round(t*1e6);begin=end-33333
        if begin<self.begin or end>self.ends[-1]:return None
        first=int(np.searchsorted(self.ends,begin,side='right'));last=int(np.searchsorted(self.ends,end,side='left'))
        pieces=[]
        for e in self.packets[first:last+1]:
            if not len(e):continue
            lo=0 if e['t'][0]>=begin else int(np.searchsorted(e['t'],begin))
            hi=len(e) if e['t'][-1]<end else int(np.searchsorted(e['t'],end))
            if hi>lo:pieces.append(e[lo:hi])
        result=join_native(pieces,self.packets[0].dtype)
        assert not len(result) or (result['t'][0]>=begin and result['t'][-1]<end)
        return result


def main():
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=True);snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=False)
    names=SOURCE_FILES+['nn_rotor_fusion.py','live_structure_bridge.py','tools/run_nn_rotor166.py']
    hashes={}
    for name in names:
        b=(ROOT/name).read_bytes();p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);hashes[name]=hashlib.sha256(b).hexdigest()
    protocol=dict(source_sha256=hashes,labels_used_for_runtime=False,detector_retrained=False,
        predictions_feedback_into_detection=False,NN_tapes_frozen=True,arms=ARMS,
        full_RAW_or_live_NN_FPS_measured=False,parameters_swept=False,clips={})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf-8')
    worker=CudaRotorFrontWorker('C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',native_front=True,remote_unpack=True)
    summary={}
    try:
        for name,nn_path,cache_path,end in SPECS:
            print('START',name,flush=True);cache=NativeCache(cache_path);features=NativeNNRotorFeatures(worker)
            paths={arm:(OUT/f'{name}_{arm}.jsonl').open('w',encoding='utf-8') for arm in ARMS}
            observer={arm:NNRotorFusion(arm) for arm in ARMS};counts={arm:dict(valid=0,force=0,center=0) for arm in ARMS}
            rows=[json.loads(s) for s in (nn_path/'predictions.jsonl').read_text('utf-8').splitlines()];rows=[r for r in rows if r['t']<=end]
            feature_times=[];costs={arm:[] for arm in ARMS};modes={};native_frames=0
            try:
                with (OUT/f'{name}_observations.jsonl').open('w',encoding='utf-8') as tape:
                    for index,r in enumerate(rows):
                        events=cache.window(r['t']);tick=time.perf_counter()
                        if events is None:observation=unavailable(r['t'],r['body'],'outside_full_native_cache_window')
                        else:
                            observation=features.step(events,r['t'],r['body']);native_frames+=1
                            observation.update(native_event_count=len(events),native_begin_us=round(r['t']*1e6)-33333,native_end_us=round(r['t']*1e6))
                        feature_times.append((time.perf_counter()-tick)*1000)
                        observation['fixed_current_flow']=copy.deepcopy((r.get('image_motion') or {}).get('observation',{}))
                        tape.write(json.dumps(observation,default=serial,allow_nan=False)+'\n')
                        mode=observation['structure']['mode'];modes[mode]=modes.get(mode,0)+1
                        for arm in ARMS:
                            tick=time.perf_counter();p=observer[arm].step(observation);costs[arm].append((time.perf_counter()-tick)*1000)
                            paths[arm].write(json.dumps(p,default=serial,allow_nan=False)+'\n')
                            counts[arm]['valid']+=p['valid'];counts[arm]['force']+=p['force']['active'];counts[arm]['center']+=p['center_bridge']['applied']
                        if index%60==0:print(name,index,round(r['t'],4),mode,flush=True)
            finally:
                for p in paths.values():p.close()
            summary[name]=dict(frames=len(rows),native_frames=native_frames,modes=modes,arms=counts,
                feature_mean_ms=float(np.mean(feature_times)),feature_p95_ms=float(np.percentile(feature_times,95)),
                arm_mean_ms={a:float(np.mean(v)) for a,v in costs.items()},cache_bytes=cache.amount,
                original_NN_run=str(nn_path),NN_sha256=hashlib.sha256((nn_path/'predictions.jsonl').read_bytes()).hexdigest())
            print(json.dumps({name:summary[name]}),flush=True)
            features.geometry.count_current=None;features=None;cache=None;events=None
            (OUT/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
    finally:worker.close()
    changed=[name for name,digest in hashes.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    protocol.update(source_changed_during_run=changed,clips=summary)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf-8')
    if changed:raise RuntimeError('Source changed during inference: '+str(changed))


if __name__=='__main__':main()
