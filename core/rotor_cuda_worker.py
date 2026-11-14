"""Persistent sparse CUDA front adapter across SDK/CUDA Python interpreters."""
from pathlib import Path
from multiprocessing import shared_memory
import argparse, json, subprocess, sys, time
import numpy as np

H,W = 720,1280
P = H*W
CAPACITY = 4000000
DTYPE = np.dtype({'names':['x','y','p','t'],'formats':['<u2','<u2','<i2','<i8'],
    'offsets':[0,2,4,8],'itemsize':16})


def planes(buffer):
    return (np.ndarray((P,7),np.float32,buffer=buffer,offset=0),
            np.ndarray((P,12),np.float32,buffer=buffer,offset=P*28),
            np.ndarray((P,2),np.float32,buffer=buffer,offset=P*76))


def dense_planes(buffer,slot):
    base=slot*P*31
    floats=np.ndarray((7,P),np.float32,buffer=buffer,offset=base)
    bits=np.ndarray((3,P),np.uint8,buffer=buffer,offset=base+P*28)
    return dict(zip(['on','off','local','frequency','evidence','on20','off20'],floats),
        dense=bits[0].view(bool),mask=bits[1],selected=bits[2].view(bool))


def fill_dense_maps(activity,period,native,meta,w):
    for a in w.values():a.fill(0)
    on,off,local,dense=[w[k] for k in ['on','off','local','dense']]
    ap=activity[:,0].astype(np.int32)
    on[ap]=activity[:,1];off[ap]=activity[:,2];local[ap]=activity[:,3];dense[ap]=activity[:,4]>0
    pixels=period[:,0].astype(np.int32)
    w['mask'][pixels]=period[:,5].astype(np.uint8)*255
    w['frequency'][pixels]=period[:,1];w['evidence'][pixels]=period[:,2]
    selected=w['selected']
    if len(native) and (meta['accepted_pixels'] or meta.get('regional_seed_pixels',0)):
        import cv2
        mp=native[:,0].astype(np.int32);ys,xs=mp//W,mp%W
        x,y,X,Y=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
        roi=np.zeros((Y-y,X-x),np.uint8);roi[ys-y,xs-x]=1
        n,labels=cv2.connectedComponents(roi,connectivity=8)
        seed=native[:,1]>0;hits=np.unique(labels[ys[seed]-y,xs[seed]-x]);hits=hits[hits>0]
        lookup=np.zeros(n,bool);lookup[hits]=True
        selected[mp]=lookup[labels[ys-y,xs-x]]
    support_pixels=int(selected.sum());selected&=(on+off)>0
    if activity.shape[1]==7:w['on20'][ap]=activity[:,5];w['off20'][ap]=activity[:,6]
    return dict(strict_pixels=meta['accepted_pixels'],native_pixels=int(selected.sum()),
        regional_seed_groups=meta.get('regional_seed_groups',0),regional_seed_pixels=meta.get('regional_seed_pixels',0),
        mask_support_pixels=support_pixels,seed_pixels=int(native[:,1].sum()),
        added_native_pixels_not_period_confirmed=True,mask_boundary_not_used_as_physical_rim=True,
        spatial_backend='CUDA_blur_morphology_sparse_CPU_connected_components')


class CudaRotorFrontWorker:
    def __init__(self, python, fused_spectrum=False, native_front=False,regional_seeds=False,remote_unpack=False,slots=3,weak_regional=False,regional_graph=False):
        if not 3<=int(slots)<=6:raise ValueError('Dense feature slots must be3..6')
        self.slots=int(slots)
        self.input = shared_memory.SharedMemory(create=True,size=CAPACITY*DTYPE.itemsize)
        self.output = shared_memory.SharedMemory(create=True,size=P*84)
        self.events = np.ndarray(CAPACITY,DTYPE,buffer=self.input.buf)
        self.planes = planes(self.output.buf); self.process = None
        # Commit shared pages before the source-time clock, instead of paying
        # first-touch page faults across the first several real observations.
        np.ndarray(CAPACITY*DTYPE.itemsize,np.uint8,buffer=self.input.buf)[::4096]=0
        np.ndarray(P*84,np.uint8,buffer=self.output.buf)[::4096]=0
        self.remote_unpack=bool(remote_unpack);self.dense_memory=None
        self.cached=None;self.workspace_index=0
        self.workspaces=[dict(on=np.zeros(P,np.float32),off=np.zeros(P,np.float32),local=np.zeros(P,np.float32),
            dense=np.zeros(P,bool),mask=np.zeros(P,np.uint8),frequency=np.zeros(P,np.float32),evidence=np.zeros(P,np.float32),
            selected=np.zeros(P,bool),on20=np.zeros(P,np.float32),off20=np.zeros(P,np.float32)) for _ in range(self.slots)]
        if self.remote_unpack:
            self.dense_memory=shared_memory.SharedMemory(create=True,size=self.slots*P*31)
            self.workspaces=[dense_planes(self.dense_memory.buf,i) for i in range(self.slots)]
        try:
            self.process = subprocess.Popen([str(python),str(Path(__file__).resolve()),
                '--input-memory',self.input.name,'--output-memory',self.output.name]+(['--fused-spectrum'] if fused_spectrum else [])+(['--native-front'] if native_front else [])+(['--regional-seeds'] if regional_seeds else [])+(['--weak-regional'] if weak_regional else [])+(['--regional-graph'] if regional_graph else [])+(['--dense-memory',self.dense_memory.name,'--dense-slots',str(self.slots)] if self.remote_unpack else []),
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=sys.stderr,text=True,bufsize=1)
            meta = json.loads(self.process.stdout.readline())
            if not meta.get('ready'): raise RuntimeError(meta)
        except BaseException:
            self.close(); raise

    def prepare_rotor_support(self, events, cutoff_s, width=1280, height=720, window_us=33333):
        if self.cached is not None:
            t,result=self.cached
            if abs(t-cutoff_s)>1e-9:raise ValueError('Prefetched features do not match current source time')
            self.cached=None;return result
        return self.prepare_uncached(events,cutoff_s,width,height,window_us)

    def prepare_uncached(self, events, cutoff_s, width=1280, height=720, window_us=33333):
        if (width,height)!=(W,H): raise ValueError('Native front currently1280x720')
        if len(events)>CAPACITY: raise ValueError('Native event capacity exceeded; no silent CPU fallback or event truncation')
        if events.dtype==DTYPE and events.flags.c_contiguous:
            # Canonical SDK ABI is already checked. Copy one contiguous native
            # byte span rather than running a strided structured-field cast.
            np.copyto(self.events[:len(events)].view(np.uint8),events.view(np.uint8))
        elif events.dtype==DTYPE:
            np.copyto(self.events[:len(events)],events)
        else:
            for k in DTYPE.names: self.events[k][:len(events)] = events[k]
        started = time.perf_counter()
        request=dict(n=len(events),t=cutoff_s,window_us=window_us)
        self.process.stdin.write(json.dumps(request)+'\n'); self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line: raise RuntimeError('CUDA native front exited')
        meta=json.loads(line)
        if 'error' in meta: raise RuntimeError(meta['error'])
        activity,period,native = [a[:n] for a,n in zip(self.planes,meta['sizes'])]
        activity=activity[:,:meta.get('activity_columns',5)]
        period=period[:,:meta.get('period_columns',6)]
        slot=meta.get('dense_slot',self.workspace_index%self.slots)
        w=self.workspaces[slot];self.workspace_index+=1
        diagnostic=meta.pop('dense_diagnostic') if self.remote_unpack else fill_dense_maps(activity,period,native,meta,w)
        on,off,local,dense=[w[k] for k in ['on','off','local','dense']]
        pixels=period[:,0].astype(np.int32)
        mask,frequency,evidence=[w[k] for k in ['mask','frequency','evidence']]
        result=dict(mask=mask.reshape(H,W),frequency_hz=frequency.reshape(H,W),evidence=evidence.reshape(H,W),
            pixels=pixels,candidate_frequency_hz=period[:,3:5].copy(),
            eligible_pixels=meta['eligible_pixels'],accepted_pixels=meta['accepted_pixels'],parameters=meta['parameters'])
        if period.shape[1]==12:result['first_harmonic_statistics']=period[:,6:10].copy()
        selected=w['selected']
        meta['cuda_roundtrip_ms']=(time.perf_counter()-started)*1000
        meta['sparse_transfer_bytes']=int(sum(a.size*a.itemsize for a in (activity,period,native)))
        meta['native_precomputed']=(selected.reshape(H,W),diagnostic)
        if activity.shape[1]==7:
            on20,off20=w['on20'],w['off20']
            meta['native_counts20']=(on20.reshape(H,W),off20.reshape(H,W))
        return on.reshape(H,W),off.reshape(H,W),meta,dense.reshape(H,W),local.reshape(H,W),result

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.write('{"stop":true}\n');self.process.stdin.flush();self.process.wait(timeout=15)
            except (OSError,ValueError,subprocess.TimeoutExpired):
                self.process.terminate();self.process.wait(timeout=10)
            finally:
                self.process.stdin.close();self.process.stdout.close();self.process=None
        self.events=None;self.planes=None;self.cached=None;self.workspaces=[]
        for name in ('input','output','dense_memory'):
            block=getattr(self,name,None)
            if block is not None:block.close();block.unlink();setattr(self,name,None)


def worker():
    p=argparse.ArgumentParser();p.add_argument('--input-memory');p.add_argument('--output-memory');p.add_argument('--dense-memory');p.add_argument('--dense-slots',type=int,default=3);p.add_argument('--fused-spectrum',action='store_true');p.add_argument('--native-front',action='store_true');p.add_argument('--regional-seeds',action='store_true');p.add_argument('--weak-regional',action='store_true')
    p.add_argument('--regional-graph',action='store_true')
    a=p.parse_args()
    from rotor_cuda_device import DeviceRotorFront
    options=dict(track=False) if sys.version_info>=(3,13) else {}
    source=shared_memory.SharedMemory(name=a.input_memory,**options)
    target=shared_memory.SharedMemory(name=a.output_memory,**options)
    events=np.ndarray(CAPACITY,DTYPE,buffer=source.buf);views=planes(target.buf)
    dense_memory=None;workspaces=None;tick=0
    if a.dense_memory:
        import cv2
        cv2.setNumThreads(2)
        dense_memory=shared_memory.SharedMemory(name=a.dense_memory,**options)
        workspaces=[dense_planes(dense_memory.buf,i) for i in range(a.dense_slots)]
    import torch
    torch.set_num_threads(2)
    engine=DeviceRotorFront(fused_harmonics=a.fused_spectrum,native_preprocessing=a.native_front,native_spatial=a.native_front,regional_seeds=a.regional_seeds,weak_regional=a.weak_regional)
    if a.regional_graph:
        from regional_graph_front import RegionalGraphFallback
        engine=RegionalGraphFallback(engine)
    # Warm all spatial/spectral paths using native-like synthetic events, not
    # test labels; no warm-up observations enter the runtime estimator.
    warm=np.zeros(64,DTYPE);warm['x']=100;warm['y']=100
    warm['t']=np.linspace(0,33332,64).astype(np.int64);warm['p']=np.arange(64)%2
    engine.process(warm,.033333);torch.cuda.synchronize()
    if a.native_front:
        # A one-pixel warmup does not exercise transfer buckets, sparse output
        # sizes or realistic GPU load. Use synthetic periodic patches among
        # native-like background events; never read test frames for warmup.
        count=1152*64*14;warm=np.empty(count,DTYPE);rng=np.random.default_rng(165)
        warm['x']=rng.integers(0,W,count,dtype=np.uint16);warm['y']=rng.integers(0,H,count,dtype=np.uint16)
        warm['t']=np.arange(count,dtype=np.int64)*33332//(count-1)
        warm['p']=np.cos(2*np.pi*600*warm['t']/1e6)>=0
        selected=np.arange(0,count,14);pixel=np.arange(len(selected))%1152;patch=pixel//9
        warm['x'][selected]=100+(patch%16)*8+pixel%3
        warm['y'][selected]=100+(patch//16)*8+(pixel%9)//3
        for _ in range(2):engine.process(warm,.033333)
        torch.cuda.synchronize();del warm,selected,pixel,patch
    if workspaces is not None:
        for workspace in workspaces:
            for array in workspace.values():array.fill(0)
        # Initialize the same native connected-component backend with synthetic
        # support; no warmup sample enters the estimator or published results.
        scratch=np.zeros((128,128),np.uint8);scratch[20:100,20:100]=1
        cv2.connectedComponents(scratch,connectivity=8)
    print(json.dumps(dict(ready=True)),flush=True)
    try:
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('stop'):break
            try:
                started=time.perf_counter()
                activity,period,native,meta=engine.process(events[:request['n']],request['t'],request['window_us'])
                if workspaces is not None:
                    slot=tick%a.dense_slots;tick+=1;unpack_started=time.perf_counter()
                    meta['dense_diagnostic']=fill_dense_maps(activity,period,native,meta,workspaces[slot])
                    meta['dense_slot']=slot;meta['dense_unpack_ms']=(time.perf_counter()-unpack_started)*1000
                for dst,src in zip(views,(activity,period,native)):dst[:len(src),:src.shape[1]]=src
                meta.update(sizes=[len(activity),len(period),len(native)],activity_columns=activity.shape[1],period_columns=period.shape[1],cuda_service_ms=(time.perf_counter()-started)*1000)
                print(json.dumps(meta),flush=True)
            except Exception:
                import traceback
                print(json.dumps(dict(error=traceback.format_exc())),flush=True)
    finally:
        events=None;views=None;source.close();target.close()
        workspaces=None
        if dense_memory is not None:dense_memory.close()


if __name__=='__main__':worker()
