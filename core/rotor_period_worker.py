"""Persistent adapter for the EXISTING CUDA harmonic estimator; no neural model."""
from pathlib import Path
from multiprocessing import shared_memory
import argparse,json,sys,time
import numpy as np

DTYPE=np.dtype([('x','<u2'),('y','<u2'),('p','<i2'),('t','<i8')])
CAPACITY=200000
H,W=720,1280
PIXELS=H*W
OUTPUT_BYTES=PIXELS*9+CAPACITY*12


def arrays(buffer):
    mask=np.ndarray((H,W),np.uint8,buffer=buffer,offset=0)
    frequency=np.ndarray((H,W),np.float32,buffer=buffer,offset=PIXELS)
    evidence=np.ndarray((H,W),np.float32,buffer=buffer,offset=PIXELS*5)
    pixels=np.ndarray(CAPACITY,np.int32,buffer=buffer,offset=PIXELS*9)
    candidates=np.ndarray((CAPACITY,2),np.float32,buffer=buffer,offset=PIXELS*9+CAPACITY*4)
    return mask,frequency,evidence,pixels,candidates


class CudaPeriodWorker:
    def __init__(self,python):
        import subprocess
        self.input=shared_memory.SharedMemory(create=True,size=CAPACITY*DTYPE.itemsize)
        self.output=shared_memory.SharedMemory(create=True,size=OUTPUT_BYTES)
        self.events=np.ndarray(CAPACITY,DTYPE,buffer=self.input.buf)
        self.planes=arrays(self.output.buf)
        self.process=None
        try:
            self.process=subprocess.Popen([str(python),str(Path(__file__).resolve()),
                '--input-memory',self.input.name,'--output-memory',self.output.name],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=sys.stderr,text=True,bufsize=1)
            ready=json.loads(self.process.stdout.readline())
            if not ready.get('ready'):raise RuntimeError(ready)
        except BaseException:
            self.close();raise

    def __call__(self,events,cutoff_s,window_us=33333,width=1280,height=720,base_frequencies=None):
        if (width,height)!=(W,H):raise ValueError('CUDA shared adapter currently1280x720')
        if len(events)>CAPACITY:
            from predictive_harmonics import predictive_harmonics
            return predictive_harmonics(events,cutoff_s,window_us,width,height,base_frequencies)
        for field in DTYPE.names:self.events[field][:len(events)]=events[field]
        request=dict(n=len(events),t=cutoff_s,window_us=window_us,
            frequencies=None if base_frequencies is None else list(base_frequencies))
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('CUDA harmonic worker exited')
        meta=json.loads(line)
        if 'error' in meta:raise RuntimeError(meta['error'])
        mask,frequency,evidence,pixels,candidates=self.planes;n=meta['eligible_pixels']
        return dict(mask=mask.copy(),frequency_hz=frequency.copy(),evidence=evidence.copy(),
            pixels=pixels[:n].copy(),candidate_frequency_hz=candidates[:n].copy(),
            eligible_pixels=n,accepted_pixels=meta['accepted_pixels'],
            cuda_service_ms=meta['service_ms'],parameters=meta['parameters'])

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.write('{"stop":true}\n');self.process.stdin.flush();self.process.wait(timeout=15)
            except (OSError,ValueError,TimeoutError):
                self.process.terminate();self.process.wait(timeout=10)
            finally:
                self.process.stdin.close();self.process.stdout.close()
        self.events=None;self.planes=None
        for name in ('input','output'):
            block=getattr(self,name,None)
            if block is not None:block.close();block.unlink();setattr(self,name,None)


def worker():
    p=argparse.ArgumentParser();p.add_argument('--input-memory');p.add_argument('--output-memory');args=p.parse_args()
    from predictive_harmonics_cuda import predictive_harmonics_cuda
    # Python3.13 worker must not unlink memory owned by the SDK Python3.11 parent.
    options=dict(track=False) if sys.version_info>=(3,13) else {}
    source=shared_memory.SharedMemory(name=args.input_memory,**options)
    target=shared_memory.SharedMemory(name=args.output_memory,**options)
    events=np.ndarray(CAPACITY,DTYPE,buffer=source.buf);planes=arrays(target.buf)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable in selected GPU interpreter')
    torch.zeros(1,device='cuda');torch.cuda.synchronize()
    print(json.dumps(dict(ready=True)),flush=True)
    try:
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('stop'):break
            try:
                started=time.perf_counter()
                d=predictive_harmonics_cuda(events[:request['n']],request['t'],window_us=request['window_us'],
                    base_frequencies=request['frequencies'])
                n=d['eligible_pixels']
                if n>CAPACITY:raise ValueError('Output capacity exceeded')
                for dst,key in zip(planes[:3],['mask','frequency_hz','evidence']):dst[:]=d[key]
                dst=None
                planes[3][:n]=d['pixels'];planes[4][:n]=d['candidate_frequency_hz']
                print(json.dumps(dict(eligible_pixels=n,accepted_pixels=d['accepted_pixels'],
                    parameters=d['parameters'],service_ms=(time.perf_counter()-started)*1000)),flush=True)
            except Exception:
                import traceback
                print(json.dumps(dict(error=traceback.format_exc())),flush=True)
    finally:
        events=None;planes=None;source.close();target.close()


if __name__=='__main__':worker()
