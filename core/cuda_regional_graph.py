"""Fixed-shape CUDA Graph implementation of corrected208 regional timing.

Regional evidence seeds ORIGINAL count pixels. It never reports per-motor
RPM or grants measured-force credit. No NN, no frame reconstruction.
"""
from pathlib import Path
import ctypes as C
import os,time
import numpy as np
from regional_timing_period import bank,PARAMS
from predictive_harmonics import BASE_FREQUENCIES
from rotor_nvrtc import FusedHarmonics

DTYPE=np.dtype({'names':['x','y','p','t'],'formats':['<u2','<u2','<i2','<i8'],
    'offsets':[0,2,4,8],'itemsize':16})

class RegionalGraph:
    def __init__(self,width=1280,height=720,capacity=4000000):
        import torch as T
        self.T=T;self.w=width;self.h=height;self.capacity=capacity
        self.cols=(width+15)//16;self.rows=(height+15)//16
        self.ng=self.rows*self.cols*2;self.size=width*height
        T.empty(1,device='cuda')  # establish the active Torch driver context
        self.loader=self._load()
        self.host_events=T.empty(capacity*16,dtype=T.uint8,pin_memory=True)
        self.host_header=T.empty(5,dtype=T.int64,pin_memory=True)
        self.events=T.empty_like(self.host_events,device='cuda')
        self.header=T.zeros(5,dtype=T.int64,device='cuda')
        self.ranks=T.empty((self.size*2,32),dtype=T.int64,device='cuda')
        self.counts=T.empty((4,self.size),dtype=T.int32,device='cuda')
        self.seen=T.empty(self.size*2,dtype=T.int32,device='cuda')
        self.spatial=T.empty(self.ng,dtype=T.int32,device='cuda')
        self.hist=T.empty((self.ng,512),dtype=T.float32,device='cuda')
        _,_,spectrum,_,validation=bank(33333)
        self.spectrum=T.as_tensor(spectrum,device='cuda')
        self.validation=T.as_tensor(validation,device='cuda')
        self.frequencies=T.as_tensor(BASE_FREQUENCIES,device='cuda')
        self.x=T.arange(16,device='cuda',dtype=T.float32)-7.5
        self.energy=(self.x*self.x).sum()
        self.kernel=T.ones((1,1,3,3),device='cuda')
        self.threshold=float(np.log(self.ng*2/PARAMS['family_alpha']))
        self.output=T.empty((6,self.size),dtype=T.float32,device='cuda')
        self.summary=T.empty((self.ng,5),dtype=T.float32,device='cuda')
        self.host_output=T.empty_like(self.output,device='cpu',pin_memory=True)
        self.host_summary=T.empty_like(self.summary,device='cpu',pin_memory=True)
        stream=T.cuda.Stream();stream.wait_stream(T.cuda.current_stream())
        with T.cuda.stream(stream):
            for _ in range(3):self._run()
        T.cuda.current_stream().wait_stream(stream);T.cuda.synchronize()
        self.graph=T.cuda.CUDAGraph()
        with T.cuda.graph(self.graph):self._run()
        T.cuda.synchronize()
        # Zero-length capture warmup does not touch real event paths or pinned
        # host pages. Exercise those with label-free synthetic native events.
        rng=np.random.default_rng(211);warm=np.zeros(min(capacity,1000000),DTYPE)
        warm['x']=rng.integers(0,width,len(warm));warm['y']=rng.integers(0,height,len(warm))
        warm['p']=rng.integers(0,2,len(warm));warm['t']=np.arange(len(warm),dtype=np.int64)*33332//max(1,len(warm)-1)
        for _ in range(2):self.measure(warm,.033333)

    def _load(self):
        import torch
        root=Path(os.environ.get('CUDA_PATH','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2'))
        candidates=list((root/'bin').rglob('nvrtc64_*.dll'))
        if not candidates:raise RuntimeError('Existing NVRTC not found')
        loader=FusedHarmonics.__new__(FusedHarmonics);loader.torch=torch
        loader.dll_dir=os.add_dll_directory(str(candidates[0].parent))
        nv=C.WinDLL(str(candidates[0]));loader.driver=C.WinDLL('nvcuda.dll')
        program=C.c_void_p();source=(Path(__file__).resolve().parents[1]/'native/regional_graph211.cu').read_bytes()
        check=loader.check;check(nv.nvrtcCreateProgram(C.byref(program),C.c_char_p(source),C.c_char_p(b'regional_graph211.cu'),0,None,None),'create')
        major,minor=torch.cuda.get_device_capability()
        options=[f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--std=c++14',b'--fmad=false']
        args=(C.c_char_p*len(options))(*options);status=nv.nvrtcCompileProgram(program,len(options),args)
        if status:
            size=C.c_size_t();nv.nvrtcGetProgramLogSize(program,C.byref(size));log=C.create_string_buffer(size.value)
            nv.nvrtcGetProgramLog(program,log);nv.nvrtcDestroyProgram(C.byref(program));raise RuntimeError(log.value.decode())
        size=C.c_size_t();check(nv.nvrtcGetPTXSize(program,C.byref(size)),'PTX size');ptx=C.create_string_buffer(size.value)
        check(nv.nvrtcGetPTX(program,ptx),'PTX');nv.nvrtcDestroyProgram(C.byref(program))
        loader.module=C.c_void_p();check(loader.driver.cuModuleLoadData(C.byref(loader.module),ptx),'load')
        loader.functions={}
        for name in ['regional_rank','regional_histogram']:
            f=C.c_void_p();check(loader.driver.cuModuleGetFunction(C.byref(f),loader.module,C.c_char_p(name.encode())),name)
            loader.functions[name]=f
        return loader

    def _run(self):
        T=self.T;ptr=lambda x:C.c_void_p(x.data_ptr())
        self.ranks.fill_(-1);self.counts.zero_();self.seen.zero_();self.spatial.zero_();self.hist.zero_()
        self.loader.launch('regional_rank',self.capacity,[ptr(x) for x in [self.events,self.header,self.ranks,self.counts,self.seen,self.spatial]])
        self.loader.launch('regional_histogram',self.capacity,[ptr(x) for x in [self.events,self.header,self.ranks,self.hist]])
        a=self.hist[:,:256].reshape(-1,16,16)
        a=a-a.mean(2,keepdim=True)
        a=a-(a*self.x).sum(2,keepdim=True)/self.energy*self.x
        z=(a.reshape(-1,256).to(T.complex64)@self.spectrum).reshape(self.ng,-1,4)
        power=(z.abs()**2).sum(2);first=power.argmax(1)
        separated=(self.frequencies[None]/self.frequencies[first,None]).log().abs()>.2
        second=T.where(separated,power,-1).argmax(1);choices=T.stack((first,second),1)
        coeff=z.gather(1,choices[:,:,None].expand(-1,-1,4))
        predicted=(coeff[:,:,None,:]*self.validation[choices]).sum(3).real.reshape(self.ng,2,16,16)
        g=predicted-predicted.mean(3,keepdim=True)
        g=g-(g*self.x).sum(3,keepdim=True)/self.energy*self.x
        b=self.hist[:,256:].reshape(self.ng,16,16)
        projection=(g*b[:,None]).sum((2,3))
        variance=((g*g).amax(3)*b.sum(2)[:,None]).sum(2)
        bound=g.abs().amax((2,3))
        scores=T.where(projection>0,projection**2/(2*(variance+bound*projection/3).clamp_min(1e-15)),0.)
        choice=scores.argmax(1);best=scores.gather(1,choice[:,None]).squeeze(1)
        na=self.hist[:,:256].sum(1);nb=self.hist[:,256:].sum(1)
        eligible=(na>=6)&(nb>=6)&(self.spatial>=PARAMS['min_spatial_pixels'])
        good=eligible&(best>=self.threshold)
        cells=good.reshape(self.rows,self.cols,2).any(2)
        cells=cells.repeat_interleave(16,0).repeat_interleave(16,1)[:self.h,:self.w]
        total=(self.counts[0]+self.counts[1]).reshape(self.h,self.w)
        active=total>=2
        neighbors=T.nn.functional.conv2d(active.float()[None,None],self.kernel,padding=1)[0,0]
        support=active&(neighbors>=3);seed=support&cells
        self.output[:4].copy_(self.counts);self.output[4].copy_(support.flatten());self.output[5].copy_(seed.flatten())
        selected=choices.gather(1,choice[:,None]).squeeze(1)
        self.summary[:,0].copy_(best);self.summary[:,1].copy_(self.frequencies[selected])
        self.summary[:,2].copy_(eligible);self.summary[:,3].copy_(self.spatial);self.summary[:,4].copy_(good)

    def measure(self,events,cutoff_s,window_us=33333):
        if window_us!=33333:raise ValueError('Graph bins frozen for33333us window')
        end=round(cutoff_s*1e6);begin=end-window_us
        # The data-independent hash is indexed AFTER the exact window filter,
        # matching corrected208, including boundary packets with extra events.
        if len(events) and (events['t'][0]<begin or events['t'][-1]>=end):
            events=events[(events['t']>=begin)&(events['t']<end)]
        if len(events)>self.capacity:raise ValueError('Capacity exceeded; no event truncation')
        e=np.ascontiguousarray(events,dtype=DTYPE)
        if len(e) and (e['x'].max()>=self.w or e['y'].max()>=self.h):raise ValueError('Invalid native coordinates')
        T=self.T;st=time.perf_counter()
        self.host_events.numpy()[:e.nbytes]=e.view(np.uint8).reshape(-1)
        self.host_header.numpy()[:]=[len(e),begin,end,self.w,self.h]
        self.events[:e.nbytes].copy_(self.host_events[:e.nbytes],non_blocking=True)
        self.header.copy_(self.host_header,non_blocking=True);self.graph.replay()
        self.host_output.copy_(self.output,non_blocking=True);self.host_summary.copy_(self.summary,non_blocking=True)
        T.cuda.current_stream().synchronize()
        arrays=self.host_output.numpy().reshape(6,self.h,self.w);summary=self.host_summary.numpy()
        return arrays,summary,dict(graph_ms=1000*(time.perf_counter()-st),events=len(e),threshold=self.threshold,
            qualified_groups=int(summary[:,4].sum()),groups_tested=int(summary[:,2].sum()),
            mechanical_RPM_measured=False,individual_pixel_confirmation=False,
            camera_false_alarm_calibrated=False,all_native_events_counted=True)
