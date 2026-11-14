"""NVRTC CUDA kernel in the active Torch context; no compiler installation."""
from pathlib import Path
import ctypes as C
import os


class FusedHarmonics:
    def __init__(self,warp_reduction=False,reuse_buffers=True):
        import torch
        self.torch=torch
        self.reuse_buffers=bool(reuse_buffers);self.workspace=None;self.spatial_workspace=None
        if os.name!='nt':raise RuntimeError('This NVRTC loader currently targets Windows')
        root=Path(os.environ.get('CUDA_PATH','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2'))
        candidates=list((root/'bin').rglob('nvrtc64_*.dll'))
        if not candidates:raise RuntimeError('Existing CUDA NVRTC library not found')
        self.dll_dir=os.add_dll_directory(str(candidates[0].parent))
        nv=C.WinDLL(str(candidates[0]));self.driver=C.WinDLL('nvcuda.dll')
        program=C.c_void_p();source=(Path(__file__).resolve().parents[1]/'native/rotor_harmonics.cu').read_bytes()
        self.check(nv.nvrtcCreateProgram(C.byref(program),C.c_char_p(source),C.c_char_p(b'rotor_harmonics.cu'),0,None,None),'create')
        major,minor=torch.cuda.get_device_capability()
        options=[f'--gpu-architecture=compute_{major}{minor}'.encode(),b'--std=c++14',b'--fmad=false']
        if warp_reduction:options.append(b'-DROTOR_WARP_REDUCTION')
        args=(C.c_char_p*len(options))(*options)
        status=nv.nvrtcCompileProgram(program,len(options),args)
        if status:
            size=C.c_size_t();nv.nvrtcGetProgramLogSize(program,C.byref(size))
            log=C.create_string_buffer(size.value);nv.nvrtcGetProgramLog(program,log)
            nv.nvrtcDestroyProgram(C.byref(program))
            raise RuntimeError(log.value.decode(errors='replace'))
        size=C.c_size_t();self.check(nv.nvrtcGetPTXSize(program,C.byref(size)),'ptx size')
        ptx=C.create_string_buffer(size.value);self.check(nv.nvrtcGetPTX(program,ptx),'ptx')
        nv.nvrtcDestroyProgram(C.byref(program))
        self.module=C.c_void_p();self.function=C.c_void_p()
        self.check(self.driver.cuModuleLoadData(C.byref(self.module),ptx),'load module')
        self.check(self.driver.cuModuleGetFunction(C.byref(self.function),self.module,C.c_char_p(b'rotor_harmonics')),'get function')
        self.functions={}
        for name in ['event_counts','density_horizontal','density_vertical','gather_pixel_events','rotor_phase_coefficients',
                'native_blur_horizontal','native_feature_vertical','native_feature_horizontal','native_support_vertical','native_binary_morph','native_period_seeds']:
            pointer=C.c_void_p()
            self.check(self.driver.cuModuleGetFunction(C.byref(pointer),self.module,C.c_char_p(name.encode())),name)
            self.functions[name]=pointer

    @staticmethod
    def check(code,where):
        if code:raise RuntimeError(f'CUDA/NVRTC {where} error {code}; no silent reference fallback')

    def __call__(self,times,polarity,pixels,lengths,ntrain,frequencies,window_us):
        T=self.torch;n=len(pixels)
        if len(frequencies)>64:raise ValueError('Fused kernel supports at most64 candidate bases')
        offsets=T.cumsum(lengths,0)-lengths
        output=T.empty((n,12),dtype=T.float32,device=times.device)
        if not n:return output
        tensors=[times,polarity,pixels,lengths,offsets,ntrain,frequencies]
        values=[C.c_void_p(x.data_ptr()) for x in tensors]+[C.c_int(len(frequencies)),C.c_int(window_us),C.c_void_p(output.data_ptr())]
        params=(C.c_void_p*len(values))(*[C.cast(C.byref(x),C.c_void_p) for x in values])
        self.check(self.driver.cuLaunchKernel(self.function,C.c_uint(n),C.c_uint(1),C.c_uint(1),
            C.c_uint(256),C.c_uint(1),C.c_uint(1),C.c_uint(0),
            C.c_void_p(T.cuda.current_stream().cuda_stream),params,None),'launch')
        return output

    def launch(self,name,n,values):
        if not n:return
        params=(C.c_void_p*len(values))(*[C.cast(C.byref(x),C.c_void_p) for x in values])
        self.check(self.driver.cuLaunchKernel(self.functions[name],C.c_uint((n+255)//256),C.c_uint(1),C.c_uint(1),
            C.c_uint(256),C.c_uint(1),C.c_uint(1),C.c_uint(0),
            C.c_void_p(self.torch.cuda.current_stream().cuda_stream),params,None),name)

    def prepare(self,events,begin,end,width,height,minimum_half=6):
        import numpy as np
        T=self.torch;size=width*height
        dtype=np.dtype({'names':['x','y','p','t'],'formats':['<u2','<u2','<i2','<i8'],
            'offsets':[0,2,4,8],'itemsize':16})
        if events.dtype!=dtype:
            packed=np.zeros(len(events),dtype)
            for key in dtype.names:packed[key]=events[key]
        else:packed=np.ascontiguousarray(events)
        if self.reuse_buffers:
            capacity=max(4000000,len(events))
            if self.workspace is None or self.workspace['size']!=size or self.workspace['capacity']<len(events):
                self.workspace=dict(size=size,capacity=capacity,
                    events=T.empty(capacity*16,dtype=T.uint8,device='cuda'),
                    counts=T.empty((8,size),dtype=T.int32,device='cuda'),
                    spatial=T.empty((2,size),dtype=T.int32,device='cuda'),
                    flags=T.empty((2,size),dtype=T.uint8,device='cuda'),
                    times=T.empty(capacity,dtype=T.int64,device='cuda'),
                    polarity=T.empty(capacity,dtype=T.int16,device='cuda'))
            w=self.workspace;e=w['events'][:packed.nbytes]
            e.copy_(T.from_numpy(packed.view(np.uint8).reshape(-1)),non_blocking=True)
            w['counts'].zero_();counts=w['counts'][:2];halves=w['counts'][2:4];counts20=w['counts'][4:6];cursors=w['counts'][6:8]
            horizontal,sums=w['spatial'];dense,eligible=w['flags']
            times=w['times'][:len(events)];polarity=w['polarity'][:len(events)]
        else:
            e=T.as_tensor(packed.view(np.uint8).reshape(-1),device='cuda')
            counts=T.zeros((2,size),dtype=T.int32,device='cuda');halves=T.zeros_like(counts);counts20=T.zeros_like(counts)
            horizontal=T.empty(size,dtype=T.int32,device='cuda');sums=T.empty_like(horizontal)
            dense=T.empty(size,dtype=T.uint8,device='cuda');eligible=T.empty_like(dense)
            times=T.empty(len(events),dtype=T.int64,device='cuda');polarity=T.empty(len(events),dtype=T.int16,device='cuda')
            cursors=T.zeros_like(halves)
        ptr=lambda a:C.c_void_p(a.data_ptr())
        dims=[C.c_int(width),C.c_int(height)]
        head=[ptr(e),C.c_int(len(events)),C.c_longlong(begin),C.c_longlong(end)]+dims
        self.launch('event_counts',len(events),head+[ptr(counts),ptr(halves),ptr(counts20)])
        self.launch('density_horizontal',size,[ptr(counts)]+dims+[ptr(horizontal)])
        self.launch('density_vertical',size,[ptr(horizontal),ptr(counts),ptr(halves)]+dims+[ptr(sums),ptr(dense),ptr(eligible),C.c_int(minimum_half)])
        total=counts[0]+counts[1]
        lengths_all=T.where(eligible>0,total,0).long()
        offsets=T.cumsum(lengths_all,0)-lengths_all
        pixels=T.nonzero(eligible).flatten();lengths=lengths_all[pixels];nt=halves[0,pixels].long()
        # Space bounded by the incoming native packet, not a guessed per-pixel
        # event limit; all selected timestamps are retained.
        self.launch('gather_pixel_events',len(events),head+[ptr(eligible),ptr(offsets),ptr(halves),ptr(cursors),ptr(times),ptr(polarity)])
        return counts[1],counts[0],total,sums,dense>0,pixels,lengths,nt,times,polarity,counts20

    def spatial_support(self,off,strict,g,width,height):
        T=self.torch;size=width*height
        if self.spatial_workspace is None or self.spatial_workspace[0].shape[1]!=size:
            self.spatial_workspace=(T.empty((3,size),dtype=T.float32,device='cuda'),T.empty((3,size),dtype=T.uint8,device='cuda'))
        floating,binary=self.spatial_workspace;ptr=lambda a:C.c_void_p(a.data_ptr());dims=[C.c_int(width),C.c_int(height)]
        self.launch('native_blur_horizontal',2*size,[ptr(off),ptr(floating),ptr(g)]+dims)
        self.launch('native_feature_vertical',size,[ptr(floating),ptr(off),ptr(floating[2]),ptr(g)]+dims)
        self.launch('native_feature_horizontal',size,[ptr(floating[2]),ptr(floating[0]),ptr(g)]+dims)
        self.launch('native_support_vertical',size,[ptr(floating[0]),ptr(binary[0]),ptr(g)]+dims)
        for i,(k,erode) in enumerate([(2,1),(2,0),(3,0),(3,1)]):
            self.launch('native_binary_morph',size,[ptr(binary[i%2]),ptr(binary[1-i%2])]+dims+[C.c_int(k),C.c_int(erode)])
        self.launch('native_period_seeds',size,[ptr(strict),ptr(binary[0]),ptr(binary[2])]+dims)
        return binary[0],binary[2]

    def prepare_cpu_histogram(self,events,begin,end,width,height):
        from native_histogram_front import prepare_histogram
        import numpy as np
        T=self.torch
        h=prepare_histogram(events,begin,end,width,height,getattr(self,'cpu_hist',None));self.cpu_hist=h['hist']
        arrays=[h[k] for k in ['counts','pixels','lengths','offsets','ntrain','times','polarity']]
        amount=sum(a.nbytes for a in arrays)
        if not hasattr(self,'hybrid_host') or len(self.hybrid_host)<amount:
            self.hybrid_host=np.empty(max(amount,64*1024**2),np.uint8)
            self.hybrid_device=T.empty(len(self.hybrid_host),dtype=T.uint8,device='cuda')
        position=0;spans=[]
        for array in arrays:
            a=np.ascontiguousarray(array);self.hybrid_host[position:position+a.nbytes]=a.view(np.uint8).reshape(-1)
            spans.append((position,a.nbytes,a.shape,a.dtype));position+=a.nbytes
        self.hybrid_device[:amount].copy_(T.from_numpy(self.hybrid_host[:amount]),non_blocking=True)
        result=[]
        for position,nbytes,shape,dtype in spans:
            kind={np.dtype('int32'):T.int32,np.dtype('int64'):T.int64,np.dtype('int16'):T.int16}[dtype]
            result.append(self.hybrid_device[position:position+nbytes].view(kind).reshape(shape))
        counts,pixels,lengths,offsets,nt,times,polarity=result
        # __call__ reconstructs compact offsets from lengths, so the exported
        # offsets are retained solely to keep this packing ABI self-describing.
        return counts[1],counts[0],pixels,lengths,nt,times,polarity,h['activity'],h['events']

    def phase_statistics(self,times,polarity,lengths,ntrain,period):
        T=self.torch;n=len(lengths);output=T.empty((n,4),dtype=T.float32,device='cuda')
        offsets=T.cumsum(lengths,0)-lengths;ptr=lambda a:C.c_void_p(a.data_ptr())
        self.launch('rotor_phase_coefficients',n,[ptr(times),ptr(polarity),ptr(lengths),ptr(offsets),ptr(ntrain),ptr(period),C.c_int(n),ptr(output)])
        return output
