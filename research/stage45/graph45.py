"""Standard ORT CUDA graph/IO binding, fixed addresses and updated current input."""
from pathlib import Path
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage44'))
import common44
OUT=ROOT.parent/'output/stage45_cuda_graph_20261019'
PARAMS=dict(use_tf32=0,enable_cuda_graph=[0,1],intra_threads=2,inter_threads=1,shape=[1,3,384,640],output_shape=[1,5,5040],warmup=5,repetitions=5,
    scope='Existing official ORT CUDA graph sample; fixed shape/batch1/FP32. No mixed precision or detector threshold change. Both binding variants disable TF32, only graph flag differs. Same current image, no temporal batch accumulation.')

def load(p):return common44.load(p)
def save(p,x):return common44.save(p,x)
def sha(p):return common44.sha(p)
def sources():
    result=common44.sources()
    for n in ['graph45.py','benchmark45.py']:result['research/stage45/'+n]=sha(Path(__file__).with_name(n))
    return result
def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];common44.verify();assert sha(common44.OUT/'weights/best.onnx')==p['onnx_sha256'];return p
def freeze():
    assert not (OUT/'protocol.json').exists();p=common44.verify();save(OUT/'protocol.json',dict(utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),source_hashes=sources(),onnx_sha256=sha(common44.OUT/'weights/best.onnx'),parameters=PARAMS,samples=p['samples'],
        design='Freeze after44 ordinary ORT pilot. Existing official CUDA graph and IO binding, no architecture/model training change. Same12 no-GT frames, compare plain run/binding/graph vs PyTorch; record both tensor-ready and full-native-image preprocessing+NMS wall time, five repetitions alternating methods. New input copied into same GPU buffer before each replay, test repeated frame and changed frame for stale-output regression.',
        references=['https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview','https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html']))
    print('Stage45 CUDA graph frozen',flush=True)

class Backend:
    def __init__(self,kind,model_path=None):
        import onnxruntime as ort
        assert kind in ['run','binding','graph'];self.kind=kind;so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1;so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options=dict(use_tf32='0',enable_cuda_graph='1' if kind=='graph' else '0');began=time.perf_counter()
        self.session=ort.InferenceSession(str(model_path if model_path is not None else common44.OUT/'weights/best.onnx'),sess_options=so,providers=[('CUDAExecutionProvider',options)]);assert 'CUDAExecutionProvider' in self.session.get_providers();self.session.disable_fallback();self.input_name=self.session.get_inputs()[0].name;self.output_name=self.session.get_outputs()[0].name
        self.provider_options=self.session.get_provider_options();assert self.provider_options['CUDAExecutionProvider']['use_tf32']=='0';self.binding=None
        if kind!='run':
            self.input=ort.OrtValue.ortvalue_from_numpy(np.zeros(PARAMS['shape'],np.float32),'cuda',0);self.output=ort.OrtValue.ortvalue_from_numpy(np.zeros(PARAMS['output_shape'],np.float32),'cuda',0)
            self.binding=self.session.io_binding();self.binding.bind_ortvalue_input(self.input_name,self.input);self.binding.bind_ortvalue_output(self.output_name,self.output)
            self.session.run_with_iobinding(self.binding);self.binding.synchronize_outputs()
        self.initialization_ms=(time.perf_counter()-began)*1000
    def run(self,x):
        assert x.dtype==np.float32 and tuple(x.shape)==tuple(PARAMS['shape']) and x.flags.c_contiguous
        if self.binding is None:return self.session.run([self.output_name],{self.input_name:x})[0]
        self.input.update_inplace(x);self.binding.synchronize_inputs();self.session.run_with_iobinding(self.binding);self.binding.synchronize_outputs();return self.output.numpy()

def preprocess(image):
    from ultralytics.data.augment import LetterBox
    im=LetterBox(new_shape=(384,640),auto=False,stride=32)(image=image)
    x=np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)[None]).astype(np.float32);x*=np.float32(1/255.)
    return x

def boxes(output,confidence=.25):
    import torch
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    b=non_max_suppression(torch.from_numpy(output),conf_thres=confidence,iou_thres=.45,nc=1)[0];b[:,:4]=scale_boxes((384,640),b[:,:4],(720,1280));return b.numpy().copy()
if __name__=='__main__':freeze()
