"""Reuse frozen45 CUDA graph implementation with only the ONNX artifact changed."""
from pathlib import Path
import datetime,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage45'),str(ROOT/'research/stage44')]
import graph45,common44
from graph45 import load,save,sha
OUT=ROOT.parent/'output/stage46_fp16_graph_20261019';MODEL=OUT/'weights/best_fp16.onnx'
PARAMS=dict(conversion='Installed onnxruntime.transformers.float16.convert_float_to_float16, also used by official Ultralytics CPU16 exporter',keep_io_types=True,min_positive_val=1e-7,max_finite_val=1e4,
    backend='Unchanged45 CUDA graph fixed384x640,batch1,use_tf32=0. FP16 internal weights/ops except converter default blocklist; FP32 input/output. No training/calibration or threshold/input-size changes.',
    timing='Same12 fixed no-GT frames; same process rotating PyTorch32/FP32graph/FP16graph with5warmup/5repetitions. Full1550 current/forecast assessment frozen before results; no default promotion from speed alone.')
def sources():
    result=graph45.sources();result['research/stage45/sequence45.py']=sha(ROOT/'research/stage45/sequence45.py')
    for n in ['backend46.py','study46.py']:result['research/stage46/'+n]=sha(Path(__file__).with_name(n))
    return result
def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];graph45.verify();assert sha(common44.OUT/'weights/best.onnx')==p['original_onnx_sha256'];return p
def freeze():
    assert not (OUT/'protocol.json').exists();p=graph45.verify();save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,original_onnx_sha256=sha(common44.OUT/'weights/best.onnx'),samples=p['samples'],
        design='Fixed one standard16-bit conversion, no label calibration or adaptive precision search. Preserve FP32 artifact; detect finite outputs, stale/current input correctness, actual providers, original/model numeric difference. Predefine full native41 current coverage and exact future endpoint re-evaluation.',
        references=['https://onnxruntime.ai/docs/performance/model-optimizations/float16.html','https://docs.ultralytics.com/modes/export/']))
    print('Stage46 fixed FP16 graph experiment frozen',flush=True)
def fp16_backend(kind='graph'):
    verify();meta=load(OUT/'conversion.json');assert sha(MODEL)==meta['sha256']
    source=(ROOT/'research/stage45/graph45.py').read_text(encoding='utf-8');a="common44.OUT/'weights/best.onnx'";assert source.count(a)==1;source=source.replace(a,'FP16_MODEL_PATH')
    namespace={'__name__':'isolated_fp16_graph46','__file__':str(ROOT/'research/stage45/graph45.py'),'FP16_MODEL_PATH':MODEL};exec(compile(source,namespace['__file__'],'exec'),namespace)
    return namespace['Backend'](kind)
def finite_boxes(raw):
    import numpy as np
    assert np.isfinite(raw).all(),'Nonfinite neural output';return graph45.boxes(raw)
if __name__=='__main__':freeze()
