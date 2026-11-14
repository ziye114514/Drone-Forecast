"""Select the candidate model only at the constructor's unique str(path) anchor."""
from backend46 import *
def make_backend(kind='graph'):
    verify();meta=load(OUT/'conversion.json');assert sha(MODEL)==meta['sha256']
    source=(ROOT/'research/stage45/graph45.py').read_text(encoding='utf-8');old="str(common44.OUT/'weights/best.onnx')";assert source.count(old)==1,'Constructor path anchor not unique';source=source.replace(old,'str(FP16_MODEL_PATH)')
    namespace={'__name__':'scoped_fp16_backend46','__file__':str(ROOT/'research/stage45/graph45.py'),'FP16_MODEL_PATH':MODEL};exec(compile(source,namespace['__file__'],'exec'),namespace);return namespace['Backend'](kind)
