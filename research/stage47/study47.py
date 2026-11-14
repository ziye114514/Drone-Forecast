"""RAW shared-memory integration with full/position-only scopes kept explicit."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage47_shared_raw_bridge_20261019'
CASES={'train3_full32':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=2.,precision=32,body_only=False,preview=True),
    'train3_body32':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=2.,precision=32,body_only=True,preview=True),
    'train3_body16':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=2.,precision=16,body_only=True,preview=True),
    'negative_full32':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=1.,precision=32,body_only=False,preview=True),
    'prefix_body16':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=.8,precision=16,body_only=True,preview=False)}
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module():
    spec=importlib.util.spec_from_file_location('shared_raw47',ROOT/'run_shared_body_forecast.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sources():
    m=module();ns,memory,workers=m.build(32,False,True);assert not memory and not workers;result=ns['fingerprints']();result['research/stage47/study47.py']=sha(Path(__file__));return result
def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,
        design='New root shared-memory RAW entry, original42 geometry and six history computations reused as in-memory copies. FP32/FP16 graph artifacts already frozen44/46. Native BGR SharedMemory owner allocates, child attaches read-only, id+ack serializes updates, both SHA full pixels every test frame. No PNG payload, labels/caches or future frames in runtime.',
        comparison='Full32 retains all4 geometry/body fusion branches. body-only scopes omit geometry and duplicate forecast branches, explicitly single body position path, no rotor/pose/physics claim. Precision16 and32 each real RAW2s, same preview and hash-check overhead. Prefix16 real0.8s repeats for causal/numerical check; empty RAW1s for output failure control.',
        verification='Compare native frame hashes with original41 images after runs, full geometry vs41, body outputs vs45/46 corresponding continuous cached references. Report exactness and all floating differences. Videos decode, deadline-over20ms fraction reported. Not another accuracy benchmark or hardware-general realtime claim.',
        references=['https://docs.python.org/3/library/multiprocessing.shared_memory.html','https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview']))
    print('Stage47 RAW shared-memory pilot frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p
def run():
    p=verify();m=module()
    for c,cfg in CASES.items():
        m.run(output=OUT/c,verify_frames=True,**cfg);print('Shared RAW case completed',c,flush=True)
    verify()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);globals()[p.parse_args().action]()
