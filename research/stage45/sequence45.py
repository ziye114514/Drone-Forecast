"""Complete native41 sequence, predictions saved before coarse-box scoring."""
import argparse,datetime,importlib.util,json,os,shutil,sys,time
import cv2,numpy as np
from graph45 import *
INPUT=common44.INPUT;SEQ=OUT/'sequence'
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage41'),str(ROOT/'research/stage23')]

def freeze():
    assert not (SEQ/'protocol.json').exists();p=verify();parent=load(INPUT/'protocol.json');metadata=dict(parent)
    metadata.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=dict(p['source_hashes'],**{'research/stage45/sequence45.py':sha(Path(__file__))}),
        parent45_protocol_sha256=sha(OUT/'protocol.json'),input_manifest_sha256=sha(INPUT/'inputs.json'),model_sha256=sha(common44.OUT/'weights/best.onnx'),
        design='All1550 native41 frames, fixed45 CUDAgraph/use_tf32=0/FP32, CPU preprocessing shown exact to original tensor on12 pilot frames. No threshold selection. Save new independent per-frame neural detections before replay/scoring. Original geometry and public pretrained detections remain frozen references only. Replay41 unchanged in timestamp order, score41 unchanged. Not new blind data.')
    save(SEQ/'protocol.json',metadata);print('Full graph sequence frozen',flush=True)
def verify_seq():
    verify();p=load(SEQ/'protocol.json')
    for name,value in p['source_hashes'].items():assert sha(ROOT/name)==value,name
    assert sha(INPUT/'inputs.json')==p['input_manifest_sha256'];assert sha(common44.OUT/'weights/best.onnx')==p['model_sha256'];return p
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.name=='targets.json' or path.is_relative_to((ROOT.parent/'output/raw_only_project/evaluation_data').resolve()):raise RuntimeError('Inference/replay truth forbidden: '+str(path))
def infer():
    sys.addaudithook(guard);p=verify_seq();assert not (SEQ/'detections.jsonl').exists()
    import torch
    torch.set_num_threads(2);backend=Backend('graph');rows=load(INPUT/'inputs.json')['images'];timing=[];n=0
    with (SEQ/'detections.jsonl').open('x',encoding='utf-8') as f:
        for r in rows:
            assert sha(INPUT/r['image'])==r['image_sha256'];im=cv2.imdecode(np.fromfile(INPUT/r['image'],np.uint8),1);began=time.perf_counter();b=boxes(backend.run(preprocess(im)));timing.append((time.perf_counter()-began)*1000)
            result=[dict(bbox=a[:4].tolist(),confidence=float(a[4]),class_id=int(a[5])) for a in b]
            f.write(json.dumps(dict(id=r['id'],case=r['case'],t=r['t'],method='finetuned',boxes=result),allow_nan=False)+'\n');n+=1
            if n%300==0:print('Graph current detections saved',n,flush=True)
        # Only unchanged public baseline, never a future position proxy.
        for line in (INPUT/'detections.jsonl').read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['method']=='pretrained':f.write(line+'\n')
    for case in p['cases']:
        folder=SEQ/case;folder.mkdir(exist_ok=True);shutil.copyfile(INPUT/case/'geometry.jsonl',folder/'geometry.jsonl')
    save(SEQ/'inference.json',dict(frames=n,predictions_sha256=sha(SEQ/'detections.jsonl'),median_model_chain_ms=float(np.median(timing)),p95_model_chain_ms=float(np.percentile(timing,95)),backend_provider_options=backend.provider_options,backend_initialization_ms=backend.initialization_ms,ground_truth_guard=True,sources_unchanged=verify_seq()['source_hashes']==p['source_hashes']))
    print('All graph detections saved before scores',flush=True)
def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'research/stage41'/filename);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=SEQ;m.verify=verify_seq;return m
def replay():
    sys.addaudithook(guard);verify_seq();module('graph_replay41','replay41.py').main()
def score():
    verify_seq();m=module('graph_score41','score41.py');m.save=lambda path,value:path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()),encoding='utf-8');m.main()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','infer','replay','score']);globals()[p.parse_args().action]()
