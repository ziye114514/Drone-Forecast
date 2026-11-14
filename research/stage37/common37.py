from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT.parent/'output/stage37_upstream_detector_20261018'
CASES={'clear6':dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=2.,labels='train6.json',baseline='stage12_clear_22_27_20261017/observations.jsonl'),
    'side6':dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=12.,labels='train6.json',baseline='stage12_side_108_120_20261017/observations.jsonl'),
    'train3':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=6.,labels='train3.json',baseline='stage31_cross_recording_physics_20261018/train3/observations_forecasts.jsonl'),
    'negative5':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=8.,negative_timestamps=[1.,2.,3.,4.,5.,6.,7.])}
PARAMS=dict(formats=['upstream_dark','native_count_white','sdk_default'],windows_us=[10000,50000],imgsz=640,confidence=.25,iou_nms=.45,batch_size=4,device='0',half=False,
    sample='Every archived label in case intervals, including absent; plus7known-empty train5 timestamps. Cutoff=floor(target_t/10ms)*10ms; only preceding10/50ms events. No GT to model.',
    primary_selection='Maximum confidence box before labels, all outputs retained; IoU0.5 and center<=.25targetDiagonal, any-box IoU diagnostic separately.')


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def sources():
    names=['common37.py','render37.py','checks37.py','study37.py','predict37.py','score37.py','model37.py','audit_load37.py','audit37.py','fetch37.py']
    result={str((Path(__file__).parent/name).relative_to(ROOT)):sha(Path(__file__).parent/name) for name in names}
    for name,value in load(ROOT.parent/'output/stage36_full_side_body_support_20261018/protocol.json')['source_hashes'].items():assert sha(ROOT/name)==value;result[name]=value
    return result


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes']
    for name,value in p['asset_hashes'].items():assert sha(OUT/name)==value
    return p
