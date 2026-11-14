from pathlib import Path
import hashlib,json,sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage37'))
import common37
OUT=ROOT.parent/'output/stage38_noise_scale_detector_20261018';OLD=common37.OUT;CASES=common37.CASES
PARAMS=dict(divisors=[1,4,8],activity_filter=[False,True],activity_threshold_us=10000,windows_us=[10000,50000],context_us=60000,
    palette='SDK default matching original FRED frames',imgsz=640,confidence=.25,iou_nms=.45,batch_size=4,device='0',precision='float32',
    projection='Integer floor x/d,y/d centered in native1280x720 canvas, timestamps/polarity retained; predicted bbox mapped by (p-offset)*d, clipped original FOV. Detection center must be in active viewport.',
    primary='Fixed confidence-ranked NMS union across all12 views before GT; all individual views separately scored. No threshold selection from positives.')


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def sources():
    names=['common38.py','render38.py','checks38.py','study38.py','predict38.py','score38.py']
    result={str((Path(__file__).parent/name).relative_to(ROOT)):sha(Path(__file__).parent/name) for name in names}
    for name,value in load(OLD/'protocol.json')['source_hashes'].items():assert sha(ROOT/name)==value;result[name]=value
    return result


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];common37.verify()
    for name,value in p['parent_frozen_artifacts'].items():assert sha(OLD/name)==value
    return p


def viewport(w,h,d):
    if w%d or h%d:raise ValueError('Resolution must be divisible by scale')
    return [(w-w//d)//2,(h-h//d)//2,w//d,h//d]


def source_box(box,view,d,w,h):
    x,y,vw,vh=view;center=[(box[0]+box[2])/2,(box[1]+box[3])/2]
    if not (x<=center[0]<x+vw and y<=center[1]<y+vh):return None
    b=[(box[0]-x)*d,(box[1]-y)*d,(box[2]-x)*d,(box[3]-y)*d]
    b=[max(0,min(w,b[0])),max(0,min(h,b[1])),max(0,min(w,b[2])),max(0,min(h,b[3]))]
    return b if b[2]>b[0] and b[3]>b[1] else None
