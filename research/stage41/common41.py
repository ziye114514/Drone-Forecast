"""Frozen continuous test of a record-disjoint detector, identical history models."""
from pathlib import Path
import datetime,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage40'),str(ROOT/'research/stage21'),str(ROOT/'research/stage23'),str(ROOT/'research/stage22'),str(ROOT/'src')]
import common40
from common40 import load,save,sha
OUT=ROOT.parent/'output/stage41_continuous_body_forecast_20261018';PARENT=common40.OUT
CASES={'clear6':dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=5.),'side6':dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=12.),'train3':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=6.),'negative5':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=8.)}
METHODS=['geometry','pretrained_only','body_only','geometry_first','body_first']
PARAMS=dict(step_us=20000,window_us=20000,activity_threshold_us=10000,noise_warmup_us=10000,confidence=.25,iou_nms=.45,imgsz=640,batch=4,
    body='Highest confidence whole-image box per current tick, no GT or identity hint. body_only uses40 validation-selected detector; geometry_first uses original geometry whenever detected, otherwise body; body_first reverses priority. Rotors from old geometry unchanged, no fake pose/RPM from body.',
    prediction='Unchanged six stage23 causal history models, all methods same5 measurements/160mscoast/500msreset. No learned force feature here; isolate whole-UAV localization.',
    scope='One continuous31s pilot across clear/side/development6, weight-held-out3, and known-empty5. Earlier development saw3; no globally blind claim. Sparse visible-box centers, not COM; no ADE/pose/RPM truth. Training data1/2/4 absent from these intervals.')
def sources():
    result=common40.sources()
    for name in ['common41.py','prepare41.py','predict41.py','replay41.py','score41.py']:result[str((Path(__file__).parent/name).relative_to(ROOT))]=sha(Path(__file__).parent/name)
    return result
def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];common40.verify();assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256'];return p
def freeze():
    assert not (OUT/'protocol.json').exists();common40.verify()
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,parameters=PARAMS,methods=METHODS,parent_protocol_sha256=sha(PARENT/'protocol.json'),label_hashes={f'train{n}.json':sha(common40.LABELS/f'train{n}.json') for n in [3,6]},
        design='Freeze41 during40 training, before test inference40/41 results. Same20ms geometry and history predictors; every original geometry observation and six coefficient outputs must reproduce21/23 or24. Native full-frame SDK ON/OFF filtered with persistent Activity10ms. Offline lossless PNG bridge to installed GPU Python313, no inference labels. Save all continuous predictions before scoring.',
        causality='GPU batches contain independent eval-mode frame detections. Recurrent history replay consumes outputs only in timestamp order, never later detections. Offline batch wall time is not live latency.'))
    print('Stage41 frozen',flush=True)
if __name__=='__main__':freeze()
