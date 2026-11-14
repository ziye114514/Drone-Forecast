"""Current-event extraction audit. Instrument the old frontend without changing it."""
from pathlib import Path
import argparse, copy, datetime, hashlib, json, sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'research/stage21')]
from windows21 import Observer, RecoveryFrontend
from event_drone_forecast.raw_io import open_raw, raw_chunks

OUT = ROOT.parent / 'output/stage109_rotor_extraction_audit_20261022'
CASES = {
    'continuous6': dict(raw='E:/dataset/train/6/media/event.raw', start=110., duration=9.5,
        reference='output/stage106_continuous_screen_20261022/raw/observations_forecasts.jsonl'),
    'clear6': dict(raw='E:/dataset/train/6/media/event.raw', start=22., duration=1.,
        reference='output/stage104_joint_center_20261022/raw/clear6/observations_forecasts.jsonl'),
    'negative5': dict(raw='E:/dataset/train/5/media/event.raw', start=0., duration=.6,
        reference='output/stage107_center_bias_filter_20261022/raw/negative5/observations_forecasts.jsonl')}
TIMES = [110.02, 111.02, 112.02, 114.02, 116.52, 118.02, 22.5, .5]

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()), encoding='utf-8')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open(encoding='utf-8') as f: return [json.loads(s) for s in f if s.strip()]
def sources():
    paths = [Path(__file__), ROOT/'research/stage21/windows21.py', ROOT/'research/stage9/recovery.py',
        ROOT/'research/stage8/polarity.py', ROOT/'src/event_drone_forecast/perception.py',
        ROOT/'src/event_drone_forecast/rotor_sensing.py', ROOT/'src/event_drone_forecast/raw_io.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}

def freeze():
    assert not OUT.exists()
    save(OUT/'protocol109.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources=sources(), cases=CASES, fixed_snapshot_times=TIMES,
        design='Trace native20 bipolar feature, global threshold, morphology, contours, actual accepted candidates and actual grouping. Capture only; unchanged old detector/state. Exact geometry20 replay required.',
        selection='C112.02 user reports four visible disks; earlier current10/current20 produced three. Development diagnostic, not blind evaluation. All fixed continuous110-119.5 retained.',
        inputs='Re-read RAW. Existing contemporaneous NN boxes used only to crop audit displays; not passed to old geometry detector. No future targets, frequencies, hidden hubs or new forecast claim.',
        followup='Check standard distance-transform/watershed and arc-support ellipse detectors before adding any event-specific method. Generic splitting itself is not claimed novel.', default_promoted=False))

def compact(r):
    return {k: copy.deepcopy(v) for k,v in r.items() if k not in ['normal','bbox']}

class Audit(RecoveryFrontend):
    def begin(self, capture=False):
        self.capture=capture; self.calls=[]; self.grouped=[]; self.selected=[]
    def feature(self,on,off):
        f=super().feature(on,off)
        if self.capture: self.feature_image=f.copy()
        return f
    def _candidates(self,contours,activity,t,b,thin_only=False):
        result=super()._candidates(contours,activity,t,b,thin_only)
        if self.capture:
            self.calls.append(dict(thin_only=thin_only,
                contours=[c.copy() for c in contours], activity=activity.copy(),
                accepted=[copy.deepcopy(c) for c in result]))
        return result
    def _group(self,candidates,t):
        result=super()._group(candidates,t)
        if self.capture:
            self.grouped=[copy.deepcopy(r) for r in candidates]
            self.selected=[] if result is None else [copy.deepcopy(r) for r in result]
        return result

def audit_frame(front,body,events,t,folder):
    f=front.feature_image.copy(); on=front.prepared['on'];off=front.prepared['off']
    f[on+off<front.cfg.min_pixel_count]=0
    density=cv2.GaussianBlur(f,(3,3),.8); pos=density[density>0]
    threshold=max(2.,float(np.quantile(pos,front.cfg.density_quantile))) if len(pos) else 2.
    mask=np.uint8(density>=threshold)*255
    opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    closed=cv2.morphologyEx(opened,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    b=body.get('bbox')
    lo=np.maximum(np.floor(np.asarray(b[:2])-50).astype(int),0) if b else np.array([0,0])
    hi=np.minimum(np.ceil(np.asarray(b[2:])+50).astype(int),[front.w,front.h]) if b else np.array([front.w,front.h])
    x0,y0=lo;x1,y1=hi
    np.savez_compressed(folder/f'{t:.2f}_counts.npz',on=on[y0:y1,x0:x1],off=off[y0:y1,x0:x1],
        feature=f[y0:y1,x0:x1],density=density[y0:y1,x0:x1],raw_mask=mask[y0:y1,x0:x1],
        opened=opened[y0:y1,x0:x1],closed=closed[y0:y1,x0:x1],origin=lo,threshold=threshold)
    records=[]
    for call in front.calls:
        for contour in call['contours']:
            x,y,w,h=cv2.boundingRect(contour)
            if x+w<x0 or x>x1 or y+h<y0 or y>y1: continue
            records.append(dict(thin_only=call['thin_only'],bbox=[x,y,x+w,y+h],
                area=cv2.contourArea(contour),points=len(contour),contour=contour[:,0,:].tolist(),
                accepted=[compact(r) for r in call['accepted'] if np.linalg.norm(r['center']-[x+w/2,y+h/2])<max(w,h)]))
    save(folder/f'{t:.2f}_audit.json',dict(t=t,crop=np.r_[lo,hi],threshold=threshold,
        global_nonzero_bins=len(pos),contours=records,ranked_candidates=[compact(r) for r in front.grouped],
        selected=[compact(r) for r in front.selected],actual_current_body=body))

def run(case):
    protocol=json.loads((OUT/'protocol109.json').read_text('utf-8'));assert protocol['sources']==sources()
    cfg=CASES[case];folder=OUT/case;folder.mkdir(exist_ok=False)
    reference={round(r['t'],8):r for r in rows(ROOT.parent/cfg['reference'])}
    reader=open_raw(cfg['raw']);h,w=reader.get_size();old=Observer(w,h,'native20');old.geometry=Audit(w,h,'neighbor10')
    begin=max(0.,cfg['start']-.1);duration=cfg['duration']+cfg['start']-begin
    result=[]
    for t,e in raw_chunks(reader,begin,duration,20000):
        if t-begin<.1-1e-8:
            obs=dict(t=t,detected=False,rotors=[])
        else:
            capture=any(abs(t-v)<1e-8 for v in TIMES);old.geometry.begin(capture)
            obs=old.step(e,t)['observation']
        if t<=cfg['start']+1e-8: continue
        r=reference[round(t,8)];assert obs==r['geometry20'],(case,t,'audit altered old geometry')
        result.append(dict(t=t,detected=obs['detected'],rotors=obs.get('rotors',[]),
            observed_count=len(obs.get('rotors',[])),body=r['body_observation']))
        if getattr(old.geometry,'capture',False):audit_frame(old.geometry,r['body_observation'],e,t,folder)
    assert len(result)==round(cfg['duration']*50)
    save(folder/'audit109.json',dict(records=result,exact_old_geometry=True,labels_read=False,
        raw_input=True,source_unchanged=protocol['sources']==sources()))
    print(json.dumps(dict(case=case,ticks=len(result),exact_old_geometry=True)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--case',choices=CASES)
    a=p.parse_args();freeze() if a.action=='freeze' else run(a.case)
