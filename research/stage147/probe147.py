"""RAW observation replay, frozen current body boxes; NOT end-to-end inference."""
from pathlib import Path
import json, sys, time
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent/'output/stage147_curvature_split_20261024'
OLD = ROOT.parent/'output/stage146_threshold_branches_20261024'
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'research/stage147'), str(ROOT/'research/stage129')]
from curvature147 import CurvatureFrontend, split_contour
from branches146 import ThresholdFrontend
from boundary129 import bounded
from polarity import counts
from event_drone_forecast.raw_io import open_raw, raw_chunks
from metavision_sdk_core import BaseFrameGenerationAlgorithm

def js(value): return json.dumps(value, indent=2, default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())

def controls():
    results = []
    for angle in (0, 20, 60):
        for gap in (18, 25, 32):
            mask = np.zeros((200, 250), np.uint8)
            centers = np.array([[110., 90.], [110., 90.+gap]])
            theta = np.deg2rad(angle); R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            centers = (centers-[110,100])@R.T+[110,100]
            for center in centers: cv2.ellipse(mask, tuple(np.rint(center).astype(int)), (40,20), angle,0,360,255,-1)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            arcs,diag=split_contour(max(contours,key=cv2.contourArea))
            error=None
            if arcs is not None:
                fitted=np.array([r['parameters'][:2] for r in arcs])
                error=float(min(np.linalg.norm(fitted-centers,axis=1).mean(),np.linalg.norm(fitted-centers[::-1],axis=1).mean()))
            results.append(dict(kind='two_overlap',angle=angle,gap=gap,accepted=arcs is not None,center_error_px=error,diag=diag))
    for kind in ('single','dent','line','contained'):
        mask=np.zeros((200,250),np.uint8)
        cv2.ellipse(mask,(110,100),(50,2 if kind=='line' else 25),0,0,360,255,-1)
        if kind=='dent': cv2.circle(mask,(110,75),6,0,-1)
        if kind=='contained': cv2.ellipse(mask,(110,100),(20,10),0,0,360,255,-1)
        c,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        arcs,diag=split_contour(max(c,key=cv2.contourArea))
        assert arcs is None, (kind,diag)
        results.append(dict(kind=kind,accepted=False))
    assert sum(r['accepted'] for r in results) >= 6, results
    assert all(r['center_error_px']<3 for r in results if r['accepted'])
    (OUT/'controls147.json').write_text(js(results),encoding='utf-8')
    print(js(dict(synthetic_recovered=sum(r['accepted'] for r in results),total_overlap=9,negatives=4)),flush=True)

def main():
    OUT.mkdir(exist_ok=True); controls(); summary={}
    cases=[('merge',26.66,'clear4_final'),('C',112.02,'side4_final'),
           ('line',118.12,'side4_final'),('body',114.02,'side4_final'),('overlap',116.52,'side4_final')]
    for name,t,cache in cases:
        rows={round(r['t'],6):r for r in map(json.loads,(OLD/cache/'predictions.jsonl').read_text().splitlines())}
        reader=open_raw('E:/dataset/train/6/media/event.raw'); h,w=reader.get_size()
        _,events=next(iter(raw_chunks(reader,round(t-.02,6),.02,20000)))
        row=rows[round(t,6)]; prepared=counts(events,w,h)
        front=CurvatureFrontend(w,h); front.debug_capture=True; front.set_hint(events,row['body'],t)
        tic=time.perf_counter();observation,_=front.detect_prepared(events,t,prepared);ms=(time.perf_counter()-tic)*1000
        diag=dict(t=t,ms=ms,rotors=observation.get('rotors',[]),splits=front.curvature_audit,roi=front.current_roi,contours=front.debug_contours)
        summary[name]=diag
        (OUT/'probe147.json').write_text(js(summary),encoding='utf-8')
        image=np.zeros((h,w,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(events,image)
        display=cv2.cvtColor(front.mask146,cv2.COLOR_GRAY2BGR)
        display=255-display
        for d in front.curvature_audit:
            for notch in d['notches']: cv2.circle(display,tuple(notch['point']),3,(0,0,255),-1)
        for r in diag['rotors']:
            cv2.ellipse(display,tuple(np.rint([r['cx'],r['cy']]).astype(int)),tuple(np.rint([r['major']/2,r['minor']/2]).astype(int)),r['angle'],0,360,(0,170,0),1)
            if r.get('original_arc_points'):
                for point in r['original_arc_points']:cv2.circle(display,tuple(np.rint(point).astype(int)),1,(255,100,0),-1)
        x1,y1,x2,y2=np.asarray(front.current_roi or [0,0,w,h],int);lo=np.maximum([x1-10,y1-10],0);hi=np.minimum([x2+10,y2+10],[w,h])
        panel=np.concatenate([image[lo[1]:hi[1],lo[0]:hi[0]],display[lo[1]:hi[1],lo[0]:hi[0]]],axis=1)
        scale=min(2.,1400/panel.shape[1]);panel=cv2.resize(panel,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT/f'probe_{name}.png'),panel)
        print(js(dict(case=name,count=len(diag['rotors']),splits=sum(d['accepted'] for d in front.curvature_audit),ms=ms)),flush=True)
    (OUT/'probe147.json').write_text(js(summary),encoding='utf-8')

if __name__=='__main__':main()
