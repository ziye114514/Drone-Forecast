"""Verify actual standalone output, prefix causality, videos and batch-size drift."""
import hashlib,json
import cv2,numpy as np
from study42 import ROOT,OUT,save
def rows(p):return list(map(json.loads,p.read_text(encoding='utf-8').splitlines()))
def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for name,value in p['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,name
    full=rows(OUT/'train3/observations_forecasts.jsonl');prefix=rows(OUT/'prefix_train3/observations_forecasts.jsonl');assert len(full)==100 and len(prefix)==40
    for a,b in zip(full,prefix):
        for key in ['t','event_count','filtered_event_count','observations','forecasts','body_boxes']:assert a[key]==b[key],(a['t'],key)
    old={round(r['t'],8):r for r in rows(ROOT.parent/'output/stage41_continuous_body_forecast_20261018/train3/observations.jsonl')};det={round(r['t'],8):r for r in rows(ROOT.parent/'output/stage41_continuous_body_forecast_20261018/detections.jsonl') if r['case']=='train3' and r['method']=='finetuned'}
    differences=[];same_count=0
    for a in full:
        assert a['observations']['geometry']==old[round(a['t'],8)]['methods']['geometry']['observation']
        b=det[round(a['t'],8)];assert len(a['body_boxes'])==len(b['boxes']);same_count+=1
        for x,y in zip(a['body_boxes'],b['boxes']):differences.append(float(np.max(abs(np.array(x['bbox'])-y['bbox']))))
    videos=[]
    for c in ['train3','negative5']:
        rr=rows(OUT/c/'observations_forecasts.jsonl');cap=cv2.VideoCapture(str(OUT/c/'forecasts.mp4'));n=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(720,1280,3);n+=1
        cap.release();assert n==len(rr);videos.append(dict(case=c,decoded_frames=n))
    negative=rows(OUT/'negative5/observations_forecasts.jsonl');assert all(not o['detected'] for r in negative for o in r['observations'].values());assert all(not q['valid'] for r in negative for q in r['forecasts'].values())
    save(OUT/'regression.json',dict(real_raw_prefix_full_observations_all_forecasts_exact=len(prefix),geometry_vs41_exact=len(full),body_box_count_vs41_equal=same_count,batch1_vs4_max_bbox_difference_px=max(differences,default=0.),videos=videos,negative_all_current_and_forecast_zero=True,sources_unchanged=True))
    print(json.dumps(json.loads((OUT/'regression.json').read_text(encoding='utf-8'))),flush=True)
if __name__=='__main__':main()
