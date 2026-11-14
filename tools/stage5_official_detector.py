"""Unmodified official EVPeriscope Python detector through a RAW IO bridge.

No flight-control scripts imported. Author camera/airframe calibration is not
used for this different sensor/airframe. A separate causal ROI input ablation
does not count as the original full-frame pipeline.
"""
from pathlib import Path
import collections,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'external/evperiscope/scripts'))
sys.path.insert(0,str(ROOT/'research/stage5'))
from detect import EvPropDet
from study import OUT,RAW,save
from event_drone_forecast.raw_io import open_raw,raw_chunks

CASES={'clear':(22.5,2.5),'side':(109.5,2.5),'absent':(0.,2.)}

def main():
    folder=OUT/'official_evperiscope';folder.mkdir(exist_ok=True)
    if (folder/'protocol.json').exists():raise RuntimeError('Already frozen')
    source=ROOT/'external/evperiscope/scripts/detect.py';sha=hashlib.sha256(source.read_bytes()).hexdigest()
    save(folder/'protocol.json',dict(upstream='https://github.com/grasp-lyrl/evperiscope',
        commit='a2e6f2208bf12ffbe8663d2ea01d3f32f93c2bbc',source_sha256=sha,source_modified=False,
        tau_s=.02,batch_s=.005,cases=CASES,seed=41,
        full_frame='Official EvPropDet.process_frame unchanged, distorted native sensor coordinates, no author camera calibration.',
        roi_ablation='Same official core, but event input restricted to latest current/past frozen original bbox (<.1s old), expanded by 30px. Separately reported.',
        pose_not_run='Missing true camera calibration and unknown asymmetric airframe. Do not import aircraft default dimensions as truth.',
        detector_stability='Author stable flag is not independent proof of four correct motors.',
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    summaries={}
    for case,(start,duration) in CASES.items():
        cv2.setNumThreads(1);cv2.setRNGSeed(41)
        reader=open_raw(RAW);h,w=reader.get_size();detectors={n:EvPropDet(h=h,w=w,tau=.02,n_centroids=4) for n in ['full','causal_roi']}
        obs=[]
        if case!='absent':obs=[json.loads(l)['observation'] for l in (OUT/case/'trace.jsonl').open()]
        times=np.array([v['t'] for v in obs]);counts={n:collections.Counter() for n in detectors};records=[];begin=time.perf_counter();last=begin
        dest=folder/case;dest.mkdir(exist_ok=True);saved=[];frame_times=[]
        with (dest/'trace.jsonl').open('w',encoding='utf-8') as f:
            for i,(t,e) in enumerate(raw_chunks(reader,start,duration,5000)):
                o=None
                if len(times):
                    k=np.searchsorted(times,t+1e-9,side='right')-1
                    if k>=0 and t-times[k]<.1 and obs[k]['detected']:o=obs[k]
                masked=e[:0]
                if o is not None:
                    b=np.array(o['bbox'])+[-30,-30,30,30]
                    selected=(e['x']>=b[0])&(e['x']<=b[2])&(e['y']>=b[1])&(e['y']<=b[3]);masked=e[selected]
                rr=dict(t=t,arms={})
                for name,det in detectors.items():
                    events=e if name=='full' else masked
                    r=det.process_frame(events['p'],events['t']/1e6,events['x'],events['y'],t,render_image=False)
                    pts=r['centroids'][:,::-1].tolist() if r['centroids'] is not None else None
                    rr['arms'][name]=dict(stable=bool(r['is_stable']),points_xy=pts,input_events=len(events),roi_available=o is not None)
                    counts[name]['frames']+=1;counts[name]['stable']+=int(r['is_stable']);counts[name]['four_returned']+=int(pts is not None)
                records.append(rr);f.write(json.dumps(rr,allow_nan=False)+'\n')
                if (i+1)%10==0:
                    total=np.bincount(e['y'].astype(int)*w+e['x'],minlength=w*h).reshape(h,w)
                    im=np.uint8(np.clip(np.log1p(total)/4,0,1)*255);saved.append(cv2.resize(im,(640,360)));frame_times.append(t)
                if time.perf_counter()-last>15:print('official',case,round(t,2),{n:dict(c) for n,c in counts.items()},flush=True);last=time.perf_counter()
        summaries[case]=dict(stats={n:dict(c) for n,c in counts.items()},wall_s=time.perf_counter()-begin)
        # Author points on actual current RAW, not assumed rotor identity truth.
        writer=cv2.VideoWriter(str(dest/'official_points.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1280,420));assert writer.isOpened()
        rt=np.array([r['t'] for r in records])
        for im,t in zip(saved,frame_times):
            r=records[int(np.argmin(abs(rt-t)))];panels=[]
            for n in detectors:
                panel=np.zeros((420,640,3),np.uint8);panel[60:]=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR);a=r['arms'][n]
                for j,p in enumerate(a['points_xy'] or []):
                    p=np.int32(np.array(p)/2)+[0,60];cv2.circle(panel,tuple(p),5,(60,230,90) if a['stable'] else (50,160,250),1)
                    cv2.putText(panel,str(j),tuple(p+[5,-5]),0,.4,(220,220,220),1)
                cv2.putText(panel,f'Official detector {n}; t={t:.2f}',(8,22),0,.5,(230,230,230),1)
                cv2.putText(panel,f'author stable={a["stable"]}; no pose or motor GT',(8,46),0,.48,(230,230,230),1);panels.append(panel)
            joined=np.hstack(panels);writer.write(joined)
            if abs(t-(24. if case=='clear' else 110.5 if case=='side' else 1.))<.01:
                ok,b=cv2.imencode('.png',joined);assert ok;(dest/'preview.png').write_bytes(b.tobytes())
        writer.release();print('official complete',case,summaries[case],flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
    save(folder/'summary.json',summaries)

if __name__=='__main__':main()
