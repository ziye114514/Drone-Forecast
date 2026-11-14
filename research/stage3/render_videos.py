"""Actual future RAW, with a common crop chosen only from current observations."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from experiments import OUT,save
from analyze import traces
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.motion import extrapolate

def write_image(path,frame):
    ok,b=cv2.imencode('.png',frame);assert ok;Path(path).write_bytes(b.tobytes())

def render(case):
    folder=OUT/'raw_v2'/case;manifest=json.loads((folder/'run_manifest.json').read_text());observations=traces(folder/'observations.jsonl')
    times=np.array([r['t'] for r in observations]);arms={n:traces(folder/n/'forecasts.jsonl') for n in ('frozen','consensus','consensus_audit')}
    start,stop=(22.,28.) if case=='train6' else (108.,113.5)
    reader=open_raw(manifest['raw']);images=[];frame_times=[]
    for i,(t,e) in enumerate(raw_chunks(reader,start,stop-start+.45,10000)):
        if (i+1)%5:continue
        image=np.bincount(e['y'].astype(int)*1280+e['x'],minlength=1280*720).reshape(720,1280)
        images.append(np.uint8(np.clip(np.log1p(image)/4,0,1)*255));frame_times.append(t)
    writer=cv2.VideoWriter(str(folder/'actual_future_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1280,840));count=0
    for j,t in enumerate(frame_times):
        if t>stop or j+8>=len(images):continue
        k=np.searchsorted(times,t+1e-8,side='right')-1;o=observations[k]['observation']
        # Crop is fixed for all four panels at this anchor, based on present
        # tracking only. No future target location is used to center it.
        center=np.array(o['center']) if o['detected'] else np.array([640.,360.])
        x=int(np.clip(center[0]-400,0,480));y=int(np.clip(center[1]-225,0,270));panels=[]
        current=cv2.cvtColor(images[j],cv2.COLOR_GRAY2BGR)
        for r in o.get('rotors',[]):cv2.ellipse(current,((r['cx'],r['cy']),(r['major'],r['minor']),r['angle']),(80,210,90),1)
        panel=np.zeros((420,640,3),np.uint8);panel[60:]=cv2.resize(current[y:y+450,x:x+800],(640,360))
        cv2.putText(panel,f'Current RAW t={t:.2f}s; measured rotors={len(o["rotors"])}',(10,23),0,.55,(235,235,235),1)
        cv2.putText(panel,'All other panels: actual RAW at +0.4s',(10,47),0,.55,(235,235,235),1);panels.append(panel)
        for name,records in arms.items():
            q=records[k];future=cv2.cvtColor(images[j+8],cv2.COLOR_GRAY2BGR)
            for method,color in [('cv',(40,155,250)),('paper_fused',(215,100,240))]:
                if q['active']:
                    p=np.int32(np.round(extrapolate(q['states'][method],.4)));cv2.drawMarker(future,tuple(p),color,cv2.MARKER_CROSS,22,2)
            panel=np.zeros((420,640,3),np.uint8);panel[60:]=cv2.resize(future[y:y+450,x:x+800],(640,360))
            cv2.putText(panel,f'{name}; actual t={frame_times[j+8]:.2f}s',(10,23),0,.55,(235,235,235),1)
            cv2.putText(panel,f'physics={q["physics_active"]}; purple:fused orange:CV',(10,47),0,.48,(235,235,235),1)
            if not q['active']:cv2.putText(panel,'NO ACTIVE FORECAST',(10,85),0,.55,(150,150,150),1)
            panels.append(panel)
        joined=np.vstack((np.hstack(panels[:2]),np.hstack(panels[2:])));writer.write(joined)
        if count==39:write_image(folder/'comparison_frame39.png',joined)
        count+=1
    writer.release();save(folder/'video_manifest.json',dict(frames=count,fps=20,horizon_s=.4,actual_future_raw=True,
        current_crop_only=True,labels_in_video=False,anchor_start_s=start+.05,anchor_end_s=stop,
        prediction_inputs='Saved current/past-only forecasts; future RAW read only here for post-hoc display.'))
    print('Rendered '+case,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case',choices=['train6','side6']);a=parser.parse_args();render(a.case)
