"""Offline visualization of already-published forecasts against actual future RAW.

Future events are used for display ONLY. No estimator is instantiated here.
Clip selection uses detection visibility, never future prediction errors.
"""
from pathlib import Path
from collections import deque
import argparse,json,sys
import cv2,numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.motion import extrapolate
from event_drone_forecast.cli import draw_rotor

def select_clip(observations,duration):
    dt=.02;n=int(duration/dt)
    score=np.array([float(o.get('detected',False))+.5*min(len(o.get('rotors',[])),4)/4 for o in observations])
    if len(score)<n:return observations[0]['t']-dt
    sums=np.convolve(score,np.ones(n),'valid')
    # Exclude first second so motion states have a chance to initialize.
    sums[:min(50,len(sums))]=-1
    k=int(np.argmax(sums));return max(0,round(observations[k]['t']-dt,2))

def event_image(events,w,h):
    pixel=events['y'].astype(int)*w+events['x']
    a=np.bincount(pixel,minlength=w*h).reshape(h,w)
    return cv2.cvtColor(np.uint8(np.clip(np.log1p(a)/4,0,1)*255),cv2.COLOR_GRAY2BGR)

def crop_view(image,cx,cy,size=800):
    left=int(np.clip(cx-size/2,0,max(0,image.shape[1]-size)))
    top=int(np.clip(cy-size*.625/2,0,max(0,image.shape[0]-size*.625)))
    return image[top:top+int(size*.625),left:left+size].copy(),np.array([left,top])

def render(run,output,duration=8.,start=None,baseline=None):
    run=Path(run);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((run/'run_manifest.json').read_text())
    observations=[json.loads(s) for s in (run/'observations.jsonl').read_text().splitlines()]
    traces=[json.loads(s) for s in (run/'forecasts.jsonl').read_text().splitlines()]
    times=np.array([o['t'] for o in observations])
    baseline_traces=[json.loads(s) for s in (Path(baseline)/'forecasts.jsonl').read_text().splitlines()] if baseline else None
    baseline_times=np.array([q['t'] for q in baseline_traces]) if baseline else None
    explicit_start=start is not None
    if start is None:start=select_clip(observations,duration)
    start=min(start,max(0,times[-1]-duration-.42));start=round(start/.02)*.02
    reader=open_raw(manifest['input_raw']);h,w=reader.get_size()
    queue=deque();frames=[];frame_count=0;video=None;snapshots=[]
    for t,events in raw_chunks(reader,start,duration+.4,20000):
        index=int(np.searchsorted(times,t+.000001,side='right')-1)
        if index<0:continue
        image=event_image(events,w,h)
        queue.append((t,image,observations[index],traces[index]))
        if len(queue)<21:continue
        old_t,old_image,old_obs,result=queue.popleft()
        if frame_count%2:
            frame_count+=1;continue
        frame_count+=1
        left=old_image.copy();right=image.copy()
        for q in old_obs.get('rotors',[]):
            draw_rotor(left,q)
        if result.get('active'):
            center=np.array(result['states']['cv']['position'])
            plotted=[('cv',result['states'].get('cv'),(235,190,50)),('adaptive',result['states'].get('adaptive'),(40,170,255))]
            if baseline_traces:
                k=int(np.searchsorted(baseline_times,old_t+.000001,side='right')-1)
                old=baseline_traces[k] if k>=0 else {}
                plotted=[('v0.1',old.get('states',{}).get('adaptive') if old.get('active') else None,(235,190,50)),
                         ('v0.2',result['states'].get('adaptive'),(40,170,255))]
            for name,state,color in plotted:
                if state is None:continue
                line=np.array([extrapolate(state,x) for x in np.linspace(0,t-old_t,21)],np.int32)
                cv2.polylines(left,[line],False,color,3)
                endpoint=tuple(line[-1]);cv2.drawMarker(right,endpoint,color,cv2.MARKER_CROSS,25,3)
                cv2.circle(left,endpoint,5,color,-1)
        else:center=np.array([w/2,h/2])
        left,origin=crop_view(left,*center)
        right=right[int(origin[1]):int(origin[1])+left.shape[0],int(origin[0]):int(origin[0])+left.shape[1]].copy()
        left=cv2.resize(left,(800,500));right=cv2.resize(right,(800,500))
        canvas=np.zeros((600,1600,3),np.uint8);canvas[80:580,:800]=left;canvas[80:580,800:]=right
        cv2.putText(canvas,f'PUBLISHED at t={old_t:.2f}s',(18,30),0,.8,(245,245,245),2)
        cv2.putText(canvas,f'ACTUAL RAW at t={t:.2f}s (+{t-old_t:.2f}s)',(818,30),0,.75,(245,245,245),2)
        legend='Green: ellipse | Pink: arc | Yellow: axis | Blue: v0.1 | Orange: v0.2' if baseline else 'Green: ellipse | Pink: arc | Yellow: axis | Blue: CV | Orange: adaptive'
        cv2.putText(canvas,legend,(18,63),0,.53,(220,220,220),1)
        cv2.putText(canvas,'Crosses are old predictions; this panel never enters prediction',(818,63),0,.58,(220,220,220),1)
        learning='UPDATE' if result.get('rotor_learning_allowed') else 'FROZEN / HISTORY'
        cv2.putText(canvas,f"Geometry quality={old_obs.get('geometry_quality',0):.2f} | Rotor learning: {learning} | Axis has no resolved thrust sign",
                    (18,595),0,.48,(230,230,230),1)
        if not result.get('active'):cv2.putText(canvas,'NO FORECAST / TARGET LOST',(30,120),0,.8,(80,80,255),2)
        if video is None:
            video=cv2.VideoWriter(str(out/'current_vs_future.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(1600,600))
            if not video.isOpened():raise RuntimeError('Video writer failed')
        video.write(canvas)
        if frame_count%6==1:
            frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(canvas,(1200,450)),cv2.COLOR_BGR2RGB)))
        if len(snapshots)<4 and frame_count>len(snapshots)*100+20:
            p=out/f'comparison_{len(snapshots)+1}.png';cv2.imwrite(str(p),canvas);snapshots.append(str(p))
    if video:video.release()
    if frames:frames[0].save(out/'current_vs_future.gif',save_all=True,append_images=frames[1:],duration=120,loop=0)
    info=dict(raw=manifest['input_raw'],source_forecasts=str(run/'forecasts.jsonl'),start_s=start,duration_s=duration,
              forecast_horizon_s=.4,selection=('Explicit preidentified clear-rotor interval, not error-based selection.' if explicit_start else 'Maximum rotor/detection visibility in a fixed-length window; no error-based selection.'),
              comparison='Left uses published states only. Right contains actual future RAW for offline visualization only.',
              inference_rerun=False,video_frames=frame_count//2,gif_frames=len(frames),snapshots=snapshots)
    info['baseline_published_forecasts']=str(Path(baseline)/'forecasts.jsonl') if baseline else None
    (out/'visualization_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--duration',type=float,default=8.);p.add_argument('--start',type=float)
    p.add_argument('--baseline',help='Optional v0.1 run; blue displays its already-published adaptive forecast')
    a=p.parse_args();render(a.run,a.output,a.duration,a.start,a.baseline)
