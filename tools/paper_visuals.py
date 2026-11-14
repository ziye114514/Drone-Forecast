"""Offline side-by-side: immutable forecasts and actual future RAW, never inference."""
from pathlib import Path
from collections import deque
import argparse,json,sys
import cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.motion import extrapolate

def records(path):
    result=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.endswith('\n'):result.append(json.loads(line))
    return result

def render(run,out,start,duration):
    out=Path(out);out.mkdir(exist_ok=True,parents=True);run=Path(run)
    meta=json.loads((run/'run_manifest.json').read_text());obs=records(run/'observations.jsonl');traces=records(run/'forecasts.jsonl')
    count=min(len(obs),len(traces));obs=obs[:count];traces=traces[:count];times=np.array([q['t'] for q in obs])
    reader=open_raw(meta['input_raw']);height,width=reader.get_size();pending=deque();video=None;gifs=[];frames=0
    for t,events in raw_chunks(reader,start,duration+.4,10000):
        k=np.searchsorted(times,t+1e-7,side='right')-1
        if k<0 or k>=count:continue
        pixels=events['y'].astype(int)*width+events['x'];density=np.bincount(pixels,minlength=width*height).reshape(height,width)
        im=cv2.cvtColor(np.uint8(np.clip(np.log1p(density)/4,0,1)*255),cv2.COLOR_GRAY2BGR)
        pending.append((t,im,obs[k],traces[k]))
        if len(pending)<41:continue
        old_t,left,o,q=pending.popleft();frames+=1
        if frames%5:continue
        right=im.copy()
        for r in o.get('rotors',[]):
            color=(50,220,70) if r['observed_now'] else (110,110,110)
            cv2.ellipse(left,((r['cx'],r['cy']),(r['major'],r['minor']),r['angle']),color,1)
            hz=r['blade_pass_hz'];text=f"#{r['rotor_id']} {hz:.0f}Hz" if hz else f"#{r['rotor_id']} ?"
            cv2.putText(left,text,(int(r['cx']-30),int(r['cy']+12)),0,.38,color,1)
        center=np.asarray(o.get('center',[width/2,height/2]))
        if q.get('active'):
            center=np.asarray(q['states']['cv']['position'])
            for name,color in [('cv',(235,190,50)),('paper_nominal',(210,90,220)),('paper_fused',(40,170,255))]:
                state=q['states'][name];curve=np.array([extrapolate(state,h) for h in np.linspace(0,.4,25)],np.int32)
                cv2.polylines(left,[curve],False,color,2);cv2.drawMarker(right,tuple(curve[-1]),color,cv2.MARKER_CROSS,24,2)
        x=int(np.clip(center[0]-400,0,width-800));y=int(np.clip(center[1]-250,0,height-500))
        canvas=np.zeros((720,1600,3),np.uint8);canvas[100:600,:800]=left[y:y+500,x:x+800];canvas[100:600,800:]=right[y:y+500,x:x+800]
        cv2.putText(canvas,f'PUBLISHED t={old_t:.2f}s',(16,32),0,.9,(245,245,245),2)
        cv2.putText(canvas,f'ACTUAL RAW t={t:.2f}s (+{t-old_t:.2f}s)',(815,32),0,.85,(245,245,245),2)
        cv2.putText(canvas,'Green: observed rotor | labels: provisional blade-pass Hz',(16,68),0,.58,(220,220,220),1)
        cv2.putText(canvas,'Crosses = forecasts made 0.4s earlier; future is display only',(815,68),0,.55,(220,220,220),1)
        cv2.putText(canvas,'Blue: CV   Pink: nominal thrust model   Orange: causal gated fusion',(16,631),0,.65,(230,230,230),1)
        status='ACTIVE' if q.get('physics_active') else 'UNAVAILABLE; fusion uses CV'
        cv2.putText(canvas,f"Four ellipses: {o.get('four_ellipses',False)} | Physics: {status}",(16,663),0,.65,(230,230,230),1)
        cv2.putText(canvas,'Image-space experiment; no independent flight RPM / attitude / metric position ground truth',(16,695),0,.6,(160,170,190),1)
        if not q.get('active'):cv2.putText(canvas,'NO FORECAST',(40,160),0,1,(70,70,255),2)
        if video is None:
            video=cv2.VideoWriter(str(out/'current_vs_future.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1600,720))
            if not video.isOpened():raise RuntimeError('Video codec unavailable')
        video.write(canvas)
        if frames%10==0:gifs.append(Image.fromarray(cv2.cvtColor(cv2.resize(canvas,(1200,540)),cv2.COLOR_BGR2RGB)))
        if frames in (100,300,500):cv2.imwrite(str(out/f'comparison_{frames//100}.png'),canvas)
    if video:video.release()
    if gifs:gifs[0].save(out/'current_vs_future.gif',save_all=True,append_images=gifs[1:],duration=100,loop=0)
    info=dict(source_forecasts=str(run/'forecasts.jsonl'),source_raw=meta['input_raw'],start_s=start,duration_s=duration,
        horizon_s=.4,inference_rerun=False,selection='Fixed preidentified interval, not chosen by prediction error.',
        video_fps=20,video_frames=frames//5,gif_frames=len(gifs),future_raw_used_for_display_only=True)
    (out/'visualization_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print(info,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,required=True);p.add_argument('--duration',type=float,required=True)
    a=p.parse_args();render(a.run,a.output,a.start,a.duration)
