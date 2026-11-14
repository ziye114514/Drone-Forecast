"""Render actualRAW90 pilot predictions, no target overlays or fake pixels."""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage129')]
from study129 import OUT,rows,save
from event_drone_forecast.raw_io import open_raw,raw_chunks
def main():
    rr=rows(OUT/'raw/prefix_natural6/observations_forecasts.jsonl');reader=open_raw('E:/dataset/train/6/media/event.raw');h,w=reader.get_size();dest=OUT/'guarded_startup129.mp4';video=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),50,(w+440,h));assert video.isOpened();n=0
    for r,(t,e) in zip(rr,raw_chunks(reader,22.,1.2,20000)):
        assert abs(t-r['t'])<1e-8;pix=e['y'].astype(np.int32)*w+e['x'];on=np.bincount(pix[e['p']>0],minlength=w*h).reshape(h,w)>0;off=np.bincount(pix[e['p']<=0],minlength=w*h).reshape(h,w)>0
        im=np.full((h,w+440,3),248,np.uint8);source=im[:,:w];source[:]=255;source[on]=(100,100,240);source[off]=(220,170,20);source[on&off]=0
        q=r['frontends']['delayed_feedback'];worker=q['learned_structure118'];safe=worker['support129']
        for z,a in zip(safe['guarded_current_inputs']['occupancy'],safe['boundary_audit']['occupancy']):
            col=(0,0,240) if a['ellipse_outside_sensor'] else (180,0,160);cv2.ellipse(im,((z['cx'],z['cy']),(z['major'],z['minor']),z['angle']),col,2)
        for name,col in [('partial_affine3_kf120',(110,110,110)),('support_FOV_guard_kf129',(0,165,35))]:
            c=q['coefficients'].get(name)
            if q['valid'] and c:
                points=np.rint([np.array(c['position'])+dt*np.array(c['velocity']) for dt in [0,.1,.2,.3,.4]]).astype(np.int32);cv2.polylines(im,[points],False,col,2)
        # Source overlays strictly clipped to sensor; never draw an inferred
        # ellipse across the panel as if it were measured outside the camera.
        im[:,w:]=248
        lines=[f'RAW t={t:.2f}s','Red: fitted ellipse exceeds sensor','Purple: ellipse inside sensor','Gray: original count-model0.4s','Green: FOV-guarded0.4s','',f'Before guard current: {worker["support128"]["output"]["center"]["current"]}',f'Guard current: {safe["center"]["current"]}',f'Chosen source: {safe["chosen"]}','No outside-FOV full-disk claim','50fps offline replay, NOT realtime']
        for j,line in enumerate(lines):cv2.putText(im,line,(w+10,25+j*25),0,.46,(25,40,50),1)
        b=r['body_observation'].get('bbox')
        if b:
            b=np.asarray(b);lo=np.maximum(np.floor(b[:2]-15).astype(int),0);hi=np.minimum(np.ceil(b[2:]+15).astype(int),[w,h]);crop=source[lo[1]:hi[1],lo[0]:hi[0]]
            if crop.size:
                scale=min(415/crop.shape[1],300/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST);im[370:370+crop.shape[0],w+10:w+10+crop.shape[1]]=crop
        video.write(im);n+=1
        if abs(t-22.68)<1e-8:cv2.imwrite(str(OUT/'guarded_startup129.png'),im)
    video.release();assert n==60;cap=cv2.VideoCapture(str(dest));decoded=0
    while True:
        ok,_=cap.read()
        if not ok:break
        decoded+=1
    cap.release();assert decoded==60;save(OUT/'video129_checks.json',dict(frames=60,fully_decoded=True,actual_RAW_render=True,actual_current_forecasts=True,future_targets_read=False,scope='Post-inference rendering of60 positive pilot, no full8s new RAW forecast.'))
if __name__=='__main__':main()
