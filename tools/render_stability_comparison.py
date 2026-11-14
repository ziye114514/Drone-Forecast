"""Retrospective side-by-side native ON/OFF, identical source cutoffs."""
from pathlib import Path
import sys,json
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_decode import native_batches
from sdk_activity_backend import SDKActivityAVX2
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from prepare_fred_transfer import parse
from score_fred_transfer import reference


def main():
    out=ROOT.parent/'output/stability_audit_20261031';a=out/'RAW_FRED9_control4s';b=out/'RAW_FRED9_budget4s'
    streams=[[json.loads(s) for s in (p/'predictions.jsonl').read_text('utf-8').splitlines()] for p in (a,b)]
    labels=parse((out/'holdout9/input/coordinates.txt').read_text('utf-8-sig'))
    protocol=json.loads((a/'protocol.json').read_text());noise=SDKActivityAVX2(1280,720,10000);buffer=noise.get_empty_output_buffer()
    video=cv2.VideoWriter(str(out/'FRED9_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30.,(1280,480))
    assert video.isOpened();poster=None;count=0
    for i,(t,e) in enumerate(native_batches(Path(protocol['raw']),protocol['actual_iterator_start'],protocol['actual_duration'],False,33333)):
        if i>=len(streams[0]):break
        assert abs(streams[0][i]['t']-t)<1e-7 and abs(streams[1][i]['t']-t)<1e-7
        noise.process_events(e,buffer);clean=buffer.numpy();clean=clean[clean['t']>=round(t*1e6)-20000]
        frame=np.empty((720,1280,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(clean,frame)
        current=reference(labels,t);future=reference(labels,t+.4)
        points=[]
        for box in (current,future):
            if box is not None:points.extend([box[:2],box[2:]])
        for rows in streams:
            row=rows[i]
            if row['valid']:points.append(row['physical']['0.4'])
        if not points:points=[[0,0],[1280,720]]
        p=np.asarray(points);center=.5*(p.min(0)+p.max(0));size=np.maximum([320.,240.],p.max(0)-p.min(0)+[100.,80.])
        size=np.minimum([1280.,720.],size);lo=np.clip(center-size/2,[0,0],[1280,720]-size).astype(int);hi=np.minimum([1280,720],lo+np.ceil(size).astype(int))
        panels=[]
        for title,rows in zip(('CONTROL','BUDGETED CURRENT-NN RECOVERY'),streams):
            row=rows[i];image=frame.copy();error=None
            if row['body'].get('detected'):
                box=np.rint(row['body']['bbox']).astype(int);cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(60,230,60),1)
            if future is not None:
                target=(future[:2]+future[2:])/2;cv2.drawMarker(image,tuple(np.rint(target).astype(int)),(255,255,255),cv2.MARKER_CROSS,14,2)
                if row['valid']:
                    pred=np.array(row['physical']['0.4']);error=float(np.linalg.norm(pred-target))
                    cv2.circle(image,tuple(np.rint(pred).astype(int)),5,(0,200,255),2)
                    cv2.line(image,tuple(np.rint(pred).astype(int)),tuple(np.rint(target).astype(int)),(0,80,255),1)
            crop=image[lo[1]:hi[1],lo[0]:hi[0]];scale=min(640/crop.shape[1],400/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            panel=np.full((480,640,3),32,np.uint8);panel[60:60+crop.shape[0],:crop.shape[1]]=crop
            cv2.putText(panel,title,(10,24),0,.5,(240,240,240),1)
            text=f't={t:.3f}s  0.4s error={error:.1f}px' if error is not None else f't={t:.3f}s  NO VALID FORECAST'
            cv2.putText(panel,text,(10,48),0,.5,(0,210,255),1)
            cv2.putText(panel,'white: future label (OFFLINE ONLY); yellow: prediction',(10,472),0,.4,(210,210,210),1);panels.append(panel)
        combined=np.hstack(panels);video.write(combined);count+=1
        if i==len(streams[0])//2:poster=combined.copy()
    video.release();assert count==120,count
    check=cv2.VideoCapture(str(out/'FRED9_comparison.mp4'));decoded=0
    while check.read()[0]:decoded+=1
    check.release();assert decoded==count
    if poster is not None:cv2.imwrite(str(out/'FRED9_comparison.png'),poster)
    (out/'video_protocol.json').write_text(json.dumps(dict(frames=count,decoded_frames=decoded,playback_fps=30,
        actual_computation_FPS_not_inferred_from_video=True,source_horizon_s=.4,
        retrospective_labels_only=True,visual_crop_uses_labels_not_runtime_input=True),indent=2),'utf-8')
    print('saved 120-frame native ON/OFF comparison')


if __name__=='__main__':main()
