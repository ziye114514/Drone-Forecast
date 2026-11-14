"""Render saved causal outputs on matching native ON/OFF RAW, without inference."""
from pathlib import Path
import sys,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from live_decode import native_batches


def render(run):
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
    from metavision_sdk_core import BaseFrameGenerationAlgorithm
    run=Path(run);p=json.loads((run/'protocol.json').read_text('utf-8'))
    rows=[json.loads(s) for s in (run/'predictions.jsonl').read_text('utf-8').splitlines()]
    noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer()
    image=np.empty((720,1280,3),np.uint8);writer=cv2.VideoWriter(str(run/'results.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,720))
    assert writer.isOpened();count=0;poster=False
    for row,(t,e) in zip(rows,native_batches(p['raw'],p['actual_iterator_start'],p['actual_duration'],False,p['native_step_us'])):
        assert abs(t-row['t'])<1e-8
        noise.process_events(e,buffer);clean=buffer.numpy();last=clean[int(np.searchsorted(clean['t'],round(t*1e6)-20000)):]
        BaseFrameGenerationAlgorithm.generate_frame(last,image)
        b=row['body'];s=row['structure']
        if b.get('detected'):
            box=np.rint(b['bbox']).astype(int);cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(60,200,60),1)
            if s.get('model_centers') is not None:
                hubs=np.asarray(s['model_centers']);center=s.get('current_geometric_center')
                center=np.asarray(center) if center is not None else hubs.mean(0)
                for i,h in enumerate(hubs):
                    color=(150,150,150) if i in s['inferred_model_ids'] else (210,80,230)
                    cv2.line(image,tuple(np.rint(center).astype(int)),tuple(np.rint(h).astype(int)),color,1)
                    cv2.circle(image,tuple(np.rint(h).astype(int)),5,color,1)
                cv2.circle(image,tuple(np.rint(center).astype(int)),5,(210,80,230),1)
            for h in s['observed_centers']:cv2.circle(image,tuple(np.rint(h).astype(int)),2,(0,0,255),-1)
            if s.get('axis_xy') is not None:
                z=np.asarray(b['center']);n=np.asarray(s['axis_xy']);n=-n if n[1]>0 else n
                cv2.arrowedLine(image,tuple(np.rint(z).astype(int)),tuple(np.rint(z+55*n).astype(int)),(0,200,255),2)
        if row['valid'] and row['physical']:
            c=row['coefficient'];pts=np.rint([c['position']]+[row['physical'][str(h)] for h in (.1,.2,.4)]).astype(np.int32)
            cv2.polylines(image,[pts],False,(0,200,255),2)
        text=f'{t:.3f}s | {s["mode"]} | force proxy={row["force"]["active"]} | source30Hz'
        cv2.putText(image,text,(15,28),0,.62,(0,200,255),1)
        cv2.putText(image,'Saved-output visualization; playback30fps is NOT measured processing FPS',(15,55),0,.5,(0,180,255),1)
        writer.write(image);count+=1
        if not poster and s['mode']=='four':cv2.imwrite(str(run/'poster.jpg'),image);poster=True
        if count==len(rows)//2 and not poster:cv2.imwrite(str(run/'poster.jpg'),image)
    writer.release();assert count==len(rows)
    cap=cv2.VideoCapture(str(run/'results.mp4'));decoded=0
    while cap.read()[0]:decoded+=1
    cap.release();assert decoded==count
    (run/'render_check.json').write_text(json.dumps(dict(frames=count,decoded=decoded,inference_rerun=False,
        benchmark_includes_video=False,native_timestamps_matched=True)),'utf-8')
    print('rendered',str(run),count)


if __name__=='__main__':render(sys.argv[1])
