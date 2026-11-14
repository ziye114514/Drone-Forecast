"""Minimal actual contour probe: ill-conditioned ellipse vs explicit line."""
from pathlib import Path
import datetime,sys
import cv2,numpy as np
import study60 as current
sys.path.insert(0,str(current.ROOT/'research/stage8'))
from polarity import counts,PolarityFrontend
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.geometry import _line

def run():
    current.verify();out=current.OUT/'degenerate_contour_probe.json';assert not out.exists();reader=open_raw(Path(r'E:\dataset\train\3\aligned\event.raw'));h,w=reader.get_size();front=PolarityFrontend(w,h,'neighbor10')
    saved=[]
    for t,e in raw_chunks(reader,26.16,.01,10000):
        c=counts(e,w,h);f=front.feature(c['on'],c['off']);f[c['total']<front.cfg.min_pixel_count]=0;d=cv2.GaussianBlur(f,(3,3),.8);positive=d[d>0];threshold=max(2.,float(np.quantile(positive,front.cfg.density_quantile))) if len(positive) else 2.
        raw=np.uint8(d>=threshold)*255;opened=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((2,2),np.uint8));opened=cv2.morphologyEx(opened,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        for name,mask in [('morphology',opened),('thin_preopening',raw)]:
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:60]:
                p=contour.reshape(-1,2)
                if len(p)<6 or min(np.ptp(p,axis=0))>1:continue
                result=[]
                for shift in [0,1,2,3,4]:
                    for reverse in [False,True]:
                        q=np.roll(p[::-1] if reverse else p,shift,axis=0).astype(np.float32).reshape(-1,1,2)
                        for repeat in range(3):
                            fit=cv2.fitEllipse(q);result.append(dict(shift=shift,reverse=reverse,repeat=repeat,fit=fit))
                line=_line(p.astype(float),w*.5)
                if line is not None:line={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in line.items()}
                saved.append(dict(mask=name,points=p.tolist(),bbox=list(cv2.boundingRect(contour)),fits=result,line=line))
        current.save(out,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=current.sha(Path(__file__)),opencv_version=cv2.__version__,opencv_build=cv2.getBuildInformation(),t=t,raw_events=len(e),threshold=threshold,contours=saved,
           scope='Actual native contour, no future labels. Repeated fits and cyclic/reversed point order preserve geometric point set. Degenerate ellipse parameters are not physical rotor geometry. This alone does not identify all earlier process-context differences.'))
        print(dict(native_t=t,events=len(e),contours=len(saved),summary=[dict(bbox=r['bbox'],major_min=min(max(a['fit'][1]) for a in r['fits']),major_max=max(max(a['fit'][1]) for a in r['fits']),line_major=r['line']['major'] if r['line'] else None) for r in saved]),flush=True)

if __name__=='__main__':run()
