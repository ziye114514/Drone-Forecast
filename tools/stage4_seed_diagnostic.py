"""Explicit human-initialized diagnostic; NEVER a RAW-only deployment score."""
from pathlib import Path
import hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/stage4'))
sys.path.insert(0,str(ROOT/'src'))
from study import OUT,RAW,save,verify
from tracker import Skeleton,fit_affine,X
from event_drone_forecast.raw_io import open_raw,raw_chunks
from event_drone_forecast.motion import KinematicFilter
from event_drone_forecast.config import Config

def main():
    verify();folder=OUT/'seed_diagnostic';folder.mkdir(exist_ok=True)
    if (folder/'protocol.json').exists():raise RuntimeError('Diagnostic already frozen')
    # Once-only approximate motor shaft bases read from the t=110.00 event
    # rendering, not ellipse centers, future frames or projected hidden points.
    pts=np.array([[315.,303.],[648.,289.],[691.,351.],[401.,355.]])
    save(folder/'protocol.json',dict(seed_t=110.,points_native_px=pts.tolist(),
        manually_initialized=True,approx_annotation_uncertainty_px=8.,
        seed_source='side/view_110.0.png; shaft base visual estimates, no independent pose truth',
        scope='Can once initialized geometry survive using subsequent existing frontend outputs?',
        raw_only_deployment_result=False,thresholds_unchanged=True,until_s=114.,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows=[json.loads(l) for l in (OUT/'side/trace.jsonl').open()]
    by_t={round(r['t'],6):r for r in rows}
    trackers={n:Skeleton(body=b,partial=p) for n,b,p in [('four_only',False,False),('skeleton_partial',False,True),('skeleton_body',True,True)]}
    filters={n:KinematicFilter(Config(),2) for n in trackers};seen={n:None for n in trackers}
    stats={n:{} for n in trackers};first_loss={n:None for n in trackers}
    reader=open_raw(RAW);h,w=reader.get_size();acc=np.zeros((h,w),np.float32);output=[];count=0
    with (folder/'trace.jsonl').open('w',encoding='utf-8') as f:
        for i,(t,e) in enumerate(raw_chunks(reader,108.,6.4,10000)):
            flat=e['y'].astype(int)*w+e['x'];total=np.bincount(flat,minlength=h*w).reshape(h,w)
            acc*=np.exp(-.01/.02);acc+=total
            if (i+1)%2 or t<110.-1e-8 or t>114.+1e-8:continue
            image=np.uint8(np.clip(np.log1p(acc)/4,0,1)*255);image=cv2.GaussianBlur(image,(3,3),.6)
            r=by_t[round(t,6)];o=r['observation'];models={};states={}
            for n,tr in trackers.items():
                if abs(t-110.)<1e-8:
                    tr.A=fit_affine(np.arange(4),pts);tr.last_t=t
                    tr.last_support=tr.last_rotor=t;tr.init_count=1
                    if o.get('center') is not None:
                        tr.anchor=np.clip(np.linalg.pinv(tr.A[:,:2])@(np.array(o['center'])-tr.A[:,2]),-.6,.6)
                    if tr.body:tr.sample_body(image,o['rotors']);tr.previous_image=image.copy()
                    m=dict(t=t,active=True,mode='manual_seed',center=(tr.A[:,:2]@tr.anchor+tr.A[:,2]).tolist(),
                        points=(X@tr.A.T).tolist(),geometric_center=tr.A[:,2].tolist(),observed_slots=list(range(4)),
                        matched_count=4,body_inliers=0)
                else:m=tr.step(t,o['rotors'],image if tr.body else None,o.get('center'))
                models[n]=m;stats[n][m['mode']]=stats[n].get(m['mode'],0)+1
                if not m['active'] and first_loss[n] is None:first_loss[n]=t
                z=m['center'] if m['active'] and m['mode']!='coast' else None
                if z is not None:seen[n]=t
                x=filters[n].update(t,z,confidence=.75)
                active=x is not None and seen[n] is not None and t-seen[n]<=.3+1e-9
                states[n]=dict(active=bool(active),state=x.tolist() if x is not None else None)
            rr=dict(t=t,models=models,forecasts=states);output.append(rr);f.write(json.dumps(rr,allow_nan=False)+'\n')
            count+=1
    # Existing labels used only below, never above.
    labels=json.loads((ROOT.parent/'output/paper_reproduction_v030/evaluation_data/train6.json').read_text())['labels']
    times=np.array([r['t'] for r in output]);scores=[]
    for l in labels:
        if l['state']!='present' or not 110.42<=l['t']<=114:continue
        k=np.searchsorted(times,l['t']-.4+1e-8,side='right')-1
        if k<0:continue
        b=np.array(l['bbox']);target=(b[:2]+b[2:])/2;rr=output[k]
        errors={}
        for n,s in rr['forecasts'].items():
            if s['active']:
                x=np.array(s['state']);errors[n]=float(np.linalg.norm(x[:2]+(l['t']-rr['t'])*x[2:4]-target))
            else:errors[n]=None
        scores.append(dict(t=l['t'],anchor=rr['t'],errors_px=errors))
    save(folder/'summary.json',dict(frames=count,first_lost_t=first_loss,
         survival_after_seed_s={n:(v-110. if v is not None else 4.) for n,v in first_loss.items()},
         stats=stats,scores=scores,manually_initialized=True,not_deployment_performance=True))
    # Full frame current skeleton and true +0.4 future, fixed sampling.
    d=np.load(OUT/'side/event_frames.npz');imgs=d['images'];it=d['times']
    from stage4_report import draw_model
    writer=cv2.VideoWriter(str(folder/'manual_seed_future.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,840));assert writer.isOpened()
    for k in range(0,len(output),5):
        rr=output[k];j=int(np.argmin(abs(it-rr['t'])));fj=int(np.argmin(abs(it-rr['t']-.4)))
        if abs(it[fj]-rr['t']-.4)>.012:continue
        panels=[]
        for n in trackers:
            im=cv2.cvtColor(imgs[j],cv2.COLOR_GRAY2BGR);draw_model(im,rr['models'][n]);panel=np.zeros((420,640,3),np.uint8);panel[60:]=im
            cv2.putText(panel,f'MANUAL SEED: {n} t={rr["t"]:.2f}',(8,22),0,.48,(220,220,220),1)
            cv2.putText(panel,rr['models'][n]['mode']+'; dashed=predicted',(8,46),0,.48,(220,220,220),1);panels.append(panel)
        panel=np.zeros((420,640,3),np.uint8);im=cv2.cvtColor(imgs[fj],cv2.COLOR_GRAY2BGR)
        s=rr['forecasts']['skeleton_body']
        if s['active']:
            x=np.array(s['state']);cv2.drawMarker(im,tuple(np.int32((x[:2]+.4*x[2:])/2)),(235,100,235),cv2.MARKER_CROSS,18,2)
        panel[60:]=im;cv2.putText(panel,f'Actual future t={it[fj]:.2f}; +0.4s forecast',(8,22),0,.48,(220,220,220),1)
        panels.append(panel);im=np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]);writer.write(im)
        if k==50:
            ok,b=cv2.imencode('.png',im);assert ok;(folder/'preview.png').write_bytes(b.tobytes())
    writer.release();print((folder/'summary.json').read_text(),flush=True)

if __name__=='__main__':main()
