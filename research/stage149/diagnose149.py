"""Native RAW display-window and codec comparison; cached boxes for crops only."""
from pathlib import Path
import sys,json
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage146'),str(ROOT/'research/stage129')]
from event_drone_forecast.raw_io import open_raw,raw_chunks
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from branches146 import ThresholdFrontend
from polarity import counts
from boundary129 import bounded
OUT=ROOT.parent/'output/stage149_inertia_scale_20261024/blur'
OUT.mkdir(parents=True,exist_ok=True)
OLD=ROOT.parent/'output/stage148_temporal_rotors_20261024'
settings=[('clear','E:/dataset/train/6/media/event.raw',26.84),
          ('side','E:/dataset/train/6/media/event.raw',112.02),
          ('side','E:/dataset/train/6/media/event.raw',114.02),
          ('side','E:/dataset/train/6/media/event.raw',118.12)]
all_rows={}
for name,_,_ in settings:
    if name not in all_rows:
        all_rows[name]=[json.loads(x) for x in (OLD/(name+'_joint')/'predictions.jsonl').read_text(encoding='utf8').splitlines()]
def image(events,w,h):
    result=np.zeros((h,w,3),np.uint8)
    BaseFrameGenerationAlgorithm.generate_frame(events,result)
    return result
def tile(source,roi,label):
    x1,y1,x2,y2=roi; crop=source[y1:y2,x1:x2]
    result=np.full((245,460,3),245,np.uint8)
    zoom=min(2.,440/max(1,crop.shape[1]),205/max(1,crop.shape[0]))
    resized=cv2.resize(crop,None,fx=zoom,fy=zoom,interpolation=cv2.INTER_NEAREST)
    result[35:35+resized.shape[0],10:10+resized.shape[1]]=resized
    cv2.putText(result,label,(10,22),0,.48,(15,30,40),1)
    return result
reports=[]
for name,raw,t in settings:
    row=min(all_rows[name],key=lambda x:abs(x['t']-t)); assert abs(row['t']-t)<1e-6
    reader=open_raw(raw); h,w=reader.get_size()
    noise=ActivityNoiseFilterAlgorithm(w,h,10000); buf=noise.get_empty_output_buffer()
    e=None; clean=None
    for stamp,chunk in raw_chunks(reader,round(t-.12,6),.12,20000):
        noise.process_events(chunk,buf); e=chunk;clean=buf.numpy().copy()
    assert abs(stamp-t)<1e-6
    box=row['body'].get('bbox') or [0,0,w,h]
    roi=np.r_[np.maximum(0,np.floor(np.array(box[:2])-18)),
              np.minimum([w,h],np.ceil(np.array(box[2:])+18))].astype(int).tolist()
    x1,y1,x2,y2=roi; inside=lambda a:(a['x']>=x1)&(a['x']<x2)&(a['y']>=y1)&(a['y']<y2)
    velocity=np.asarray((row['coefficients'].get('optimized_geometry') or {}).get('velocity',[0.,0.]))
    tag=name+'_'+str(t).replace('.','_')
    panels=[]; metrics=[]; native={}
    for ms in (20,10,5,2):
        short=clean[clean['t']>=round(t*1e6-ms*1000)]
        im=image(short,w,h);native[ms]=im
        cv2.imwrite(str(OUT/(tag+'_'+str(ms)+'ms.png')),im)
        panels.append(tile(im,roi,str(ms)+'ms filtered native'))
        local=short[inside(short)]
        points=np.c_[local['x'],local['y']]
        unique=len(np.unique(points,axis=0)) if len(points) else 0
        metrics.append(dict(window_ms=ms,events_in_roi=len(local),occupied_pixels=unique,
            translation_trail_estimate_px=float(np.linalg.norm(velocity)*ms/1000)))
    cv2.imwrite(str(OUT/(tag+'_windows.png')),np.hstack(panels))
    warped=e.copy()
    xx=np.rint(e['x'].astype(float)+(t-e['t']/1e6)*velocity[0]).astype(int)
    yy=np.rint(e['y'].astype(float)+(t-e['t']/1e6)*velocity[1]).astype(int)
    valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
    warped=warped[valid];warped['x']=xx[valid];warped['y']=yy[valid]
    corrected=image(warped,w,h)
    raw20=image(e,w,h)
    video=OUT/(tag+'_codec.mp4')
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),50,(w,h))
    assert writer.isOpened()
    for _ in range(4):writer.write(native[20])
    writer.release();cap=cv2.VideoCapture(str(video));ok,decoded=cap.read();cap.release();assert ok
    diff=native[20][y1:y2,x1:x2].astype(float)-decoded[y1:y2,x1:x2].astype(float)
    mse=float(np.mean(diff**2));psnr=float(10*np.log10(255**2/max(mse,1e-12)))
    cv2.imwrite(str(OUT/(tag+'_codec_comparison.png')),np.hstack([
        tile(native[20],roi,'20ms lossless PNG'),tile(decoded,roi,'Same20ms MP4 decoded'),
        tile(native[5],roi,'5ms lossless PNG'),tile(corrected,roi,'20ms raw translation-warped')]))
    # One-frame geometry probe is explicitly NOT temporal/end-to-end inference.
    probes=[]
    for ms in (20,10,5,2):
        short=e[e['t']>=round(t*1e6-ms*1000)]
        prepared=counts(short,w,h);prepared['on']*=20/ms;prepared['off']*=20/ms
        front=ThresholdFrontend(w,h,4.,False);front.set_hint(short,row['body'],t)
        observed,_=front.detect_prepared(short,t,prepared)
        rs,_=bounded(observed.get('rotors',[]),w,h)
        probes.append(dict(window_ms=ms,measured_candidates=len(rs),
            measured_shapes=[dict(cx=r['cx'],cy=r['cy'],major=r['major'],minor=r['minor'],mode=r['mode']) for r in rs],
            rate_normalization_to_20ms=True,prior_guide=False))
    reports.append(dict(name=name,t=t,roi=roi,windows=metrics,
        codec_roi_mae=float(np.mean(abs(diff))),codec_roi_psnr_db=psnr,
        geometry_single_frame_probe=probes,source_max_event_t=int(e['t'].max()),
        cached_box_and_velocity_used_only_for_diagnostic=True,
        recording_defocus_identifiable_from_this_test=False))
(OUT/'diagnostics149.json').write_text(json.dumps(reports,indent=2),encoding='utf8')

# Short native crop movie, same current crop, 20ms versus5ms. This is diagnostic
# rendering with old boxes, not a new detector run and not an inference input.
rows=all_rows['side']; indexed={round(r['t'],6):r for r in rows}
raw='E:/dataset/train/6/media/event.raw';reader=open_raw(raw);h,w=reader.get_size()
noise=ActivityNoiseFilterAlgorithm(w,h,10000);buf=noise.get_empty_output_buffer()
writer=cv2.VideoWriter(str(OUT/'side20_vs5ms.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),50,(920,245))
count=0
for t,e in raw_chunks(reader,117.4,1.2,20000):
    noise.process_events(e,buf)
    if t<=117.5+1e-8:continue
    clean=buf.numpy().copy();row=indexed.get(round(t,6))
    if row is None:continue
    box=row['body'].get('bbox')
    if box:
        roi=np.r_[np.maximum(0,np.floor(np.array(box[:2])-18)),np.minimum([w,h],np.ceil(np.array(box[2:])+18))].astype(int).tolist()
    else:roi=[0,0,w,h]
    short=clean[clean['t']>=round(t*1e6-5000)]
    writer.write(np.hstack([tile(image(clean,w,h),roi,'20ms t='+str(round(t,2))),
                           tile(image(short,w,h),roi,'5ms same time; no interpolation')]))
    count+=1
writer.release()
print(json.dumps(dict(diagnostics=reports,diagnostic_movie_frames=count),ensure_ascii=False),flush=True)
