"""OpenCV phase correlation on native count maps: camera-motion confound audit.

Not a calibrated camera pose or independent flow truth; nothing is subtracted
from predictions. Only supplied past target ROI is excluded in this diagnostic.
"""
from pathlib import Path
import argparse,datetime,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage88')]
from inventory88 import save,load,sha,parse,unique_rows
from event_drone_forecast.raw_io import raw_chunks
OUT=ROOT.parent/'output/stage103_background_motion_audit_20261022'
DATA=ROOT.parent/'output/stage94_official_large_target_20261022'


def freeze():
    assert not OUT.exists()
    rng=np.random.default_rng(10300);im=cv2.GaussianBlur(rng.random((180,320)).astype(np.float32),(5,5),.8)
    shifted=cv2.warpAffine(im,np.float32([[1,0,4],[0,1,-3]]),(320,180),borderMode=cv2.BORDER_WRAP)
    shift,response=cv2.phaseCorrelate(im,shifted);assert np.linalg.norm(np.array(shift)-[4,-3])<.05
    save(OUT/'protocol103.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),
        source='https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html',
        windows=[dict(name='visible194',start=6.84,duration=3.34),dict(name='empty_midpoint194',start=57.,duration=10.)],
        step_us=20000,spatial_bin=4,mask_padding_native_px=32,minimum_response=.1,
        reuse='cv2.phaseCorrelate with Hann window, fourfold spatial event counts; no grayscale reconstruction/RGB.',
        wiring_control=dict(known_shift=[4,-3],measured=shift,response=response),
        interpretation='Background translation candidate only. Tree motion, parallax, event intermittency and mask edges confound it. No compensation in forecast or true camera pose claim.',
        default_promoted=False))
    print('103 native background-motion diagnostic frozen',flush=True)


def run():
    p=load(OUT/'protocol103.json');assert p['source_sha256']==sha(__file__)
    from metavision_core.event_io.h5_io import HDF5EventsReader
    d=load(DATA/'download94.json');assert sha(d['path'])==d['native_sha256']
    labels,bad=parse((DATA/'input/coordinates.txt').read_text(encoding='utf-8-sig'));assert not bad;labels=unique_rows(labels);results=[]
    for window in p['windows']:
        reader=HDF5EventsReader(d['path']);height,width=reader.get_size();hh,ww=height//4,width//4;hann=cv2.createHanningWindow((ww,hh),cv2.CV_32F)
        last_im=None;last_roi=None;index=0;latest=None;records=[]
        for t,e in raw_chunks(reader,window['start'],window['duration'],20000):
            while index<len(labels) and labels[index]['t']<=t+1e-9:latest=labels[index];index+=1
            b=None if latest is None or t-latest['t']>.05+1e-9 else latest['bbox']
            pix=(e['y'].astype(int)//4)*ww+e['x']//4
            counts=np.bincount(pix,minlength=hh*ww).reshape(hh,ww).astype(np.float32)
            im=np.log1p(counts)
            if last_im is not None:
                a=last_im.copy();q=im.copy()
                for roi in [last_roi,b]:
                    if roi is None:continue
                    lo=np.maximum(np.floor((np.array(roi[:2])-32)/4).astype(int),0)
                    hi=np.minimum(np.ceil((np.array(roi[2:])+32)/4).astype(int),[ww,hh])
                    a[lo[1]:hi[1],lo[0]:hi[0]]=0;q[lo[1]:hi[1],lo[0]:hi[0]]=0
                shift,response=cv2.phaseCorrelate(a,q,hann)
                value=np.array(shift)*4
                accepted=bool(response>=.1 and np.isfinite(value).all() and max(abs(value))<=32)
                records.append(dict(t=t,shift_xy_px=value.tolist(),response=float(response),accepted=accepted,
                    inferred_speed_px_s=float(np.linalg.norm(value)/.02),target_masked=b is not None))
            last_im=im;last_roi=b
        accepted=[r for r in records if r['accepted']]
        summary=dict(name=window['name'],pairs=len(records),accepted_pairs=len(accepted),
            median_accepted_step_px=float(np.median([np.linalg.norm(r['shift_xy_px']) for r in accepted])) if accepted else None,
            median_accepted_speed_px_s=float(np.median([r['inferred_speed_px_s'] for r in accepted])) if accepted else None,
            median_response=float(np.median([r['response'] for r in records])),
            not_independent_flow_truth=True,compensation_enabled=False)
        save(OUT/(window['name']+'.json'),dict(summary=summary,records=records));results.append(summary);print(summary,flush=True)
    save(OUT/'checks103.json',dict(records=results,source_hash_unchanged=sha(__file__)==p['source_sha256'],no_forecast_changes=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args();globals()[a.action]()
