"""Read-only suitability sampling of user-captured RAW. Not a detector benchmark."""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage8'))
from event_drone_forecast.raw_io import open_raw
from polarity import counts
DATA=Path('E:/dataset');OUT=ROOT.parent/'output/self_capture_audit_20261017'


def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def catalog():
    result=[]
    for split in ['train','experiment']:
        for folder in sorted((DATA/split).iterdir(),key=lambda p:int(p.name) if p.name.isdigit() else 999):
            if not folder.is_dir() or not folder.name.isdigit():continue
            mp=next((p for p in [folder/'portable_manifest.json',folder/'input_manifest.json'] if p.exists()),None)
            meta=load(mp) if mp else {}
            rel=meta.get('synchronized_input',{}).get('raw')
            paths=[folder/rel] if rel else []
            paths+=[folder/'media/event.raw',folder/'event.raw']
            raw=next((p for p in paths if p.exists()),None)
            ep=next((p for p in [folder/'provenance/event_record.json',folder/'event_record.json'] if p.exists()),None)
            event=load(ep) if ep else {}
            duration=event.get('duration_host_sec')
            if duration is None and meta.get('avi_frames') and meta.get('avi_fps'):
                duration=meta['avi_frames']/meta['avi_fps']
            ap=folder/'annotations/phase1_manual_boxes.json';anns=load(ap) if ap.exists() else {}
            states=collections.Counter(r.get('event',{}).get('state','unknown') for r in anns.get('records',{}).values())
            gt=folder/'annotations/recording_truth.json'
            record=dict(id=split+folder.name,split=split,folder=str(folder),raw=None if raw is None else str(raw),
                manifest=None if mp is None else str(mp),original_name=meta.get('original_name'),
                raw_bytes=None if raw is None else raw.stat().st_size,
                duration_host_or_avi_estimate_s=duration,condition=meta.get('condition'),
                physical_sync_verified=meta.get('physical_sync_verified'),
                archive_annotations=states,annotation_count=sum(states.values()),
                negative_statement=load(gt) if gt.exists() else None,
                declared_raw_sha256=meta.get('files',{}).get(rel or '',{}).get('sha256'),
                scope='Metadata only; experiment RAW not visually inspected in this audit.' if split=='experiment' else 'Training RAW eligible for fixed sparse visual sampling.')
            result.append(record)
    save(OUT/'catalog.json',dict(records=result,raw_full_sha_verified=False,
        limits='Host/AVI duration is only a sampling estimate, not verified RAW end. Annotated event boxes may be320x184 reconstructed coordinates, not native1280x720 rotor truth. No protected project contents used.'))
    print(json.dumps([dict(id=r['id'],bytes=r['raw_bytes'],duration=r['duration_host_or_avi_estimate_s'],states=r['archive_annotations']) for r in result],ensure_ascii=False),flush=True)


def white(on,off):
    a=np.log1p(on);b=np.log1p(off);a/=max(float(a.max()),1e-6);b/=max(float(b.max()),1e-6)
    return np.uint8(np.stack([255*(1-a),255*(1-a),255*(1-b)],-1))


def sample():
    assert not (OUT/'sampling_complete.json').exists()
    rows=load(OUT/'catalog.json')['records'];summaries=[];began=time.perf_counter()
    plan=[]
    for r in rows:
        if r['split']!='train' or r['raw'] is None:continue
        duration=r['duration_host_or_avi_estimate_s'];assert duration is not None,r['id']
        times=sorted(set(round(duration*f,2) for f in [.1,.25,.5,.75,.9]))
        if r['id']=='train6':times=sorted(set(times+[23.,110.]))
        plan.append(dict(id=r['id'],raw=r['raw'],times_s=times,window_us=10000))
    save(OUT/'sampling_plan.json',dict(created_before_decode=True,plan=plan,
        method='Fixed duration fractions for all6training RAW, plus known23s/110s train6 anchors. No detector, G or APS frame used. Only visual suitability sampling, no performance scoring.'))
    for item in plan:
        key=item['id'];folder=OUT/key;folder.mkdir(parents=True,exist_ok=True)
        raw=Path(item['raw']);before=raw.stat();reader=open_raw(raw);h,w=reader.get_size();images=[];frames=[]
        for t in item['times_s']:
            lo=round(t*1e6);hi=lo+10000;reader.seek_time(lo);e=reader.load_delta_t(10000)
            e=e[(e['t']>=lo)&(e['t']<hi)]
            assert not len(e) or (e['x'].max()<w and e['y'].max()<h)
            q=counts(e,w,h);im=white(q['on'],q['off']);p=folder/f'raw_onoff_{t:.2f}.png'
            p.write_bytes(cv2.imencode('.png',im)[1].tobytes())
            np.savez_compressed(folder/f'counts_{t:.2f}.npz',on=q['on'],off=q['off'])
            row=dict(id=key,begin_s=t,end_s=t+.01,events=len(e),on_events=int((e['p']>0).sum()),
                active_pixels=int((q['total']>0).sum()),bipolar_pixels=int((np.minimum(q['on'],q['off'])>=2).sum()),
                image=p.relative_to(OUT).as_posix(),image_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            frames.append(row);thumb=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)
            tile=np.full((398,640,3),255,np.uint8);tile[38:]=thumb;tile[:38]=25
            cv2.putText(tile,f'{key} t={t:.2f}-{t+.01:.2f}s | {len(e)} events',(8,25),0,.57,(240,240,240),1);images.append(tile)
        reader.__del__();after=raw.stat()
        assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
        while len(images)%2:images.append(np.full_like(images[0],255))
        sheet=np.vstack([np.hstack(images[i:i+2]) for i in range(0,len(images),2)])
        (folder/'contact.png').write_bytes(cv2.imencode('.png',sheet)[1].tobytes())
        record=dict(id=key,raw=str(raw),width=w,height=h,raw_size_mtime_unchanged=True,frames=frames,
            contact=(folder/'contact.png').relative_to(OUT).as_posix())
        save(folder/'sampling.json',record);summaries.append(record)
        print(json.dumps(dict(completed=key,frames=len(frames),resolution=[w,h])),flush=True)
    save(OUT/'sampling_complete.json',dict(recordings=summaries,wall_s=time.perf_counter()-began,
        G_used=False,labels_in_sampling=False,detector_used=False,raw_end_and_full_file_integrity_not_verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['catalog','sample']);a=p.parse_args();globals()[a.action]()
