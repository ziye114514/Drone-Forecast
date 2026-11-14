"""Exact20ms SDK frames, FRED3-5 train/6 val, plus original train1/4.

Evaluation sequences0-2 and current dense trajectory labels are excluded.
This is detection domain adaptation, not a new physical forecasting model.
"""
from pathlib import Path
import collections,hashlib,json,os,shutil,sys,zipfile
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
OUT=WORK/'output/fred_domain_adaptation_20261029'
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import cv2,numpy as np
from prepare_fred_transfer import parse,sha
from live_decode import native_batches


def main():
    resume='--resume' in sys.argv
    if OUT.exists() and not resume:raise FileExistsError(OUT)
    if shutil.disk_usage(WORK).free<3*1024**3:raise RuntimeError('Need 3GiB free')
    OUT.mkdir(exist_ok=resume);parent=OUT/'parent';parent.mkdir(exist_ok=resume)
    ledger=[]
    with zipfile.ZipFile(WORK/'archive/output_history_20261028.zip') as z:
        prefix='stage40_record_split_finetune_20261018/'
        weight=parent/'best.pt';weight.write_bytes(z.read(prefix+'training/weights/best.pt'))
        assert sha(weight)=='ed11fada477ca14ebd4ed0e34a6e73f52cd6112222dbe2941335aa917ee49988'
        for name in z.namelist():
            if not name.startswith(prefix+'dataset/train/images/') or not Path(name).name.startswith(('train1_','train4_')) or not name.endswith('.png'):continue
            target=OUT/'dataset/train/images'/Path(name).name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(name));label=name.replace('/images/','/labels/').replace('.png','.txt')
            p=OUT/'dataset/train/labels'/Path(label).name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(label))
            ledger.append(dict(split='train',domain='original_self_weak_labels',image=target.name,
                image_sha256=sha(target),label_sha256=sha(p)))
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
    from metavision_sdk_core import BaseFrameGenerationAlgorithm
    cv2.setNumThreads(2)
    for seq in (3,4,5,6):
        split='val' if seq==6 else 'train';dest=OUT/'input'/str(seq);dest.mkdir(parents=True,exist_ok=resume)
        with zipfile.ZipFile(Path(f'E:/FRED/{seq}.zip')) as z:
            names=z.namelist();ann=next(n for n in names if Path(n).name=='coordinates.txt')
            raw=next(n for n in names if n.endswith('/Event/events.raw') or n=='Event/events.raw')
            content=z.read(ann);(dest/'coordinates.txt').write_bytes(content)
            if not (dest/'events.raw').exists():
                with z.open(raw) as src,(dest/'events.raw').open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
            elif (dest/'events.raw').stat().st_size!=z.getinfo(raw).file_size:
                raise ValueError('Incomplete RAW extraction')
        rows=parse(content.decode('utf-8-sig'));counts=collections.Counter(round(r['t']*1e6) for r in rows)
        selected={};slots=set()
        for r in rows:
            b=np.array(r['bbox']);stamp=round(r['t']*1e6);slot=int(r['t']*2)
            if counts[stamp]!=1 or slot in slots or min(b[2:]-b[:2])<12 or np.any(b[:2]<0) or np.any(b[2:]>[1280,720]):continue
            if abs(stamp-round(stamp/33333)*33333)>1:continue
            selected[stamp]=r;slots.add(slot)
        images=OUT/f'dataset/{split}/images';labels=OUT/f'dataset/{split}/labels'
        images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        existing={stamp for stamp in selected if (images/f'fred{seq}_{stamp:09d}.png').is_file() and
                  (labels/f'fred{seq}_{stamp:09d}.txt').is_file()}
        noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer()
        image=np.empty((720,1280,3),np.uint8);written=0
        maximum=max(selected,default=0)
        if not maximum:raise ValueError(f'No aligned labels: {seq}')
        iterator=[] if len(existing)==len(selected) else native_batches(dest/'events.raw',0.,(maximum+33333)/1e6,False,33333)
        for t,events in iterator:
            noise.process_events(events,buffer);stamp=round(t*1e6)
            if stamp not in selected:continue
            r=selected[stamp];clean=buffer.numpy();last=clean[np.searchsorted(clean['t'],stamp-20000):]
            BaseFrameGenerationAlgorithm.generate_frame(last,image)
            name=f'fred{seq}_{stamp:09d}';p=images/(name+'.png');cv2.imwrite(str(p),image)
            b=np.array(r['bbox']);values=np.r_[(b[:2]+b[2:])/2,b[2:]-b[:2]]/[1280,720,1280,720]
            label=labels/(name+'.txt');label.write_text('0 '+' '.join(f'{v:.9f}' for v in values)+'\n','utf-8')
            written+=1
        if not iterator:written=len(existing)
        for stamp,r in selected.items():
            name=f'fred{seq}_{stamp:09d}';p=images/(name+'.png');label=labels/(name+'.txt')
            if p.is_file() and label.is_file():
                ledger.append(dict(split=split,domain=f'FRED{seq}',image=p.name,t=stamp/1e6,
                    image_sha256=sha(p),label_sha256=sha(label)))
        print(json.dumps(dict(sequence=seq,split=split,selected=len(selected),written=written)),flush=True)
        if written<.95*len(selected):raise ValueError('Missing native frames')
    (OUT/'dataset.yaml').write_text('path: '+(OUT/'dataset').as_posix()+'\ntrain: train/images\nval: val/images\nnames:\n  0: drone\n','utf-8')
    protocol=dict(parent_weight=str(weight),parent_weight_sha256=sha(weight),epochs=3,
        freeze=10,imgsz=640,rect=True,batch=8,workers=0,seed=291026,device=0,amp=False,
        optimizer='AdamW',lr0=.0001,lrf=.1,weight_decay=.0005,patience=1000,
        deterministic=True,mosaic=0.,mixup=0.,copy_paste=0.,hsv_h=0.,hsv_s=0.,hsv_v=0.,
        degrees=0.,translate=0.,scale=0.,fliplr=.5,flipud=0.,
        training_FRED_sequences=[3,4,5],validation_FRED_sequences=[6],
        original_retain_recordings=[1,4],excluded_FRED=[0,1,2],excluded_self_scoring_recordings=[2,6],
        original_labels_are_weak=True,selection='Single fixed3epoch run; FRED6 validation fitness only',
        source_window_us=20000,source_filter_us=10000,source_hz=30.,sampling_hz=2.,
        default_model_not_changed=True,official_unseen_test_claim=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf-8')
    (OUT/'preparation.json').write_text(json.dumps(dict(counts=dict(collections.Counter((r['split']+':'+r['domain']) for r in ledger)),rows=ledger),indent=2),'utf-8')
    print(json.dumps(dict(prepared=len(ledger),counts=dict(collections.Counter(r['split'] for r in ledger)))) ,flush=True)


if __name__=='__main__':main()
