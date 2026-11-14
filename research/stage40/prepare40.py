"""SDK311 creates lossless local event images and separate weak training labels."""
import math,sys
import cv2,numpy as np
from common40 import *
sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.raw_io import open_raw
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_core import BaseFrameGenerationAlgorithm

def main():
    p=verify();assert not (OUT/'inputs.json').exists();rows=[];truth=[];counts={}
    for split,recordings in {**SPLIT,'negative_test':[5]}.items():
        for n in recordings:
            if n==5:d=dict(raw='E:/dataset/train/5/media/event.raw',labels=[dict(t=float(i),state='absent') for i in range(1,8)])
            else:
                f=LABELS/f'train{n}.json';assert sha(f)==p['label_hashes'][f.name];d=load(f)
            raw=Path(d['raw']).resolve();assert raw.is_relative_to(Path('E:/dataset').resolve())
            reader=open_raw(raw);h,w=reader.get_size();assert (h,w)==(720,1280)
            for index,label in enumerate(sorted(d['labels'],key=lambda r:r['t'])):
                cutoff=int(math.floor(label['t']*1e6/PARAMS['window_us']))*PARAMS['window_us'];begin=max(0,cutoff-PARAMS['window_us']-PARAMS['warmup_us'])
                assert reader.current_time<=begin;reader.seek_time(begin);e=reader.load_delta_t(cutoff-begin);e=e[(e['t']>=begin)&(e['t']<cutoff)]
                alg=ActivityNoiseFilterAlgorithm(w,h,PARAMS['activity_threshold_us']);buf=alg.get_empty_output_buffer();alg.process_events(e,buf);filtered=buf.numpy().copy();filtered=filtered[(filtered['t']>=cutoff-PARAMS['window_us'])&(filtered['t']<cutoff)]
                frame=np.empty((h,w,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(filtered,frame)
                sample_id=f'train{n}_{index:04d}';relative=f'dataset/{split}/images/{sample_id}.png';path=OUT/relative;path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();cv2.imencode('.png',frame)[1].tofile(str(path))
                rows.append(dict(id=sample_id,recording=n,split=split,target_t=label['t'],cutoff_us=cutoff,image=relative,image_sha256=sha(path),width=w,height=h,event_count=len(filtered)))
                target=dict(id=sample_id,recording=n,split=split,target_t=label['t'],state=label['state'])
                text=''
                if label['state']=='present':
                    b=np.array(label['bbox'],float);assert 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h,(n,index,b)
                    target['bbox']=b.tolist();box=[(b[0]+b[2])/2/w,(b[1]+b[3])/2/h,(b[2]-b[0])/w,(b[3]-b[1])/h];text='0 '+' '.join(f'{x:.10f}' for x in box)+'\n'
                truth.append(target)
                if split in ['train','val']:
                    lp=OUT/f'dataset/{split}/labels/{sample_id}.txt';lp.parent.mkdir(parents=True,exist_ok=True);lp.write_text(text,encoding='utf-8')
            counts[n]=dict(images=sum(r['recording']==n for r in rows),present=sum(r['recording']==n and r['state']=='present' for r in truth));print('Prepared recording',n,counts[n],flush=True)
    save(OUT/'inputs.json',dict(images=rows,scope='No boxes/states; inference bridge only. Train/validation truth stored separately in YOLO labels.'))
    save(OUT/'targets.json',dict(targets=truth,scope='Score/preparation only. Training can read its own YOLO train/val labels, never test truth.'))
    (OUT/'dataset.yaml').write_text('path: '+(OUT/'dataset').as_posix()+'\ntrain: train/images\nval: val/images\nnames:\n  0: drone\n',encoding='utf-8')
    label_files=sorted((OUT/'dataset').glob('*/labels/*.txt'));save(OUT/'preparation.json',dict(counts=counts,images=len(rows),input_sha256=sha(OUT/'inputs.json'),targets_sha256=sha(OUT/'targets.json'),dataset_yaml_sha256=sha(OUT/'dataset.yaml'),label_hashes={x.relative_to(OUT).as_posix():sha(x) for x in label_files},all_cutoffs_before_target=True,no_gt_cropping=True,sources_unchanged=verify()['source_hashes']==p['source_hashes']))
    print('All prepared',len(rows),flush=True)
if __name__=='__main__':main()
