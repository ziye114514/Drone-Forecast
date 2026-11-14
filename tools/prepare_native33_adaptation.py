"""Match the new inference window on existing FRED3-5 train /6 validation.

Reuses the already filtered training membership and unchanged box labels.
Scoring recordings0/7/8/9/10 never enter the new dataset.
"""
from pathlib import Path
import hashlib,json,shutil,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_decode import native_batches


def main(output):
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
    from metavision_sdk_core import BaseFrameGenerationAlgorithm
    out=Path(output)/'native33_adaptation';out.mkdir(exist_ok=False)
    old=ROOT.parent/'output/fred_domain_adaptation_20261029'
    train=[Path(x) for x in (old/'train_images.txt').read_text().splitlines() if x]
    val=list((old/'dataset/val/images').glob('fred6_*.png'))
    counts={};training=[]
    for seq in [3,4,5,6]:
        split='val' if seq==6 else 'train';items=[p for p in (val if seq==6 else train) if p.name.startswith(f'fred{seq}_')]
        selected={int(p.stem.rsplit('_',1)[1]):p for p in items}
        directory=out/f'dataset/{split}/images';labels=out/f'dataset/{split}/labels'
        directory.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buffer=noise.get_empty_output_buffer();image=np.empty((720,1280,3),np.uint8)
        saved=0
        iterator=native_batches(old/f'input/{seq}/events.raw',0.,(max(selected)+33333)/1e6,False,33333)
        for t,e in iterator:
            noise.process_events(e,buffer);stamp=round(t*1e6)
            if stamp not in selected:continue
            source=selected[stamp];BaseFrameGenerationAlgorithm.generate_frame(buffer.numpy(),image)
            path=directory/source.name;cv2.imwrite(str(path),image)
            source_label=source.parent.parent/'labels'/(source.stem+'.txt');shutil.copyfile(source_label,labels/source_label.name)
            if seq!=6:training.append(str(path))
            saved+=1
        iterator.close();assert saved==len(selected),(seq,saved,len(selected));counts[str(seq)]=saved
        print('native33 dataset',seq,saved,flush=True)
    for p in train:
        if p.name.startswith('fred'):continue
        dest=out/'dataset/train/images'/p.name;shutil.copyfile(p,dest)
        lab=p.parent.parent/'labels'/(p.stem+'.txt');shutil.copyfile(lab,out/'dataset/train/labels'/lab.name);training.append(str(dest))
    (out/'train_images.txt').write_text('\n'.join(training)+'\n','utf-8')
    (out/'dataset.yaml').write_text(f'path: {out.as_posix()}/dataset\ntrain: {out.as_posix()}/train_images.txt\nval: val/images\nnames:\n  0: drone\n','utf-8')
    design=json.loads((old/'protocol.json').read_text('utf-8'))
    parent=old/'training/weights/best.pt'
    design.update(parent_weight=str(parent),parent_weight_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),epochs=5,seed=3110159,
        source_window_us=33333,selection='One fixed5epoch run; FRED6 native33 validation only',
        scoring_excluded=[0,1,2,7,8,9,10],new_dataset_counts=counts,new_train_images=len(training))
    (out/'protocol.json').write_text(json.dumps(design,indent=2),'utf-8')


if __name__=='__main__':main(sys.argv[1])
