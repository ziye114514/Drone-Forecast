"""Exclude unobservable positive TRAIN images; keep files and all validation.

No pseudo negative labels: a low-support positive image is omitted as a whole.
This gate does not filter any test/scoring tick or prediction.
"""
from pathlib import Path
import json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'output/fred_domain_adaptation_20261029'


def main():
    accepted=[];excluded=[]
    for p in sorted((OUT/'dataset/train/images').glob('*.png')):
        if not p.name.startswith('fred'):
            accepted.append(str(p));continue
        label=OUT/'dataset/train/labels'/p.with_suffix('.txt').name
        v=np.array(list(map(float,label.read_text().split()[1:])))*[1280,720,1280,720]
        b=np.r_[v[:2]-v[2:]/2,v[:2]+v[2:]/2]
        lo=np.maximum(0,np.floor(b[:2]).astype(int));hi=np.minimum([1280,720],np.ceil(b[2:]).astype(int))
        im=cv2.imread(str(p));patch=im[lo[1]:hi[1],lo[0]:hi[0]]
        count=int(np.sum(patch[:,:,0]!=52))
        if count<16:excluded.append(dict(image=p.name,current_native_unique_support=count))
        else:accepted.append(str(p))
    (OUT/'train_images.txt').write_text('\n'.join(accepted)+'\n','utf-8')
    (OUT/'dataset.yaml').write_text('path: '+(OUT/'dataset').as_posix()+'\ntrain: '+(OUT/'train_images.txt').as_posix()+'\nval: val/images\nnames:\n  0: drone\n','utf-8')
    report=dict(accepted_training_images=len(accepted),excluded_low_support=len(excluded),
        validation_unchanged=True,original_files_retained=True,excluded=excluded)
    (OUT/'training_support_filter.json').write_text(json.dumps(report,indent=2),'utf-8')
    p=OUT/'protocol.json';d=json.loads(p.read_text());d['training_positive_min_native_distinct_pixels']=16
    d['low_support_positives_omitted_not_relabelled_negative']=True
    p.write_text(json.dumps(d,indent=2),'utf-8');print(json.dumps({k:v for k,v in report.items() if k!='excluded'}))


if __name__=='__main__':main()
