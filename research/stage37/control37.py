"""Known-domain FRED event-frame functionality control; not a held-out benchmark."""
from pathlib import Path
import argparse,datetime,json,re,zipfile
import cv2,numpy as np
from common37 import ROOT,OUT,PARAMS,load,save,sha,verify


def prepare():
    verify();assert not (OUT/'control_inputs.json').exists();inputs=[];targets=[]
    save(OUT/'control_protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),main_protocol_sha256=sha(OUT/'protocol.json'),
        sequences=[0,4,6],area_quantiles=[.25,.5,.9],parameters=PARAMS,
        scope='Post-hoc positive functionality control using unmodified original Event/Frames PNGs from local FRED archives. Upstream weights trained on FRED, so never claim held-out/generalization accuracy. No RGB/reconstructed G or tuning.'))
    for seq in [0,4,6]:
        coords=ROOT.parent/f'output/stage7_localization_budget_v3_20261017/data/sequence{seq}/coordinates.txt'
        with zipfile.ZipFile(Path(f'E:/FRED/{seq}.zip')) as z:
            names={Path(n).name:n for n in z.namelist() if n.startswith('Event/Frames/') and n.endswith('.png')};candidates=[]
            for line in coords.read_text(encoding='utf-8').splitlines():
                time_text,rest=line.split(':',1);box=[float(x) for x in rest.split(',')[:4]]
                if box[2]<=box[0] or box[3]<=box[1]:continue
                key=time_text.replace('.','');key=key[1:] if key.startswith('0') else key
                name=f'Video_{seq}_frame_{key}.png'
                if name in names:candidates.append(dict(t=float(time_text),bbox=box,member=names[name],area=(box[2]-box[0])*(box[3]-box[1])))
            assert candidates,(seq,'No labelled original event PNG matches')
            candidates.sort(key=lambda x:x['area'])
            for q in [.25,.5,.9]:
                c=candidates[min(len(candidates)-1,int(q*(len(candidates)-1)))];identifier=f'fred{seq}_q{q:.2f}';path=OUT/f'control_images/{identifier}.png';path.parent.mkdir(exist_ok=True)
                path.write_bytes(z.read(c['member']));image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR);assert image is not None
                inputs.append(dict(id=identifier,image=str(path.relative_to(OUT)),image_sha256=sha(path),shape=list(image.shape)))
                targets.append(dict(id=identifier,sequence=seq,**c))
    save(OUT/'control_inputs.json',dict(images=inputs));save(OUT/'control_targets.json',dict(targets=targets));print('Prepared original FRED event controls: '+str(len(inputs)),flush=True)


def predict():
    from model37 import load_detector
    verify();assert not (OUT/'control_predictions.json').exists();model=load_detector();result=[]
    for r in load(OUT/'control_inputs.json')['images']:
        assert sha(OUT/r['image'])==r['image_sha256'];image=cv2.imdecode(np.fromfile(OUT/r['image'],np.uint8),cv2.IMREAD_COLOR)
        prediction=model.predict(image,imgsz=PARAMS['imgsz'],conf=PARAMS['confidence'],iou=PARAMS['iou_nms'],device=PARAMS['device'],verbose=False,save=False,augment=False)[0]
        rows=[] if prediction.boxes is None else prediction.boxes.data.cpu().numpy().tolist();result.append(dict(id=r['id'],boxes=rows))
    save(OUT/'control_predictions.json',dict(results=result,labels_read=False,scope='Positive functionality control, NOT FRED accuracy benchmark.'))
    print(json.dumps([(r['id'],len(r['boxes']),max([b[4] for b in r['boxes']],default=None)) for r in result]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','predict']);a=p.parse_args();globals()[a.action]()
