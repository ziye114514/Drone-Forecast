import argparse,datetime,json,sys
import cv2,numpy as np
from common37 import ROOT,OUT,CASES,PARAMS,load,save,sha,sources,verify
from checks37 import run as checks
from render37 import past,render
sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.raw_io import open_raw


def samples():
    result=[]
    for case,cfg in CASES.items():
        if 'negative_timestamps' in cfg:labels=[dict(t=t,state='absent') for t in cfg['negative_timestamps']]
        else:labels=[l for l in load(ROOT.parent/'output/raw_only_project/evaluation_data'/cfg['labels'])['labels'] if cfg['start']<l['t']<=cfg['start']+cfg['duration']]
        for i,l in enumerate(labels):
            cut=int(np.floor(l['t']*1e6))//10000*10000
            result.append(dict(id=f'{case}_{i:03}',case=case,cutoff_us=cut,target_t=l['t'],state=l['state'],bbox=l.get('bbox')))
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();assert (OUT/'model_metadata.json').exists();save(OUT/'module_checks.json',checks());metadata={}
    for cfg in CASES.values():
        path=__import__('pathlib').Path(cfg['raw']);st=path.stat();metadata[str(path)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    assets=['upstream_assets.json','serialization_audit.json','model_metadata.json','architecture.yaml','tensor_state.pt','upstream/yolo/best.pt','upstream/yolo/render_events.py']
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,parameters=PARAMS,raw_metadata=metadata,
        asset_hashes={name:sha(OUT/name) for name in assets},sample_selection=[{k:s[k] for k in ['id','case','cutoff_us','target_t']} for s in samples()],
        label_hashes={name:sha(ROOT.parent/'output/raw_only_project/evaluation_data'/name) for name in ['train6.json','train3.json']},
        design='Pinned DroneStalker detector strict tensor-state reconstruction with zero-input exact parity. No retraining or G. Three predefined native representations and10/50ms windows. Prepare inSDK311, infer inTorch313 from inputs.json/images only; GT in separate targets.json. All predictions saved before scoring.',
        amendment='Pre-inference v2 removes GT states/boxes from protocol. Original v1/sources retained in freeze_v1_before_inference; no inputs/predictions existed then. Inference file-access guard rejects evaluation truth paths.',
        scope='External detector migration pilot on development self captures, not FRED blind benchmark or future prediction improvement. Upstream says trained on entire FRED; own data are different. No access/execute/modify yolov10-main.'))
    print('Stage37 frozen',flush=True)


def prepare():
    p=verify();assert not (OUT/'inputs.json').exists();inputs=[];targets=[];truth={s['id']:s for s in samples()}
    for case,cfg in CASES.items():
        reader=open_raw(cfg['raw']);h,w=reader.get_size();assert (h,w)==(720,1280)
        if 'baseline' in cfg:
            baseline=list(map(json.loads,(ROOT.parent/'output'/cfg['baseline']).read_text(encoding='utf-8').splitlines()));bt=np.array([r['t'] for r in baseline])
        for s in [s for s in p['sample_selection'] if s['case']==case]:
            target=dict(truth[s['id']]);assert all(target[k]==s[k] for k in s)
            if 'baseline' in cfg:
                i=int(np.searchsorted(bt,s['cutoff_us']/1e6+1e-8,side='right')-1)
                target['baseline_geometry']=baseline[i]['observation'] if i>=0 and s['cutoff_us']/1e6-bt[i]<=.025 else None
            targets.append(target)
            for window in PARAMS['windows_us']:
                reader.seek_time(s['cutoff_us']-window);events=past(reader.load_delta_t(window),s['cutoff_us'],window)
                assert not len(events) or int(events['t'].max())<s['cutoff_us']
                for kind in PARAMS['formats']:
                    image=render(events,w,h,kind);name=f'images/{s["id"]}_{window}_{kind}.png';path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
                    cv2.imencode('.png',image)[1].tofile(str(path))
                    inputs.append(dict(id=s['id'],case=case,cutoff_us=s['cutoff_us'],window_us=window,format=kind,image=name,image_sha256=sha(path),events=len(events),width=w,height=h))
        print('Prepared '+case,flush=True)
    save(OUT/'inputs.json',dict(images=inputs,scope='No boxes/states/future labels. Pure causal event renderings.'))
    save(OUT/'targets.json',dict(targets=targets,scope='Evaluation only; never passed to detector. Existing approximate whole-UAV boxes, not COM.'))
    save(OUT/'preparation.json',dict(images=len(inputs),targets=len(targets),inputs_sha256=sha(OUT/'inputs.json'),targets_sha256=sha(OUT/'targets.json'),sources_unchanged=sources()==p['source_hashes']))
    print('Stage37 causal input images prepared: '+str(len(inputs)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare']);a=p.parse_args();globals()[a.action]()
