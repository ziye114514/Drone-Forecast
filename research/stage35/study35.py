from pathlib import Path
import argparse,datetime,importlib.util,json,sys,time
import cv2,numpy as np
from body35 import ROOT,PARAMS,EventImages,BodyTracker
from checks35 import run as checks

sys.path[:0]=[str(ROOT/'research/stage34'),str(ROOT/'research/stage23')]
import study34
import robust23
OUT=ROOT.parent/'output/stage35_body_feature_flow_20261018';CASES=dict(study34.CASES)
CASES['clear_dropout']=dict(CASES['clear6'],dropout=[23.,23.5])
METHODS=['baseline']+list(PARAMS['variants']);MODELS=robust23.MODELS;load=study34.load;save=study34.save;sha=study34.sha


def sources():
    result={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('body35.py'),Path(__file__).with_name('checks35.py')]}
    for n,v in load(study34.OUT/'protocol.json')['source_hashes'].items():assert sha(ROOT/n)==v;result[n]=v
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks())
    p=load(study34.OUT/'protocol.json');save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,cases=CASES,
        input_hashes={c:{key:sha(ROOT.parent/'output'/cfg[key]) for key in ['observation','forecast']} for c,cfg in CASES.items()},label_hashes=p['label_hashes'],opencv_version=cv2.__version__,
        design='Same frozen geometry at each matching RAW cutoff; pure native20ms ON+OFF count images for OpenCV GFTT/PyrLK forward-backward/RANSAC body motion. No G reconstruction and no annotated video input. Geometry seeds corners; flow-only frames never reseed, no invented rotors. Baseline positions/six models exact prior cache outside controlled-dropout case.',
        variants='Short300ms and long2s similarity, long2s translation from last independent geometry; all require current feature evidence. Long is stress-test for drift, not one clear frame guarantees success.',
        prior_art=['https://github.com/opencv/opencv/blob/5.x/doc/tutorials/others/optical_flow.markdown','https://github.com/ialzugaray/haste','https://github.com/uzh-rpg/rpg_eklt'],
        reuse='Execute installed OpenCV LK/GFTT/estimateAffinePartial2D; HASTE/EKLT inspected, not built/executed. EKLT requires frame/event complementarity, so not presented as this RAW-only implementation.',
        limits='Development known recordings, approximate sparse target boxes. Flow is measured image motion but inherited center/shape may drift, no semantic identity/3D pose claim. Dropout removes events AND baseline detections, diagnostic only. No G/yolov10-main.'))
    print('Stage35 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def run():
    protocol=verify()
    for case,cfg in CASES.items():
        for key in ['observation','forecast']:assert sha(ROOT.parent/'output'/cfg[key])==protocol['input_hashes'][case][key]
        baseline={round(r['t'],8):r['frontends'][cfg['front']] for r in map(json.loads,(ROOT.parent/'output'/cfg['forecast']).read_text(encoding='utf-8').splitlines())}
        reader=study34.open_raw(cfg['raw']);h,w=reader.get_size();images=EventImages(w,h);trackers={m:BodyTracker(w,h,m) for m in METHODS if m!='baseline'};models={m:robust23.Predictor() for m in METHODS}
        folder=OUT/case;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'forecasts.jsonl').exists();n=0;counts={m:dict(detected=0,body_measurements=0) for m in METHODS};began=time.perf_counter()
        with (ROOT.parent/'output'/cfg['observation']).open(encoding='utf-8') as fi,(folder/'observations.jsonl').open('x',encoding='utf-8') as fo,(folder/'forecasts.jsonl').open('x',encoding='utf-8') as ff:
            source=map(json.loads,fi)
            for t,events in study34.raw_chunks(reader,cfg['start'],cfg['duration'],10000):
                r=next(source);assert abs(r['t']-t)<1e-8 and len(events)==r['event_count'];o=r['observation']
                dropped=bool('dropout' in cfg and cfg['dropout'][0]<=t<cfg['dropout'][1])
                if dropped:events=events[:0];o=dict(t=t,detected=False,rotors=[])
                image,current,strong=images.step(events,t);obs={'baseline':dict(observation=o,audit=dict(source='geometry' if o['detected'] else 'none'))};forecast={}
                for m,tracker in trackers.items():
                    q,a=tracker.step(image,current,strong,t,o,len(events));obs[m]=dict(observation=q,audit=a)
                    if dropped:assert not q['detected'],(case,t,'body published without current events')
                for m,p in models.items():
                    q=obs[m]['observation'];forecast[m]=p.step(t,q['center'] if q['detected'] else None)[0]
                    counts[m]['detected']+=q['detected'];counts[m]['body_measurements']+=q.get('source')=='body_flow'
                if 'dropout' not in cfg:
                    old=baseline[round(t,8)];assert forecast['baseline']['valid']==old['valid']
                    for model in MODELS:assert forecast['baseline']['coefficients'].get(model)==old['coefficients'].get(model)
                fo.write(json.dumps(dict(t=t,methods=obs),allow_nan=False)+'\n');ff.write(json.dumps(dict(t=t,frontends=forecast),allow_nan=False)+'\n');n+=1
        save(folder/'completion.json',dict(frames=n,wall_s=time.perf_counter()-began,methods=counts,baseline_coefficients_exact=n if 'dropout' not in cfg else None,
            labels_in_inference=False,sources_unchanged=sources()==protocol['source_hashes']))
        print(json.dumps(dict(case=case,frames=n,methods=counts)),flush=True)


def score():
    verify();s=study34.scorer;s.OUT=OUT;s.INPUT=OUT;s.CASES=['clear6','side6','train3'];s.FRONTENDS=METHODS;s.MODELS=MODELS;s.verify=verify;s.evaluate=robust23.base.evaluate
    s.label_path=lambda c:ROOT.parent/'output/raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json');s.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','score','all']);a=p.parse_args()
    if a.action=='all':run();score()
    else:globals()[a.action]()
