from pathlib import Path
import argparse,datetime,importlib.util,json,sys,time
import numpy as np
from noise39 import StreamFilter,PARAMS,checks

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage9'),str(ROOT/'research/stage23'),str(ROOT/'research/stage22')]
from recovery import RecoveryFrontend
from robust23 import Predictor,MODELS
from event_drone_forecast.raw_io import open_raw,raw_chunks
spec=importlib.util.spec_from_file_location('score39',ROOT/'research/stage22/study22.py');scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
load=scorer.load;save=scorer.save;sha=scorer.sha
OUT=ROOT.parent/'output/stage39_sdk_noise_geometry_20261018';METHODS=['original','activity']
CASES={'clear6':dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=5.,observation='stage12_clear_22_27_20261017/observations.jsonl',forecast='stage29_delayed_model_weights_20261018/clear6/forecasts.jsonl',front='delayed_weights'),
    'side6':dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=12.,observation='stage12_side_108_120_20261017/observations.jsonl',forecast='stage29_delayed_model_weights_20261018/side6/forecasts.jsonl',front='delayed_weights'),
    'train3':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=6.,observation='stage31_cross_recording_physics_20261018/train3/observations_forecasts.jsonl',forecast='stage31_cross_recording_physics_20261018/train3/forecasts.jsonl',front='raw_adaptive_physics'),
    'negative5':dict(raw='E:/dataset/train/5/media/event.raw',start=0.,duration=8.)}


def sources():
    result={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('noise39.py')]}
    for name,value in load(ROOT.parent/'output/stage38_noise_scale_detector_20261018/protocol.json')['source_hashes'].items():assert sha(ROOT/name)==value;result[name]=value
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks())
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),parameters=PARAMS,cases=CASES,
        input_hashes={c:{key:sha(ROOT.parent/'output'/cfg[key]) for key in ['observation','forecast']} for c,cfg in CASES.items() if 'observation' in cfg},
        label_hashes=load(ROOT.parent/'output/stage37_upstream_detector_20261018/protocol.json')['label_hashes'],
        design='Same real RAW10ms current chunks, same legacy RecoveryFrontend(neighbor10), same six history predictors. Only branch activity uses persistent SDK10ms activity filter before unchanged geometry. Original full observations/six coefficients must exactly reproduce cache for clear/side/train3. Negative8s separately checked.',
        prior_art='https://docs.prophesee.ai/stable/api/python/cv/bindings.html#metavision_sdk_cv.ActivityNoiseFilterAlgorithm',
        limits='No neural detector or rotor frequency/physics recomputation. Isolate event cleaning contribution to current geometry and history forecasts. Development data, coarse sparse boxes, no pose/RPM truth; no G/yolov10-main.'))
    print('Stage39 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def json_text(value):
    return json.dumps(value,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())


def run():
    protocol=verify()
    for case,cfg in CASES.items():
        if 'observation' in cfg:
            for key in ['observation','forecast']:assert sha(ROOT.parent/'output'/cfg[key])==protocol['input_hashes'][case][key]
            old={round(r['t'],8):r for r in map(json.loads,(ROOT.parent/'output'/cfg['observation']).read_text(encoding='utf-8').splitlines())}
            oldf={round(r['t'],8):r['frontends'][cfg['front']] for r in map(json.loads,(ROOT.parent/'output'/cfg['forecast']).read_text(encoding='utf-8').splitlines())}
        reader=open_raw(cfg['raw']);h,w=reader.get_size();noise=StreamFilter(w,h);front={m:RecoveryFrontend(w,h,'neighbor10') for m in METHODS};models={m:Predictor() for m in METHODS}
        folder=OUT/case;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'observations.jsonl').exists();n=raw_count=kept_count=0;began=time.perf_counter();timing={m:[] for m in METHODS};published={m:0 for m in METHODS}
        with (folder/'observations.jsonl').open('x',encoding='utf-8') as fo,(folder/'forecasts.jsonl').open('x',encoding='utf-8') as ff:
            for t,e in raw_chunks(reader,cfg['start'],cfg['duration'],10000):
                tick=time.perf_counter();clean=noise.step(e);filter_ms=(time.perf_counter()-tick)*1000
                assert len(clean)<=len(e)
                if len(clean):assert float(clean['t'].min())>=t*1e6-10000-1e-4 and float(clean['t'].max())<t*1e6+1e-4
                row=dict(t=t,event_count=len(e),filtered_events=len(clean),filter_ms=filter_ms,methods={});prediction={}
                for m in (METHODS if n%2 else METHODS[::-1]):
                    tick=time.perf_counter();o,_=front[m].detect(e if m=='original' else clean,t);q,_=models[m].step(t,o['center'] if o['detected'] else None)
                    elapsed=(time.perf_counter()-tick)*1000+(filter_ms if m=='activity' else 0);timing[m].append(elapsed);published[m]+=o['detected'];row['methods'][m]=dict(observation=o,processing_ms=elapsed);prediction[m]=q
                if 'observation' in cfg:
                    assert row['methods']['original']['observation']==old[round(t,8)]['observation'],(case,t,'geometry baseline differs')
                    assert prediction['original']['valid']==oldf[round(t,8)]['valid']
                    for model in MODELS:assert prediction['original']['coefficients'].get(model)==oldf[round(t,8)]['coefficients'].get(model)
                fo.write(json_text(row)+'\n');ff.write(json_text(dict(t=t,frontends=prediction))+'\n');n+=1;raw_count+=len(e);kept_count+=len(clean)
        save(folder/'completion.json',dict(frames=n,published=published,events_raw=raw_count,events_retained=kept_count,retention=kept_count/max(raw_count,1),
            processing_median_ms={m:float(np.median(v)) for m,v in timing.items()},processing_p95_ms={m:float(np.percentile(v,95)) for m,v in timing.items()},
            baseline_observations_and_six_coefficients_exact=n if 'observation' in cfg else None,wall_s=time.perf_counter()-began,labels_in_inference=False,sources_unchanged=sources()==protocol['source_hashes'],
            timing_scope='Per branch geometry+prediction plus filtering when applicable, excludes RAW decode/write; paired process, no full real-time claim.'))
        print(json.dumps(dict(case=case,frames=n,published=published,retention=kept_count/max(raw_count,1))),flush=True)


def score():
    verify();scorer.OUT=OUT;scorer.INPUT=OUT;scorer.CASES=list(CASES);scorer.FRONTENDS=METHODS;scorer.MODELS=MODELS;scorer.verify=verify
    scorer.label_path=lambda c:ROOT.parent/'output/raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json');scorer.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','score','all']);a=p.parse_args()
    if a.action=='all':run();score()
    else:globals()[a.action]()
