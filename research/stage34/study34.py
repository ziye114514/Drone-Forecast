"""Replay frozen current geometry with real causal RAW histories; sensor-only change."""
from pathlib import Path
from collections import deque
import argparse,datetime,hashlib,importlib.util,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage33'),str(ROOT/'research/stage28'),str(ROOT/'research/stage29'),str(ROOT/'research/stage22')]
from slots33 import ReplaceStale,RotorSensing,PaperConfig,PARAMS as SLOT_PARAMS
from axis28 import Predictor
from select29 import Selector,MODELS,prior
from event_drone_forecast.raw_io import open_raw,raw_chunks
spec=importlib.util.spec_from_file_location('scorer34',ROOT/'research/stage22/study22.py');scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
OUT=ROOT.parent/'output/stage34_slot_frequency_forecast_20261018'
CASES={'clear6':dict(raw='E:/dataset/train/6/media/event.raw',start=22.,duration=2.,observation='stage12_clear_22_27_20261017/observations.jsonl',forecast='stage29_delayed_model_weights_20261018/clear6/forecasts.jsonl',front='delayed_weights'),
    'side6':dict(raw='E:/dataset/train/6/media/event.raw',start=108.,duration=5.,observation='stage12_side_108_120_20261017/observations.jsonl',forecast='stage29_delayed_model_weights_20261018/side6/forecasts.jsonl',front='delayed_weights'),
    'train3':dict(raw='E:/dataset/train/3/aligned/event.raw',start=25.,duration=6.,observation='stage31_cross_recording_physics_20261018/train3/observations_forecasts.jsonl',forecast='stage31_cross_recording_physics_20261018/train3/forecasts.jsonl',front='raw_adaptive_physics')}
METHODS=['original','replace_stale'];load=scorer.load;save=scorer.save;sha=scorer.sha


def sources():
    result={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for name,value in load(ROOT.parent/'output/stage33_rotor_slot_association_20261018/protocol.json')['source_hashes'].items():assert sha(ROOT/name)==value;result[name]=value
    return result


def freeze():
    assert not (OUT/'protocol.json').exists();metadata={}
    for c in CASES.values():
        p=Path(c['raw']);st=p.stat();metadata[str(p)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,methods=METHODS,models=MODELS,
        slot_parameters=SLOT_PARAMS,input_hashes={c:{key:sha(ROOT.parent/'output'/cfg[key]) for key in ['observation','forecast']} for c,cfg in CASES.items()},raw_metadata=metadata,
        label_hashes={name:sha(ROOT.parent/'output/raw_only_project/evaluation_data'/name) for name in ['train6.json','train3.json']},
        design='Same cached current observation consumed sequentially only at matching RAW cutoff. Real RAW causal histories feed BOTH unchanged frequency algorithms. Original sensing and full15 forecasts must exactly reproduce prior cache. Only replacement lifecycle changes; on calibration-epoch change reset physics features/parameters (keep position/history and model-selector feedback).',
        evaluation='All forecasts saved before sparse-box scoring. Same position/velocity/validity and six historical experts. Report availability, ID churn and actual FDE; internal four-frequency validity is not RPM truth.',
        scope='Development/known cross-recording diagnostic, no result tuning. Cached current geometry is not label input; event time alignment and prefix exactness checked. No G/yolov10-main.'))
    print('Stage34 frozen',flush=True)


def verify():
    p=load(OUT/'protocol.json');assert sources()==p['source_hashes'];return p


def run():
    protocol=verify()
    for case,cfg in CASES.items():
        for key in ['observation','forecast']:assert sha(ROOT.parent/'output'/cfg[key])==protocol['input_hashes'][case][key]
        baseline={round(r['t'],8):r['frontends'][cfg['front']] for r in map(json.loads,(ROOT.parent/'output'/cfg['forecast']).read_text(encoding='utf-8').splitlines())}
        reader=open_raw(cfg['raw']);sensors={'original':RotorSensing(PaperConfig()),'replace_stale':ReplaceStale(PaperConfig())}
        predictors={m:Predictor('canonical_static') for m in METHODS};selectors={m:Selector() for m in METHODS};history=deque();cfgpaper=PaperConfig()
        folder=OUT/case;folder.mkdir(parents=True,exist_ok=True);assert not (folder/'forecasts.jsonl').exists();n=0;last_epoch=0;started=time.perf_counter();last_print=started
        summary={m:dict(four_visible_ticks=0,four_frequency_ticks=0,physics_active_ticks=0,retirements=0,physics_resets=0) for m in METHODS}
        with (ROOT.parent/'output'/cfg['observation']).open(encoding='utf-8') as source,(folder/'sensing.jsonl').open('x',encoding='utf-8') as fs,(folder/'forecasts.jsonl').open('x',encoding='utf-8') as ff:
            observations=map(json.loads,source)
            for t,events in raw_chunks(reader,cfg['start'],cfg['duration'],10000):
                r=next(observations);assert abs(r['t']-t)<1e-8 and r['event_count']==len(events),(case,t,'current geometry RAW clock differs')
                stride=max(1,int(np.ceil(len(events)/max(1,cfgpaper.max_buffer_events//6))));history.append((t,events[::stride].copy()))
                while history and history[0][0]<t-cfgpaper.frequency_window_us/1e6-.03:history.popleft()
                sensed={m:s.step(r['observation'],history,t) for m,s in sensors.items()};assert sensed['original']==r['sensing'],(case,t,'baseline sensing differs')
                epoch=sensors['replace_stale'].calibration_epoch
                if epoch!=last_epoch:predictors['replace_stale'].reset_features();summary['replace_stale']['physics_resets']+=1;last_epoch=epoch
                forecast={}
                for method,predictor in predictors.items():
                    q,_=predictor.step(dict(t=t,observation=r['observation'],sensing=sensed[method]));forecast[method]=selectors[method].step(q,r['observation'])
                    summary[method]['four_visible_ticks']+=sensed[method]['four_visible'];summary[method]['four_frequency_ticks']+=sensed[method]['four_frequencies'];summary[method]['physics_active_ticks']+=q['feature_diagnostic']['active']
                    summary[method]['retirements']+=len(getattr(sensors[method],'audit',{}).get('retired_ids',[]))
                assert forecast['original']==baseline[round(t,8)],(case,t,'baseline forecast differs')
                assert forecast['original']['valid']==forecast['replace_stale']['valid']
                for model in prior.robust.MODELS:assert forecast['original']['coefficients'].get(model)==forecast['replace_stale']['coefficients'].get(model)
                fs.write(json.dumps(dict(t=t,methods=sensed,slot_audit=sensors['replace_stale'].audit),allow_nan=False)+'\n')
                ff.write(json.dumps(dict(t=t,frontends=forecast),allow_nan=False)+'\n');n+=1
                if time.perf_counter()-last_print>20:print(json.dumps(dict(case=case,t=t,frames=n)),flush=True);last_print=time.perf_counter()
        for m,s in sensors.items():summary[m]['new_ids']=s.next_id
        save(folder/'completion.json',dict(frames=n,baseline_sensing_exact=n,baseline_all_forecasts_exact=n,six_history_models_identical=True,
            sources_unchanged=sources()==protocol['source_hashes'],labels_in_inference=False,wall_s=time.perf_counter()-started,methods=summary))
        print('Completed '+case,flush=True)


def score():
    verify();scorer.OUT=OUT;scorer.INPUT=OUT;scorer.CASES=list(CASES);scorer.FRONTENDS=METHODS;scorer.MODELS=MODELS;scorer.verify=verify;scorer.evaluate=prior.evaluate
    scorer.label_path=lambda c:ROOT.parent/'output/raw_only_project/evaluation_data'/('train3.json' if c=='train3' else 'train6.json');scorer.score()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','score','all']);a=p.parse_args()
    if a.action=='all':run();score()
    else:globals()[a.action]()
