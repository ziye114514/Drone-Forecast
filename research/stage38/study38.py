import argparse,datetime,sys
import cv2,numpy as np
from common38 import ROOT,OUT,OLD,CASES,PARAMS,load,save,sha,sources,verify
from render38 import activity_filter,render
from checks38 import run as checks
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'src'))
from event_drone_forecast.raw_io import open_raw


def freeze():
    assert not (OUT/'protocol.json').exists();save(OUT/'module_checks.json',checks());old=load(OLD/'protocol.json')
    names=['inputs.json','predictions.jsonl','inference.json','model_metadata.json','tensor_state.pt','architecture.yaml']
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=CASES,parameters=PARAMS,
        sample_selection=old['sample_selection'],parent_frozen_artifacts={n:sha(OLD/n) for n in names},eval_target_sha256=sha(OLD/'targets.json'),
        amendment='Pre-inference v2 keeps GT target hash evaluation-only; runtime artifact verification does not open it. Prior freeze/source retained. No preparation or predictions had run.',
        design='Same pinned detector/weights/confidence, same40times and native event palette.2x3 noise/scale factorial at both10/50ms. ActivityNoiseFilter uses preceding60ms with10ms warmup before50ms window. Map all boxes back to original pixels; no GT-fed model/NMS. Divisor1/unfiltered input hashes must exactly match parent SDK images.',
        prior_art='https://docs.prophesee.ai/stable/api/python/cv/bindings.html#metavision_sdk_cv.ActivityNoiseFilterAlgorithm ; original public detector. Spatial event rebinning/image scale adaptation is standard, not novel algorithm.',
        scope='Post-development migration diagnostic, no training/threshold tuning, no blind accuracy/real-time/future prediction claim. Truth kept only in parent evaluation file, inference file guard blocks it. No G/yolov10-main.'))
    print('Stage38 frozen',flush=True)


def prepare():
    p=verify();assert not (OUT/'inputs.json').exists();images=[];counts=[];old={ (r['id'],r['window_us']):r for r in load(OLD/'inputs.json')['images'] if r['format']=='sdk_default'}
    for case,cfg in CASES.items():
        reader=open_raw(cfg['raw']);h,w=reader.get_size();assert (h,w)==(720,1280)
        for s in sorted([s for s in p['sample_selection'] if s['case']==case],key=lambda s:s['cutoff_us']):
            begin=s['cutoff_us']-PARAMS['context_us']
            if reader.current_time>begin:reader=open_raw(cfg['raw'])
            reader.seek_time(begin);whole=reader.load_delta_t(PARAMS['context_us']);whole=whole[(whole['t']>=begin)&(whole['t']<s['cutoff_us'])]
            selected={False:whole,True:activity_filter(whole,w,h)}
            for denoise,history in selected.items():
                for window in PARAMS['windows_us']:
                    e=history[history['t']>=s['cutoff_us']-window]
                    assert not len(e) or int(e['t'].max())<s['cutoff_us'];counts.append(dict(id=s['id'],noise=denoise,window_us=window,events=len(e)))
                    for d in PARAMS['divisors']:
                        frame,view=render(e,w,h,d);name=f'images/{s["id"]}_n{int(denoise)}_d{d}_w{window}.png';path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
                        cv2.imencode('.png',frame)[1].tofile(str(path));digest=sha(path)
                        if not denoise and d==1:assert digest==old[s['id'],window]['image_sha256'],(s['id'],window,'raw input differs')
                        images.append(dict(id=s['id'],case=case,noise=denoise,divisor=d,window_us=window,cutoff_us=s['cutoff_us'],viewport=view,image=name,image_sha256=digest,width=w,height=h,events=len(e)))
        print('Prepared noise/scale views: '+case,flush=True)
    save(OUT/'inputs.json',dict(images=images,scope='No truth states or boxes. Native causal inputs.'))
    save(OUT/'preparation.json',dict(images=len(images),targets=len(p['sample_selection']),input_hash=sha(OUT/'inputs.json'),unfiltered_divisor1_hashes_exact=len(old),counts=counts,sources_unchanged=verify()['source_hashes']==p['source_hashes']))
    print('Stage38 prepared views: '+str(len(images)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare']);a=p.parse_args();globals()[a.action]()
