import json,time
import cv2,numpy as np
from common41 import *
from windows21 import Observer
from robust23 import Predictor,MODELS
from event_drone_forecast.raw_io import open_raw,raw_chunks
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_core import BaseFrameGenerationAlgorithm

def main():
    p=verify();assert not (OUT/'inputs.json').exists();inputs=[];summary={}
    for case,cfg in CASES.items():
        oldpath=ROOT.parent/'output/stage24_raw_pipeline_20261018/train3_extra/observations_forecasts.jsonl' if case=='train3' else ROOT.parent/f'output/stage21_native_windows_20261018/{case}/observations.jsonl'
        old={round(r['t'],8):(r['observation'] if case=='train3' else r['methods']['native20']['observation']) for r in map(json.loads,oldpath.read_text(encoding='utf-8').splitlines())}
        fp=oldpath if case=='train3' else ROOT.parent/f'output/stage23_robust_acceleration_20261018/{case}/forecasts.jsonl'
        oldf={round(r['t'],8):(r['forecasting'] if case=='train3' else r['frontends']['native20']) for r in map(json.loads,fp.read_text(encoding='utf-8').splitlines())}
        reader=open_raw(cfg['raw']);h,w=reader.get_size();front=Observer(w,h,'native20');model=Predictor();alg=ActivityNoiseFilterAlgorithm(w,h,10000);buf=alg.get_empty_output_buffer()
        if cfg['start']:
            reader.seek_time(round(cfg['start']*1e6)-10000);warm=reader.load_delta_t(10000);alg.process_events(warm,buf)
        folder=OUT/case;folder.mkdir(parents=True,exist_ok=False);n=published=0;began=time.perf_counter()
        with (folder/'geometry.jsonl').open('x',encoding='utf-8') as f:
            for t,e in raw_chunks(reader,cfg['start'],cfg['duration'],20000):
                tick=time.perf_counter();o=front.step(e,t)['observation'];q,_=model.step(t,o['center'] if o['detected'] else None);geometry_ms=(time.perf_counter()-tick)*1000
                assert o==old[round(t,8)],(case,t,'original geometry differs')
                assert q['valid']==oldf[round(t,8)]['valid']
                for m in MODELS:assert q['coefficients'].get(m)==oldf[round(t,8)]['coefficients'].get(m),(case,t,m)
                tick=time.perf_counter();alg.process_events(e,buf);clean=buf.numpy().copy();assert not len(clean) or int(clean['t'].max())<round(t*1e6)
                image=np.empty((h,w,3),np.uint8);BaseFrameGenerationAlgorithm.generate_frame(clean,image);prep_ms=(time.perf_counter()-tick)*1000
                sample_id=f'{case}_{n:05d}';name=f'{case}/images/{sample_id}.png';path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);cv2.imencode('.png',image)[1].tofile(str(path))
                inputs.append(dict(id=sample_id,case=case,t=t,image=name,image_sha256=sha(path),width=w,height=h))
                f.write(json.dumps(dict(t=t,observation=o,forecast=q,event_count=len(e),filtered_events=len(clean),geometry_ms=geometry_ms,body_preparation_ms=prep_ms),allow_nan=False)+'\n');n+=1;published+=o['detected']
        summary[case]=dict(frames=n,geometry_published=published,full_observation_and_six_predictors_exact=n,wall_s=time.perf_counter()-began);print(case,summary[case],flush=True)
    save(OUT/'inputs.json',dict(images=inputs,scope='Native full-frame event images, no labels/boxes/states.'))
    save(OUT/'preparation.json',dict(cases=summary,input_sha256=sha(OUT/'inputs.json'),sources_unchanged=verify()['source_hashes']==p['source_hashes']))
if __name__=='__main__':main()
