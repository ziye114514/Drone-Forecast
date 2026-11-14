import copy,json
import numpy as np
from common41 import *
from robust23 import Predictor,MODELS

def body(row,t):
    if not row['boxes']:return dict(t=t,detected=False,center=None,rotors=[],source='body')
    b=max(row['boxes'],key=lambda b:b['confidence']);box=b['bbox'];return dict(t=t,detected=True,center=[(box[0]+box[2])/2,(box[1]+box[3])/2],bbox=box,confidence=b['confidence'],rotors=[],source='body')

def main():
    p=verify();assert sha(OUT/'detections.jsonl')==load(OUT/'inference.json')['predictions_sha256'];detections={(r['case'],round(r['t'],8),r['method']):r for r in map(json.loads,(OUT/'detections.jsonl').read_text(encoding='utf-8').splitlines())};stats={}
    for case in CASES:
        models={m:Predictor() for m in METHODS};counts={m:0 for m in METHODS};valid={m:0 for m in METHODS};folder=OUT/case;assert not (folder/'forecasts.jsonl').exists();n=0
        with (folder/'observations.jsonl').open('x',encoding='utf-8') as fo,(folder/'forecasts.jsonl').open('x',encoding='utf-8') as ff:
            for row in map(json.loads,(folder/'geometry.jsonl').read_text(encoding='utf-8').splitlines()):
                t=row['t'];g=row['observation'];b=body(detections[case,round(t,8),'finetuned'],t);old=body(detections[case,round(t,8),'pretrained'],t)
                obs=dict(geometry=g,pretrained_only=old,body_only=b,geometry_first=g if g['detected'] else b,body_first=b if b['detected'] else g);forecast={}
                for method,o in obs.items():
                    q,_=models[method].step(t,o['center'] if o['detected'] else None);forecast[method]=q;counts[method]+=o['detected'];valid[method]+=q['valid']
                    if method=='geometry':assert q==row['forecast']
                fo.write(json.dumps(dict(t=t,methods={m:dict(observation=o) for m,o in obs.items()}),allow_nan=False)+'\n');ff.write(json.dumps(dict(t=t,frontends=forecast),allow_nan=False)+'\n');n+=1
        stats[case]=dict(frames=n,published=counts,valid_forecasts=valid);print(case,stats[case],flush=True)
    save(OUT/'replay.json',dict(cases=stats,labels_read=False,sources_unchanged=verify()['source_hashes']==p['source_hashes']))
if __name__=='__main__':main()
