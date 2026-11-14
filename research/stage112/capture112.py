"""RAW support-point capture with exact110 merge-safe observation regression."""
from pathlib import Path
import argparse,copy,datetime,json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage110'),str(ROOT/'research/stage109')]
from merge110 import MergeRecovery
from study110 import CASES,OUT as OLD
from audit109 import save,sha,rows
from windows21 import Observer
from event_drone_forecast.raw_io import open_raw,raw_chunks
OUT=ROOT.parent/'output/stage112_observed_support_20261022'

def key(r):
    if 'center' in r:c=r['center']
    else:c=[r['cx'],r['cy']]
    return tuple(float(v) for v in [*c,r['major'],r['minor'],r['angle']])

class SupportFrontend(MergeRecovery):
    def __init__(self,w,h):super().__init__(w,h,'merge_safe');self.supports={}
    def contours_before_merge(self,*args):
        result=super().contours_before_merge(*args);self.effective_contours=result;return result
    def detect_prepared(self,*args,**kwargs):
        self.supports={};return super().detect_prepared(*args,**kwargs)
    def _candidates(self,contours,activity,t,b,thin_only=False):
        result=super()._candidates(contours,activity,t,b,thin_only)
        used=contours if thin_only else self.effective_contours
        bybox={tuple(np.array(cv2.boundingRect(c))[[0,1,0,1]]+[0,0,cv2.boundingRect(c)[2],cv2.boundingRect(c)[3]]):c for c in used}
        for r in result:
            # Full/line measurements retain the exact originating contour box.
            bbox=tuple(np.asarray(r['bbox'])/b)
            c=bybox.get(bbox)
            if c is not None:self.supports[key(r)]=(c.reshape(-1,2).astype(float)*b).tolist()
        return result

def sources():
    files=[Path(__file__),ROOT/'research/stage110/merge110.py',ROOT/'research/stage61/preline61.py',
        ROOT/'src/event_drone_forecast/rotor_sensing.py',ROOT/'research/stage9/recovery.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}

def freeze():
    assert not OUT.exists()
    save(OUT/'protocol112.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources(),cases=CASES,
        input_sha256={k:sha(OLD/k/'records.jsonl') for k in CASES},
        design='Instrument110 merge-safe to retain exact selected contour points. Do not change candidate data, ordering, grouping or tracking. Newly read RAW geometry must equal110 merge-safe on each frame.',
        scope='Threshold/morphology contours supported by native bipolar events, not physical propeller edge ground truth. No inferred ellipse perimeter samples, NN-inferred rotor masks, future events or future labels. Matched contemporaneous body/axis remain cached.',
        default_promoted=False))

def run(case,prefix=False):
    p=json.loads((OUT/'protocol112.json').read_text('utf-8'));assert p['sources']==sources()
    cfg=CASES[case];assert p['input_sha256'][case]==sha(OLD/case/'records.jsonl')
    reference={round(r['t'],8):r for r in rows(OLD/case/'records.jsonl')}
    folder=OUT/(case+('_prefix' if prefix else ''));folder.mkdir(exist_ok=False)
    reader=open_raw(cfg['raw']);h,w=reader.get_size();front=Observer(w,h,'native20');front.geometry=SupportFrontend(w,h)
    begin=max(0.,cfg['start']-.1);duration=min(cfg['duration'],1.2) if prefix else cfg['duration'];result=[]
    with (folder/'supports.jsonl').open('x',encoding='utf-8') as stream:
        for t,e in raw_chunks(reader,begin,duration+cfg['start']-begin,20000):
            obs=front.step(e,t)['observation'] if t-begin>=.1-1e-8 else dict(t=t,detected=False,rotors=[])
            if t<=cfg['start']+1e-8:continue
            r=reference[round(t,8)];assert obs==r['outputs']['merge_safe']['geometry'],(case,t,'capture changed geometry')
            support=[front.geometry.supports.get(key(z),[]) for z in obs.get('rotors',[])]
            q=dict(t=t,geometry=obs,support_points=support,body=r['body'],axis=r['axis'],valid=r['valid'])
            stream.write(json.dumps(q,allow_nan=False)+'\n');result.append(q)
            if len(result)%100==0:print(f'{case} captured {len(result)}',flush=True)
    assert len(result)==round(duration*50)
    missing=sum(len(p)<6 for r in result for p in r['support_points']);count=sum(len(r['support_points']) for r in result)
    save(folder/'checks112.json',dict(frames=len(result),actual_RAW=True,selected_rotors=count,support_unmatched=missing,
        exact110_geometry=True,source_unchanged=p['sources']==sources(),labels_read=False))
    if prefix:
        full=rows(OUT/case/'supports.jsonl')[:len(result)];assert full==result
        save(folder/'prefix112.json',dict(actual_truncated_RAW=True,frames=len(result),supports_and_geometry_exact=True))
    print(json.dumps(dict(case=case,prefix=prefix,frames=len(result),support_unmatched=missing)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','prefix']);p.add_argument('--case',choices=CASES)
    a=p.parse_args();freeze() if a.action=='freeze' else run(a.case,a.action=='prefix')
