import importlib.util,json
import numpy as np
from common41 import *
from robust23 import MODELS

def main():
    verify();spec=importlib.util.spec_from_file_location('sparse41',ROOT/'research/stage22/study22.py');scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    scorer.OUT=OUT;scorer.INPUT=OUT;scorer.CASES=list(CASES);scorer.FRONTENDS=METHODS;scorer.MODELS=MODELS;scorer.verify=verify;scorer.label_path=lambda c:common40.LABELS/('train3.json' if c=='train3' else 'train6.json');scorer.score()
    details=[];table=[]
    for case,cfg in CASES.items():
        rows=list(map(json.loads,(OUT/case/'observations.jsonl').read_text(encoding='utf-8').splitlines()));tt=np.array([r['t'] for r in rows])
        labels=[] if case=='negative5' else [r for r in load(scorer.label_path(case))['labels'] if cfg['start']<r['t']<=cfg['start']+cfg['duration']]
        for label in labels:
            i=int(np.searchsorted(tt,label['t'],side='right')-1);available=i>=0 and label['t']-tt[i]<=.025
            for method in METHODS:
                o=rows[i]['methods'][method]['observation'] if available else None;d=dict(case=case,method=method,t=label['t'],state=label['state'],detected=bool(o and o['detected']))
                if d['detected'] and label['state']=='present':
                    box=np.array(label['bbox']);err=float(np.linalg.norm(np.array(o['center'])-(box[:2]+box[2:])/2));d.update(error_px=err,joint_success=err<=.25*np.linalg.norm(box[2:]-box[:2]))
                details.append(d)
        for method in METHODS:
            pp=[d for d in details if d['case']==case and d['method']==method and d['state']=='present'];hit=[d for d in pp if d['detected']];aa=[d for d in details if d['case']==case and d['method']==method and d['state']=='absent']
            table.append(dict(case=case,method=method,present=len(pp),detected=len(hit),joint_success=sum(d.get('joint_success',False) for d in pp),mean_error_px=float(np.mean([d['error_px'] for d in hit])) if hit else None,absent=len(aa),false_positive_frames=sum(d['detected'] for d in aa),known_empty_published=sum(r['methods'][method]['observation']['detected'] for r in rows) if case=='negative5' else None))
    save(OUT/'current_localization.json',dict(table=table,details=details));print('All continuous current/future scores saved',flush=True)
if __name__=='__main__':main()
