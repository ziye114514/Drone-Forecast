"""One delayed-feedback mechanism; event order uses estimate availability."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from live_gain import HorizonGain
from summarize_live_pilot import read_rows,reference_at,stats
OUT=ROOT.parent/'output/live_horizon_gain_20261025';OUT.mkdir(exist_ok=False)
report={}
for name in ['side','train2','clear','negative']:
    parent=ROOT.parent/'output/live_body_association_v2_20261025'/name
    tracking=read_rows(parent/'tracking.jsonl');pubs=read_rows(parent/'predictions.jsonl')
    m=LiveMotion(25.);cal=HorizonGain(5.);states={};saved=[]
    events=sorted([(r['estimate_available_t'],0,r) for r in tracking]+[(r['t'],1,r) for r in pubs],key=lambda e:(e[0],e[1]))
    for available,kind,row in events:
        if kind==0:
            b=dict(t=row['t'],detected=False,center=None)
            if row['detector_boxes']:
                q=max(row['detector_boxes'],key=lambda q:q['confidence']);box=q['bbox']
                b=dict(t=row['t'],detected=True,center=[(box[0]+box[2])/2,(box[1]+box[3])/2],bbox=box,confidence=q['confidence'])
            c,v,d=m.step(row['t'],b);states[round(row['t'],6)]=dict(c=c,valid=v,seen=m.seen)
            cal.observe(row['t'],b['center'],available,strong=b['detected'])
        else:
            if row['observation_t'] is None:continue
            s=states[round(row['observation_t'],6)];age=row['t']-row['observation_t']
            valid=s['valid'] and age<=.16+1e-8 and s['seen'] is not None and row['t']-s['seen']<=.16+1e-8
            base=predictions(s['c'],age)[0];new,audit=cal.forecast(row['t'],s['c'],age,valid)
            assert audit.get('feedback_available_t') is None or audit['feedback_available_t']<=row['t']+1e-9
            saved.append(dict(t=row['t'],valid=valid,baseline=base,calibrated=new,audit=audit))
    table={}
    if name in ('side','train2'):
        label=ROOT.parent/'output'/('dense_labels_side_110_1195' if name=='side' else 'dense_labels_train2_40_46')/'labels.json'
        records=sorted(json.loads(label.read_text(encoding='utf-8'))['records'].values(),key=lambda q:q['t'])
        for h in (.1,.2,.4):
            errors={k:[] for k in ('baseline','calibrated')};changes=0;missing=0
            for row in saved:
                if not row['valid']:continue
                ref=reference_at(records,row['t']+h)
                if ref is None:missing+=1;continue
                for k in errors:errors[k].append(float(np.linalg.norm(np.asarray(row[k][str(h)])-ref)))
                changes+=not np.array_equal(row['baseline'][str(h)],row['calibrated'][str(h)])
            table[str(h)]=dict(metrics={k:stats(v) for k,v in errors.items()},changed=changes,missing=missing)
    report[name]=dict(scores=table,active_publications=sum(r['audit']['active'] for r in saved),publications=len(saved),
                      last_parameters=saved[-1]['audit'] if saved else None)
    (OUT/(name+'_replay.jsonl')).write_text('\n'.join(json.dumps(r) for r in saved)+'\n',encoding='utf-8')
(OUT/'comparison.json').write_text(json.dumps(dict(scope='Frozen observation replay only, ordered by actual estimate availability; no manual labels used by calibrator',results=report),indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
