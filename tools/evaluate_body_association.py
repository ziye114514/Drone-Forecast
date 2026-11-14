"""Same-NN-frame causal controls for association and fixed-time publication."""
from pathlib import Path
import bisect,json,sys,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_motion import LiveMotion,predictions
from summarize_live_pilot import reference_at,stats,read_rows

OUT=ROOT.parent/'output/live_body_association_v2_20261025';result={};checks={}
for name,duration in [('train2',6.),('side',4.),('clear',2.),('negative',1.)]:
    run=OUT/name;rr=read_rows(run/'tracking.jsonl');pp=read_rows(run/'predictions.jsonl')
    protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    models={key:LiveMotion(25.) for key in ['high_only','associated']};states={key:{} for key in models}
    counts=dict(high_detected=0,associated_detected=0,weak_selected=0,weak_only_recovery=0,high_suppressed=0)
    exact=0.
    for row in rr:
        high=row['detector_boxes'];body=dict(t=row['t'],detected=False,center=None)
        if high:
            b=max(high,key=lambda q:q['confidence']);box=b['bbox']
            body=dict(t=row['t'],detected=True,bbox=box,confidence=b['confidence'],center=[(box[0]+box[2])/2,(box[1]+box[3])/2])
        counts['high_detected']+=body['detected'];counts['associated_detected']+=row['body']['detected']
        counts['weak_selected']+=row['body'].get('low_confidence',False)
        counts['weak_only_recovery']+=row['body']['detected'] and not body['detected']
        counts['high_suppressed']+=body['detected'] and not row['body']['detected']
        assert row['high_score_invariant']
        for key,obs in [('high_only',body),('associated',row['body'])]:
            c,v,d=models[key].step(row['t'],obs)
            states[key][round(row['t'],6)]=dict(c=c,valid=v,seen=models[key].seen)
            if key=='associated' and c is not None:
                for field in ('position','velocity','acceleration'):
                    exact=max(exact,float(np.max(abs(np.asarray(c[field])-row['coefficient'][field]))))
    assert exact<1e-10
    coverage={k:0 for k in models};tables={};records=None
    if name in ('train2','side'):
        label=ROOT.parent/'output'/('dense_labels_side_110_1195' if name=='side' else 'dense_labels_train2_40_46')/'labels.json'
        records=sorted(json.loads(label.read_text(encoding='utf-8'))['records'].values(),key=lambda q:q['t'])
    controls=[]
    for row in pp:
        if row['observation_t'] is None:controls.append(dict(t=row['t'],valid={k:False for k in models}));continue
        tick=round(row['observation_t'],6);v={};pred={}
        for key in models:
            s=states[key][tick]
            v[key]=bool(s['valid'] and row['t']-row['observation_t']<=.16+1e-8 and s['seen'] is not None and row['t']-s['seen']<=.16+1e-8)
            coverage[key]+=int(v[key]);pred[key]=predictions(s['c'],row['observation_age_ms']/1000.)[0]
        assert v['associated']==row['valid']
        controls.append(dict(t=row['t'],observation_t=row['observation_t'],valid=v,predictions=pred))
    if records is not None:
        for h in (.1,.2,.4):
            errors={k:[] for k in models};extra={k:[] for k in models};missing=0
            for row in controls:
                ref=reference_at(records,row['t']+h)
                if ref is None:missing+=1;continue
                common=all(row['valid'].values())
                for key in models:
                    if row['valid'][key]:
                        error=float(np.linalg.norm(np.asarray(row['predictions'][key][str(h)])-ref))
                        (errors if common else extra)[key].append(error)
            tables[str(h)]=dict(common={k:stats(v) for k,v in errors.items()},unshared={k:stats(v) for k,v in extra.items()},missing_reference=missing)
    known_present=0;known_visible_valid={k:0 for k in models}
    if records is not None:
        for row in controls:
            if reference_at(records,row['t']) is not None:
                known_present+=1
                for k in models:known_visible_valid[k]+=int(row['valid'][k])
    completion=json.loads((run/'completion.json').read_text(encoding='utf-8'))
    assert len(pp)==round(duration*5) and completion['independent_missed_slots']==0
    assert all(row['observation_t'] is None or row['observation_t']<=row['estimate_available_t']<=row['t']+1e-9 for row in pp)
    assert all(row['max_event_t'] is None or row['max_event_t']<round(row['t']*1e6) for row in rr)
    assert len(set(row['publication_slot'] for row in pp))==len(pp)
    if name=='negative':assert not any(coverage.values())
    for relative,digest in protocol['sources'].items():
        data=(ROOT/relative).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
        dest=OUT/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    result[name]=dict(counts=counts,publication_slots=len(pp),valid_same_slots=coverage,
                      known_present_slots=known_present,valid_on_known_present=known_visible_valid,
                      interpolated_future_scores=tables,completion=completion,replay_max_difference=exact)
    checks[name]=dict(high_box_invariant=True,current_measurements_only=True,causal_available_time=True,
                      cutoff=True,fixed_deadlines=True,expected_slots=len(pp),baseline_candidates_same_frame=True)
report=dict(results=result,checks=checks,scope='Observation replay of both controls on the same RAW NN candidates and EXACT same publication slots; not a new end-to-end run for high-only.',
            reference='Interpolated adjacent present manual boxes <=150ms; unlabeled periods are UNKNOWN, not false negatives/absence truth.')
(OUT/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
