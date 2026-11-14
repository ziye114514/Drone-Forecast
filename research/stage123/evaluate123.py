"""Post-run scoring of frozen natural extension; sparse BOX proxies only."""
from pathlib import Path
import collections,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage123')]
from study123 import OUT,LABELS,CASES,METHODS,verify,rows,save,sha

def stretches(rr,predicate):
    groups=[];group=[]
    for r in rr:
        if predicate(r):group.append(r)
        elif group:groups.append(group);group=[]
    if group:groups.append(group)
    return groups

def state(r):return r['frontends']['delayed_feedback']['learned_structure118']

def availability(case,rr):
    ww=[state(r) for r in rr];counts=collections.Counter(len(w['input_rotors']) for w in ww);summary=[]
    for mode in ['body_hold','affine3','conic3','guarded3','validated122']:
        dd=[w['validated122']['center'] if mode=='validated122' else w['outputs'][mode]['center'] for w in ww]
        summary.append(dict(mode=mode,current=sum(d['current'] for d in dd),full_current=sum(d['current'] and len(w['input_rotors'])==4 for d,w in zip(dd,ww)),
            partial3_current=sum(d['current'] and len(w['input_rotors'])==3 for d,w in zip(dd,ww)),
            first_structure=next((r['t'] for r,d in zip(rr,dd) if d.get('structure_ready')),None),
            reasons=dict(collections.Counter(d['measurement']['reason'] for d in dd))))
    transitions=[]
    for group in stretches(rr,lambda r:len(state(r)['input_rotors'])==3 and state(r)['validated122']['center']['current']):
        if len(group)<5:continue
        before=[r for r in rr if r['t']<group[0]['t'] and len(state(r)['input_rotors'])==4 and state(r)['validated122']['center']['current']]
        after=[r for r in rr if r['t']>group[-1]['t'] and len(state(r)['input_rotors'])==4 and state(r)['validated122']['center']['current']]
        if not before or not after:continue
        # Same continuous body identity: do not bridge an unsupported>.5s gap.
        span=[r for r in rr if before[-1]['t']<=r['t']<=after[0]['t']];last=before[-1]['t'];identity=True
        for r in span:
            if r['body_observation'].get('detected'):
                if r['t']-last>.5:identity=False
                last=r['t']
        if identity:transitions.append(dict(last_full=before[-1]['t'],partial_begin=group[0]['t'],partial_end=group[-1]['t'],partial_frames=len(group),partial_duration_s=len(group)*.02,recovered=after[0]['t']))
    choices=collections.Counter(w['validated122']['selection']['chosen'] for w in ww if len(w['input_rotors'])==3)
    return dict(case=case,frames=len(rr),body_current=sum(r['body_observation'].get('detected',False) for r in rr),valid_forecast=sum(r['frontends']['delayed_feedback']['valid'] for r in rr),
        rotor_count_histogram={str(k):v for k,v in sorted(counts.items())},methods=summary,
        natural_partial_choices=dict(choices),validation_updates=sum(w['validated122']['selection']['validation'] is not None for w in ww),
        complete_natural_transitions=transitions,n_complete_natural_transitions=len(transitions),
        no_human_visibility_labels=True,detected_rotor_count_is_not_visibility_truth=True)

def score(case,rr,labeldata):
    cfg=CASES[case];labels=[l for l in labeldata['labels'] if cfg['start']<l['t']<=cfg['start']+cfg['duration']]
    times=np.array([r['t'] for r in rr]);present=[l for l in labels if l['state']=='present' and l.get('bbox')];table=[];pairs=[]
    for h in [.1,.2,.4]:
        eligible=[l for l in present if l['t']>cfg['start']+h];errors={m:[] for m in METHODS};actual=[];missing=collections.Counter()
        for label in eligible:
            i=int(np.searchsorted(times,label['t']-h+1e-10,side='right')-1)
            if i<0:missing['no_origin']+=1;continue
            r=rr[i];lag=label['t']-h-r['t'];q=r['frontends']['delayed_feedback']
            if lag< -1e-8 or lag>.02000001:missing['origin_time_gap']+=1;continue
            if not q['valid']:missing['no_valid_forecast']+=1;continue
            if not all(m in q['coefficients'] for m in METHODS):missing['one_or_more_method_unavailable']+=1;continue
            truth=(np.array(label['bbox'][:2])+label['bbox'][2:])/2;dt=label['t']-r['t'];actual.append(dt);predictions={}
            for method in METHODS:
                c=q['coefficients'][method];assert c['acceleration']==[0.,0.]
                prediction=np.array(c['position'])+dt*np.array(c['velocity']);err=float(np.linalg.norm(prediction-truth));errors[method].append(err)
                predictions[method]=dict(prediction=prediction.tolist(),error_px=err)
            pairs.append(dict(case=case,h=h,target_t=label['t'],origin_t=r['t'],actual_horizon_s=dt,bbox=label['bbox'],target_bbox_center=truth.tolist(),predictions=predictions))
        for method in METHODS:
            e=errors[method];table.append(dict(case=case,h=h,method=method,eligible_present_targets=len(eligible),paired_targets=len(e),coverage=len(e)/max(len(eligible),1),
                mean_px=float(np.mean(e)) if e else None,median_px=float(np.median(e)) if e else None,
                rmse_px=float(np.sqrt(np.mean(np.square(e)))) if e else None,maximum_px=max(e) if e else None,
                actual_horizon_min=min(actual) if actual else None,actual_horizon_max=max(actual) if actual else None,missing=dict(missing)))
    absent=[]
    for l in labels:
        if l['state']!='absent':continue
        i=int(np.searchsorted(times,l['t']+1e-10,side='right')-1)
        if i>=0 and 0<=l['t']-rr[i]['t']<=.02000001:absent.append(dict(t=l['t'],origin_t=rr[i]['t'],valid=rr[i]['frontends']['delayed_feedback']['valid']))
    current=[]
    for l in present:
        i=int(np.searchsorted(times,l['t']+1e-10,side='right')-1)
        if i<0 or l['t']-rr[i]['t']>.02000001:continue
        b=rr[i]['body_observation'];iou=0.;size=np.array(l['bbox'][2:])-l['bbox'][:2]
        if b.get('detected') and b.get('bbox') is not None:
            x=np.array(b['bbox']);y=np.array(l['bbox']);overlap=np.maximum(0.,np.minimum(x[2:],y[2:])-np.maximum(x[:2],y[:2]));area=float(np.prod(overlap));union=float(np.prod(x[2:]-x[:2])+np.prod(y[2:]-y[:2])-area);iou=area/max(union,1e-12)
        current.append(dict(t=l['t'],NN_detected=bool(b.get('detected')),bbox_iou=iou,target_width=float(size[0]),target_height=float(size[1]),geometry_current=state(rr[i])['validated122']['center']['current']))
    return table,pairs,dict(case=case,labels_present=len(present),absent_checked=len(absent),valid_at_absent=sum(x['valid'] for x in absent),absent_samples=absent,
        present_checked=len(current),NN_detected_at_present=sum(x['NN_detected'] for x in current),geometry_current_at_present=sum(x['geometry_current'] for x in current),
        median_bbox_iou_including_misses=float(np.median([x['bbox_iou'] for x in current])) if current else None,
        median_target_width_height=np.median([[x['target_width'],x['target_height']] for x in current],axis=0).tolist() if current else None,present_samples=current,
        label_kind=labeldata.get('label_kind'),label_source=labeldata.get('source'),label_caveat=labeldata.get('caveat'))

def main():
    protocol=verify();availability_rows=[];scores=[];pairs=[];labelchecks=[]
    # Every inference job must be complete before parsing even one target.
    for case in CASES:assert (OUT/'raw'/case/'checks123.json').is_file(),case
    for case,cfg in CASES.items():
        if case.startswith('prefix'):continue
        rr=rows(OUT/'raw'/case/'observations_forecasts.jsonl');availability_rows.append(availability(case,rr))
        if cfg['labels']:
            path=LABELS/cfg['labels'];assert sha(path)==protocol['label_sha256'][cfg['labels']]
            tab,pair,checks=score(case,rr,json.loads(path.read_text('utf-8-sig')));scores.extend(tab);pairs.extend(pair);labelchecks.append(checks)
    save(OUT/'availability123.json',availability_rows);save(OUT/'scores123.json',dict(table=scores,pairs=pairs,label_checks=labelchecks,
        same_eligible_origins_all_methods=True,targets='Existing sparse manual event-image bbox centers, not COM; no interpolation of GT.',
        real_motion_rpm_attitude_ground_truth=False,predictions_before_labels=True,no_parameter_changes=True))
    print(json.dumps(dict(availability=[{k:v for k,v in r.items() if k!='methods'} for r in availability_rows],score04=[r for r in scores if r['h']==.4])),flush=True)

if __name__=='__main__':main()
