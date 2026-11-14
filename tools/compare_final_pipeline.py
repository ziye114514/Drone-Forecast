"""Compare frozen end-to-end outputs, then open existing manual references.

Manual boxes are development labels, not new test data or physical rotor truth.
"""
from pathlib import Path
import json,sys,bisect,hashlib
import numpy as np


def stats(values):
    return dict(n=len(values),mean=float(np.mean(values)) if values else None,
        median=float(np.median(values)) if values else None,
        p95=float(np.percentile(values,95)) if values else None)


def max_difference(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b):raise AssertionError('Different output fields')
        return max((max_difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError('Different lengths')
        return max((max_difference(x,y) for x,y in zip(a,b)),default=0.)
    if isinstance(a,(int,float)) and not isinstance(a,bool):
        if not isinstance(b,(int,float)):raise AssertionError('Different types')
        if a is None or b is None:return 0. if a is b else float('inf')
        return abs(a-b)
    if a!=b:raise AssertionError((a,b))
    return 0.


def reference_at(records,t):
    stamps=[r['t'] for r in records];i=bisect.bisect_left(stamps,t)
    if i<len(records) and abs(stamps[i]-t)<1e-7:pair=[records[i],records[i]]
    elif i>0 and abs(stamps[i-1]-t)<1e-7:pair=[records[i-1],records[i-1]]
    elif 0<i<len(records) and stamps[i]-stamps[i-1]<=.1500001:pair=[records[i-1],records[i]]
    else:return None
    if any(r.get('state')!='present' or not r.get('box') for r in pair):return None
    centers=[(np.asarray(r['box'][:2])+r['box'][2:])/2 for r in pair]
    alpha=0. if pair[0]['t']==pair[1]['t'] else (t-pair[0]['t'])/(pair[1]['t']-pair[0]['t'])
    return (1-alpha)*centers[0]+alpha*centers[1]


def score(rows,truths):
    anchors=[];errors=[];ious=[];hits=0
    for truth in truths:
        if truth.get('state')!='present' or not truth.get('box'):continue
        row=min(rows,key=lambda r:abs(r['t']-truth['t']))
        if abs(row['t']-truth['t'])>1/60+.00001:continue
        body=row['body'];error=iou=None
        if body.get('detected'):
            hits+=1;a=np.asarray(body['bbox']);b=np.asarray(truth['box'])
            center=(b[:2]+b[2:])/2
            error=float(np.linalg.norm(np.asarray(body['center'])-center));errors.append(error)
            intersection=np.prod(np.maximum(0.,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
            union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
            iou=float(intersection/max(union,1e-9));ious.append(iou)
        anchors.append(dict(label_t=truth['t'],source_t=row['t'],detected=bool(body.get('detected')),error_px=error,iou=iou))
    future={str(h):[] for h in (.1,.2,.4)};last_slot=None
    for r in rows:
        slot=int(np.floor(r['t']*5+1e-8))
        if not r['valid'] or slot==last_slot:continue
        last_slot=slot
        for h in (.1,.2,.4):
            target=reference_at(truths,r['t']+h)
            if target is not None and str(h) in r['physical']:
                future[str(h)].append(float(np.linalg.norm(np.asarray(r['physical'][str(h)])-target)))
    return dict(manual_anchor_count=len(anchors),detected_anchors=hits,
        available_fraction=None if not anchors else hits/len(anchors),
        box_center_error_px=stats(errors),box_iou=stats(ious),
        physical_future_error_px={h:stats(v) for h,v in future.items()},anchors=anchors)


def compare(reference,optimized,output,labels=None):
    folders=dict(reference=Path(reference),optimized=Path(optimized));rows={};summaries={}
    for name,path in folders.items():
        rows[name]=[json.loads(s) for s in (path/'predictions.jsonl').read_text('utf-8').splitlines()]
        summaries[name]=json.loads((path/'summary.json').read_text('utf-8'))
    a,b=rows.values();assert len(a)==len(b)
    fields=['t','body','structure','geometry','rates','period','coefficient','valid','physical','force','center_bridge','tracking','motion',
        'relative_frequency_squared_proxy','period_selection','raw_events','frequency_input_events','max_event_t','nn_fresh']
    differences={k:0. for k in fields};unequal={k:0 for k in fields}
    for left,right in zip(a,b):
        for k in fields:
            try:d=max_difference(left[k],right[k])
            except AssertionError:d=float('inf')
            differences[k]=max(differences[k],d)
            if d!=0.:unequal[k]+=1
    protocols={name:json.loads((path/'protocol.json').read_text('utf-8')) for name,path in folders.items()}
    report=dict(runs={name:str(path.resolve()) for name,path in folders.items()},
        source_frames=len(a),differences=differences,unequal_frames=unequal,
        exact_all_compared_outputs=all(d==0 for d in differences.values()),
        summaries=summaries,startup={name:{k:p.get(k) for k in ['decoder_started_after_s','gpu_ready_after_s','first_source_ready_after_s','first_result_after_s']} for name,p in protocols.items()},
        timing_scope='Unprofiled full pipeline; startup excluded from steady capacity; paced full frame stream, no source drops. Short pilot on one computer.',
        precision_scope='Existing exposed development boxes, nearest source tick within16.68ms; present anchors only, not full-dataset recall/mAP. Future reference interpolates only <=150ms gaps at5Hz source slots.')
    old=summaries['reference']['steady_capacity_fps'];new=summaries['optimized']['steady_capacity_fps']
    report['steady_fps_change_percent']=100*(new/old-1)
    # Only now open annotations, after both causal inference outputs were frozen.
    if labels:
        truths=sorted(json.loads(Path(labels).read_text('utf-8'))['records'].values(),key=lambda z:z['t'])
        report['manual_scores']={name:score(r,truths) for name,r in rows.items()}
    Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=True),'utf-8')
    print(json.dumps(dict(exact=report['exact_all_compared_outputs'],frames=len(a),fps={k:v['steady_capacity_fps'] for k,v in summaries.items()},
        fps_change_percent=report['steady_fps_change_percent'],max_differences=differences,manual_scores={k:{key:val for key,val in value.items() if key!='anchors'} for k,value in report.get('manual_scores',{}).items()}),ensure_ascii=False))


if __name__=='__main__':compare(*sys.argv[1:])
