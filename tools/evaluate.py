"""Score immutable forecasts on sparse independently annotated future event boxes.

Does NOT interpolate labels, initialize tracks from labels, or use future RPM.
Ground-truth times only choose historical forecast anchors for scoring.
"""
from pathlib import Path
import argparse,json,csv,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.motion import METHODS,extrapolate

def evaluate(run_dir,label_file):
    run_dir=Path(run_dir);info=json.loads(Path(label_file).read_text(encoding='utf-8'))
    traces=[json.loads(s) for s in (run_dir/'forecasts.jsonl').read_text().splitlines()]
    obs=[json.loads(s) for s in (run_dir/'observations.jsonl').read_text().splitlines()]
    times=np.array([s['t'] for s in traces]);scores=[];detection=[]
    present_states=next((s['states'] for s in traces if 'states' in s),{})
    methods=[name for name in METHODS if name in present_states]
    for label in info['labels']:
        t=label['t'];now=np.searchsorted(times,t,side='right')-1
        if now<0 or t>times[-1]+.025:continue
        current=traces[now];o=obs[now]
        dr=dict(sequence=info['sequence'],t=t,state=label['state'],detected=o['detected'],forecast_active=current['active'])
        if label['state']=='present':
            box=np.array(label['bbox']);truth=(box[:2]+box[2:])/2
            diag=np.linalg.norm(box[2:]-box[:2])
            dr['center_error_px']=float(np.linalg.norm(np.array(o['center'])-truth)) if o['detected'] else None
            if o['detected']:
                a=np.array(o['bbox']);inter=np.maximum(0,np.minimum(a[2:],box[2:])-np.maximum(a[:2],box[:2])).prod()
                dr['iou']=float(inter/max((a[2:]-a[:2]).prod()+(box[2:]-box[:2]).prod()-inter,1e-6))
            else:dr['iou']=0.
            for horizon in (.1,.2,.4):
                k=np.searchsorted(times,t-horizon,side='right')-1
                if k<0:continue
                anchor=traces[k]
                row=dict(sequence=info['sequence'],target_time_s=t,anchor_time_s=float(times[k]),requested_horizon_s=horizon,
                         actual_horizon_s=float(t-times[k]),available=anchor['active'],gt_x=float(truth[0]),gt_y=float(truth[1]),gt_diagonal_px=float(diag))
                for name in methods:
                    r=dict(row,method=name)
                    if anchor['active']:
                        p=extrapolate(anchor['states'][name],t-times[k])
                        r.update(pred_x=float(p[0]),pred_y=float(p[1]),error_px=float(np.linalg.norm(p-truth)),normalized_error=float(np.linalg.norm(p-truth)/max(diag,1)))
                    scores.append(r)
        detection.append(dr)
    summary=[]
    for horizon in (.1,.2,.4):
        for name in methods:
            rows=[r for r in scores if r['method']==name and r['requested_horizon_s']==horizon]
            ok=[r for r in rows if r['available']]
            errors=[r['error_px'] for r in ok]
            summary.append(dict(sequence=info['sequence'],method=name,horizon_s=horizon,targets=len(rows),available=len(ok),
                coverage=len(ok)/max(len(rows),1),mean_error_px=float(np.mean(errors)) if errors else None,
                median_error_px=float(np.median(errors)) if errors else None,p90_error_px=float(np.percentile(errors,90)) if errors else None,
                mean_normalized_error=float(np.mean([r['normalized_error'] for r in ok])) if ok else None))
    positives=[r for r in detection if r['state']=='present'];negatives=[r for r in detection if r['state']=='absent']
    detection_summary=dict(positive_labels=len(positives),negative_labels=len(negatives),
        detector_presence_recall=sum(r['detected'] for r in positives)/max(len(positives),1),
        recall_iou_0_3=sum(r.get('iou',0)>=.3 for r in positives)/max(len(positives),1),
        false_positive_rate=sum(r['detected'] for r in negatives)/max(len(negatives),1),
        median_center_error_px=float(np.median([r['center_error_px'] for r in positives if r.get('center_error_px') is not None])) if any(r.get('center_error_px') is not None for r in positives) else None)
    run_manifest=json.loads((run_dir/'run_manifest.json').read_text())
    role=('Engineering regression on previously examined recordings; not a new unseen holdout.'
          if run_manifest.get('version')=='0.2.0' else info['role'])
    result=dict(source_labels=info['source'],label_sha256=info['source_sha256'],evaluation_role=role,
        original_annotation_role=info['role'],
        metrics=summary,detection=detection_summary,
        protocol='Only actual human future timestamps scored; no GT interpolation; past states only; actual horizons in [requested, requested+step). Metrics are future endpoint errors, not dense ADE.')
    (run_dir/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for filename,rows in [('evaluation_details.csv',scores),('detection_details.csv',detection),('metrics.csv',summary)]:
        if not rows:continue
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (run_dir/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(sequence=info['sequence'],detection=detection_summary,
                          adaptive_metrics=[r for r in summary if r['method']=='adaptive']),ensure_ascii=False))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--labels',required=True)
    a=p.parse_args();evaluate(a.run,a.labels)
