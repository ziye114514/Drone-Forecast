"""Score frozen FRED outputs, preserving misses and fixed forecast slots."""
from pathlib import Path
import bisect
import hashlib
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_fred_transfer import parse, OUT
from compare_final_pipeline import stats


def reference(rows, t):
    times = [r['t'] for r in rows]
    i = bisect.bisect_left(times, t)
    if i < len(rows) and abs(times[i]-t) < 1e-7:
        a = b = rows[i]
    elif 0 < i < len(rows) and times[i]-times[i-1] <= .134:
        a, b = rows[i-1:i+1]
    else:
        return None
    if a['identity'] != b['identity']:
        return None
    weight = 0. if a['t'] == b['t'] else (t-a['t'])/(b['t']-a['t'])
    return (1-weight)*np.asarray(a['bbox'])+weight*np.asarray(b['bbox'])


def score(rows, labels, start, duration):
    anchors = []
    for r in rows:
        box = reference(labels, r['t'])
        if box is None:
            continue
        detected = bool(r['body'].get('detected'))
        iou = error = None
        if detected:
            b = np.asarray(r['body']['bbox'])
            intersection = np.prod(np.maximum(0, np.minimum(b[2:],box[2:])-np.maximum(b[:2],box[:2])))
            union = np.prod(b[2:]-b[:2])+np.prod(box[2:]-box[:2])-intersection
            iou = float(intersection/max(union, 1e-9))
            error = float(np.linalg.norm(np.asarray(r['body']['center'])-(box[:2]+box[2:])/2))
        anchors.append(dict(t=r['t'], detected=detected, iou=iou, center_error_px=error,
            label_diagonal_px=float(np.linalg.norm(box[2:]-box[:2]))))
    # First valid output in each absolute 5Hz slot, as in the prior manual
    # protocol. Count slots without predictions alongside conditional error.
    groups = {}
    for r in rows:
        groups.setdefault(int(np.floor(r['t']*5+1e-8)), []).append(r)
    targets = []
    for slot, group in sorted(groups.items()):
        eligible = [r for r in group if r.get('valid') and '0.4' in r.get('physical', {})]
        chosen = eligible[0] if eligible else group[0]
        current = reference(labels, chosen['t'])
        future = reference(labels, chosen['t']+.4)
        if current is None or future is None:
            continue
        error = norm = None
        if eligible:
            center = (future[:2]+future[2:])/2
            error = float(np.linalg.norm(np.asarray(chosen['physical']['0.4'])-center))
            norm = error/max(float(np.linalg.norm(future[2:]-future[:2])),1e-9)
        targets.append(dict(slot=slot,t=chosen['t'], available=bool(eligible),error_px=error,
            normalized_error_box_diagonal=norm,target_center=((future[:2]+future[2:])/2).tolist(),
            prediction=chosen.get('physical',{}).get('0.4') if eligible else None))
    detected = [r for r in anchors if r['detected']]
    good = [r for r in targets if r['available']]
    return dict(annotation_source_ticks=len(anchors),current_outputs=len(detected),
        current_output_fraction=len(detected)/len(anchors) if anchors else None,
        IoU50_fraction_including_misses=sum(r['iou'] is not None and r['iou']>=.5 for r in anchors)/len(anchors) if anchors else None,
        current_center_error_px=stats([r['center_error_px'] for r in detected]),
        current_IoU=stats([r['iou'] for r in detected]),
        target_slots=len(targets),predicted_target_slots=len(good),
        forecast_coverage=len(good)/len(targets) if targets else None,
        error_04s_px=stats([r['error_px'] for r in good]),
        error_04s_box_diagonal=stats([r['normalized_error_box_diagonal'] for r in good]),
        within30_fraction_including_misses=sum(r['error_px'] is not None and r['error_px']<=30 for r in targets)/len(targets) if targets else None,
        anchors=anchors,targets=targets)


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    suffix=sys.argv[1] if len(sys.argv)>1 else 'candidate'
    if not suffix.replace('_','').isalnum():raise ValueError('Invalid run suffix')
    result={}
    for seq in protocol['sequences']:
        run=OUT/f"RAW_{seq['sequence']}_{suffix}"
        if not (run/'summary.json').exists():
            continue
        data=(run/'predictions.jsonl').read_bytes()
        rows=[json.loads(line) for line in data.splitlines()]
        # Predictions are already complete on disk. No labels enter inference.
        labels=parse(Path(seq['annotation']).read_text('utf-8-sig'))
        counts={r['t']:sum(x['t']==r['t'] for x in labels) for r in labels}
        labels=[r for r in labels if counts[r['t']]==1 and r['bbox'][2]>r['bbox'][0] and r['bbox'][3]>r['bbox'][1]]
        result[str(seq['sequence'])]=dict(predictions_sha256=hashlib.sha256(data).hexdigest(),
            score=score(rows,labels,seq['start'],seq['duration']),
            runtime=json.loads((run/'summary.json').read_text()))
    path=OUT/('scores.json' if suffix=='candidate' else f'scores_{suffix}.json')
    path.write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:{a:b for a,b in v['score'].items() if a not in ('anchors','targets')} for k,v in result.items()},indent=2))


if __name__=='__main__':main()
