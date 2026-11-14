"""Common-unit scene audit; labels are retrospective diagnostics only."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools')]
from compare_final_pipeline import reference_at,stats
from prepare_fred_transfer import parse
from score_fred_transfer import reference


def load(path):return [json.loads(x) for x in path.read_text('utf-8').splitlines()]


def audit(rows,box_at):
    values=[];last=None
    for r in rows:
        slot=int(np.floor(r['t']*5+1e-8))
        if not r['valid'] or slot==last:continue
        last=slot;t=r['t'];current=box_at(t);future=box_at(t+.4)
        if current is None or future is None:continue
        z=(current[:2]+current[2:])/2;target=(future[:2]+future[2:])/2
        p=np.asarray(r['coefficient']['position']);v=np.asarray(r['coefficient']['velocity'])
        pred=np.asarray(r['physical']['0.4']);diag=float(np.linalg.norm(future[2:]-future[:2]))
        error=float(np.linalg.norm(pred-target))
        value=dict(t=t,error_px=error,error_box_diagonals=error/max(diag,1e-9),future_box_diagonal_px=diag,
            true_displacement_04_px=float(np.linalg.norm(target-z)),source_position_error_px=float(np.linalg.norm(p-z)),
            exact_position_substitution_error_px=float(np.linalg.norm(pred+z-p-target)),
            current_state_CV_error_px=float(np.linalg.norm(p+.4*v-target)),
            force_input_current=r['force']['active'],acceleration_qualified=r['image_motion']['acceleration'].get('applied',False),
            mode=r['structure']['mode'])
        past=box_at(t-.2);half=box_at(t+.2)
        if past is not None:
            zp=(past[:2]+past[2:])/2;vp=(z-zp)/.2
            value['past_labeled_secant_CV_error_px']=float(np.linalg.norm(z+.4*vp-target))
        if half is not None:
            zh=(half[:2]+half[2:])/2
            value['future_half_displacement_change_px']=float(np.linalg.norm((target-zh)-(zh-z)))
        values.append(value)
    fields=['error_px','error_box_diagonals','future_box_diagonal_px','true_displacement_04_px',
        'source_position_error_px','exact_position_substitution_error_px','current_state_CV_error_px',
        'past_labeled_secant_CV_error_px','future_half_displacement_change_px']
    present=[]
    for r in rows:
        box=box_at(r['t'])
        if box is not None:
            present.append(dict(t=r['t'],detected=r['body'].get('detected',False),valid=r['valid']))
    return dict(statistics={k:stats([v[k] for v in values if k in v]) for k in fields},records=values,
        annotated_source_ticks=len(present),current_image_detections=sum(v['detected'] for v in present),
        valid_source_forecasts=sum(v['valid'] for v in present),
        qualified_force_inputs=sum(v['force_input_current'] for v in values),
        qualified_accelerations=sum(v['acceleration_qualified'] for v in values),
        no_independent_holdout_claim=True)


if __name__=='__main__':
    out=ROOT.parent/'output/stability_audit_20261031';out.mkdir(exist_ok=True)
    base=ROOT.parent/'output';result={}
    for name,path,label in [('train2','target25_continuity_20261031/RAW_other_center6s','dense_labels_train2_40_46'),
        ('side','reacquisition_motion_20261029/RAW_side_default8s','dense_labels_side_110_1195')]:
        labels=sorted(json.loads((base/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        def at(t):
            p=reference_at(labels,t)
            if p is None:return None
            # Interpolate sizes with the same <=150ms labelled gap gate.
            stamps=np.array([r['t'] for r in labels]);i=int(np.searchsorted(stamps,t));i=min(max(i,1),len(labels)-1)
            a,b=labels[i-1:i+1]
            if not a.get('box') or not b.get('box'):return None
            u=np.clip((t-a['t'])/(b['t']-a['t']),0,1)
            sizes=[np.asarray(r['box'][2:])-r['box'][:2] for r in (a,b)]
            size=(1-u)*sizes[0]+u*sizes[1]
            return np.r_[p-size/2,p+size/2]
        result[name]=audit(load(base/path/'predictions.jsonl'),at)
    for n,path in [(0,'target25_continuity_20261031/RAW_fred0_center4s'),
        (1,'fred_domain_adaptation_20261029/RAW_1_verified'),(2,'fred_transfer_20261029/RAW_2_candidate'),
        (7,'fred_domain_adaptation_20261029/RAW_7_baseline')]:
        p=base/path
        if n==7:
            protocol=json.loads((base/'fred_domain_adaptation_20261029/protocol.json').read_text())
            # Existing unseen7 input is kept separate from the adaptation train split.
            candidates=list((base/'fred_transfer_20261029/unseen7').rglob('coordinates.txt'))
            if not candidates: candidates=list((base/'fred_domain_adaptation_20261029').glob('input/7/coordinates.txt'))
            if not candidates:continue
            label=candidates[0]
        else:label=base/f'fred_transfer_20261029/input/{n}/coordinates.txt'
        labels=parse(label.read_text('utf-8-sig'))
        result[f'FRED{n}']=audit(load(p/'predictions.jsonl'),lambda t:reference(labels,t))
    (out/'scene_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:dict(mean_px=v['statistics']['error_px']['mean'],
        box_diagonals=v['statistics']['error_box_diagonals']['mean'],
        diagonal=v['statistics']['future_box_diagonal_px']['median'],
        displacement=v['statistics']['true_displacement_04_px']['median'],
        perfect_current_p_error=v['statistics']['exact_position_substitution_error_px']['mean'],
        prior_labeled_CV=v['statistics']['past_labeled_secant_CV_error_px']['mean'],
        force=v['qualified_force_inputs'],n=len(v['records'])) for k,v in result.items()},indent=2))
