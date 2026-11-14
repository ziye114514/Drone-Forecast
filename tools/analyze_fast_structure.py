"""Read predictions first, then sparse development references; never feeds runner."""
from pathlib import Path
import json,sys
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from summarize_live_pilot import reference_at,stats
from fast_force_response import FastForceResponse
from live_structure_bridge import GeometryMotion
from live_scale_rollout import scale_predictions


def analyze(run,labels):
    run=Path(run);rows=[json.loads(s) for s in (run/'predictions.jsonl').read_text('utf-8').splitlines()]
    # User removed traditional prediction from real time. Reconstruct its
    # matched p/v control HERE, after the completed run, with no extra RAW/GPU.
    offline=GeometryMotion(30.)
    force_control=FastForceResponse(early_disturbance=False,hold_acceleration_gaps=False,gain_signal_to_noise=0.)
    for r in rows:
        observed=r['body'].get('center') if r['body'].get('detected') else None
        r['physical_8row_control'],_=force_control.step(r['t'],r['coefficient'],observed,r['force']['input_u'],.35)
        b=dict(r['body']);diag=r['center_bridge']
        if diag.get('applied'):b['center']=diag['fused_position']
        offline.kf.noise_scale=diag.get('noise_scale',1.)
        c,_,_=offline.step(r['t'],b)
        if not r['baseline'] and c is not None:
            # The lightweight physical KF intentionally omits the legacy weak
            # confidence weighting. Hold ACTUAL online p/v fixed; only compare
            # acceleration/scale forecasts, not a silently different filter.
            c.update(position=r['coefficient']['position'],velocity=r['coefficient']['velocity'])
            r['baseline'],r['cv']=scale_predictions(c)
    # Complete causal output already exists before references are opened.
    truths=sorted(json.loads(Path(labels).read_text('utf-8'))['records'].values(),key=lambda z:z['t'])
    errors={str(H):{name:[] for name in ['cv','kinematic','physical','physical_8row_control','no_force']} for H in (.1,.2,.4)}
    samples=[];last_slot=None;centers=[]
    for r in rows:
        truth=reference_at(truths,r['t'])
        if truth is not None and r['body'].get('detected'):centers.append(float(np.linalg.norm(np.array(r['body']['center'])-truth)))
        slot=int(np.floor(r['t']*5+1e-8))
        if not r['valid'] or slot==last_slot:continue
        last_slot=slot;c=r['coefficient'];p,v,a=[np.asarray(c[k]) for k in ['position','velocity','acceleration']]
        for H in (.1,.2,.4):
            truth=reference_at(truths,r['t']+H)
            if truth is None:continue
            h=str(H);points=dict(cv=r['cv'][h],kinematic=r['baseline'][h],physical=r['physical'][h],
                physical_8row_control=r['physical_8row_control'][h],no_force=p+H*v+.5*H*H*a)
            e={name:float(np.linalg.norm(np.asarray(point)-truth)) for name,point in points.items()}
            for name,value in e.items():errors[h][name].append(value)
            samples.append(dict(t=r['t'],target=r['t']+H,horizon=H,reference=truth.tolist(),errors=e,force_active=r['force']['active']))
    result=dict(reference_scope='Previously exposed sparse development bbox centres; interpolation only across<=150ms; not COM/RPM/pose truth',
        labels_read_after_predictions=True,source_samples=len(rows),source_fps=30.,score_slots_hz=5.,
        whole_body_center=stats(centers),scores={h:{name:stats(values) for name,values in names.items()} for h,names in errors.items()},
        baseline_scope='Offline reconstruction on same new observations and p/v; traditional forecast was NOT run by physical-only entry. NOT full v7 optical-flow pipeline',
        physical_scope='Default/online effective-gain plus lumped image disturbance; no calibrated force truth',
        turn_onset_independently_annotated=False,samples=samples)
    (run/'accuracy.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({k:result[k] for k in ['source_samples','whole_body_center','scores']},ensure_ascii=False))


if __name__=='__main__':analyze(sys.argv[1],sys.argv[2])
