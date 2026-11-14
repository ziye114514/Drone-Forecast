"""Fixed causal disturbance-startup mechanisms; reuse saved online inputs."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from fast_force_response import FastForceResponse
from summarize_live_pilot import reference_at,stats


def synthetic():
    arms={name:FastForceResponse(early_disturbance=name!='legacy',hold_acceleration_gaps=False,gain_signal_to_noise=0.) for name in ['legacy','early']}
    a=np.array([80.,-30.]);u=np.array([0.,-1.]);rows=[]
    for i in range(27):
        t=i/30;p=.5*t*t*a;v=t*a
        c=dict(position=p.tolist(),velocity=v.tolist(),acceleration=[0.,0.])
        d={n:m.step(t,c,p,u,.8)[1] for n,m in arms.items()}
        rows.append(dict(t=t,arms=d))
    error={n:float(np.linalg.norm(np.array(rows[-1]['arms'][n]['acceleration_px_s2'])-a)) for n in arms}
    assert error['early']<error['legacy'] and not any(r['arms']['early']['gain_learned'] for r in rows)
    first={n:next((r['t'] for r in rows if r['arms'][n]['disturbance_updated']),None) for n in arms}
    assert first['early']>=.2 and first['legacy'] is None
    return dict(constant_input_and_acceleration=True,early_disturbance_does_not_identify_gain=True,
        acceleration_error_at_0867s_px_s2=error,first_disturbance_update_s=first)


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    sources=dict(side=ROOT.parent/'output/continuous_five_circle_20261026/raw_side',
        clear=ROOT.parent/'output/three_continuation_20261026/raw_normal')
    observations={};replay_error=[]
    for name,path in sources.items():
        names=['legacy','early','aligned','hold','gain_quality']
        models={n:FastForceResponse(early_disturbance=n!='legacy',hold_acceleration_gaps=n in ('hold','gain_quality'),
            gain_signal_to_noise=2. if n=='gain_quality' else 0.) for n in names}
        result=[]
        for r in map(json.loads,(path/'predictions.jsonl').read_text('utf-8').splitlines()):
            arms={};b=r['body'];position=b.get('center') if b.get('detected') else None
            for n,m in models.items():
                observed=position
                if n in ('aligned','hold','gain_quality') and r['center_bridge'].get('applied'):observed=r['center_bridge']['fused_position']
                pred,diag=m.step(r['t'],r['coefficient'],observed,r['force']['input_u'],.35)
                arms[n]=dict(prediction=pred,force=diag)
            if r['physical']:
                replay_error.extend(abs(np.asarray(arms['legacy']['prediction'][str(h)])-r['physical'][str(h)]).max() for h in (.1,.2,.4))
            result.append(dict(t=r['t'],valid=r['valid'],arms=arms))
        observations[name]=result
    assert max(replay_error,default=0)<1e-9,max(replay_error)
    # Persist every arm before opening the external development references.
    (out/'causal_outputs.json').write_text(json.dumps(observations),'utf-8')
    control=synthetic()
    labels=ROOT.parent/'output/dense_labels_side_110_1195/labels.json'
    truth=sorted(json.loads(labels.read_text('utf-8'))['records'].values(),key=lambda x:x['t'])
    errors={str(h):{n:[] for n in names} for h in (.1,.2,.4)}
    last=None
    for r in observations['side']:
        slot=int(np.floor(r['t']*5+1e-8))
        if not r['valid'] or slot==last:continue
        last=slot
        for h in (.1,.2,.4):
            ref=reference_at(truth,r['t']+h)
            if ref is None:continue
            for n,z in r['arms'].items():errors[str(h)][n].append(float(np.linalg.norm(z['prediction'][str(h)]-ref)))
    summary=dict(synthetic=control,legacy_replay_max_error_px=max(replay_error),
        same_saved_online_inputs=True,predictions_saved_before_labels=True,RAW_or_GPU_repeated=False,
        side_scores={h:{n:stats(v) for n,v in arm.items()} for h,arm in errors.items()},
        first_corrections={name:{n:next((r['t'] for r in rows if r['arms'][n]['force']['disturbance_updated']),None)
            for n in names} for name,rows in observations.items()},
        final_gains={name:{n:rows[-1]['arms'][n]['force']['effective_gain_px_s2'] for n in names} for name,rows in observations.items()},
        short_gap_hold_counts={name:sum(r['arms']['hold']['force']['short_gap_acceleration_held'] for r in rows) for name,rows in observations.items()},
        limitations='Previously exposed sparse side bbox labels only. Clear has no independent matched labels. No RPM/thrust/wind truth.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8');print(json.dumps(summary))


if __name__=='__main__':run(sys.argv[1])
