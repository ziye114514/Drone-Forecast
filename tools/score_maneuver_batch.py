"""Score frozen outputs and describe measured future path changes.

Future/manual data are evaluation only. 'Future changed' does not prove no
earlier attitude/event cue existed, and is not an unpredictable-command label.
"""
from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from compare_final_pipeline import reference_at,score,stats
from diagnose_maneuver_horizon import SPECS,OUT
from score_fred_adaptation import matched
from probe_same_window_motion import replay
OPTIONS=dict(curved=True,evidence_gain=True,memory=True,endpoint=True,memory_gaps=True,weak_support=True)


def load_rows(path):return [json.loads(x) for x in path.read_text().splitlines()]


def revisions(rows):
    result=[]
    for old,new in zip(rows,rows[1:]):
        dt=new['t']-old['t']
        if not old['valid'] or not new['valid'] or not 0.<dt<=.08 or not old['physical'] or not new['physical']:continue
        h=.4-dt;x=h/.2
        # Interpolate the existing one-rollout predictions at a common
        # absolute target; not an additional kinematic prediction branch.
        value=.5*(x-1)*(x-2)*np.array(new['coefficient']['position'])-x*(x-2)*np.array(new['physical']['0.2'])+.5*x*(x-1)*np.array(new['physical']['0.4'])
        result.append(dict(t=new['t'],revision_px=float(np.linalg.norm(value-old['physical']['0.4']))))
    return dict(statistics=stats([r['revision_px'] for r in result]),records=result,
        scope='Prediction revisions include legitimate new information; lower is not automatically more accurate')


def main():
    result={};deadline='--deadline' in sys.argv
    for name,path,label in SPECS:
        original=load_rows(ROOT.parent/'output'/path)
        control=load_rows(OUT/(name+'_control.jsonl'))
        candidate,_=replay(original,False,position_history_rebase='stable-axis',**OPTIONS)
        (OUT/(name+'_stable_side_rebase.jsonl')).write_text('\n'.join(json.dumps(r,allow_nan=False) for r in candidate)+'\n','utf-8')
        labels=sorted(json.loads((ROOT.parent/'output'/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        comparison=matched(control,candidate,lambda t:reference_at(labels,t));changes=[]
        for point in comparison['records']:
            t=point['t'];r=min(control,key=lambda r:abs(r['t']-t));s=min(candidate,key=lambda r:abs(r['t']-t))
            z=[reference_at(labels,t+h) for h in (-.2,0.,.2,.4)]
            if all(x is not None for x in z):
                before,early,late=[(z[i+1]-z[i])/.2 for i in range(3)]
                dv=late-early
                # Shared middle endpoint has coefficient-2: covariance6*sigma²/dt².
                chi=float(dv@dv/(6*6.**2/.2**2))
                point.update(observed_velocity_secants_px_s=dict(before=before.tolist(),early=early.tolist(),late=late.tolist()),
                    future_half_velocity_change_chi2_proxy=chi,future_half_change_detected=chi>5.991464547107979)
            a=r['image_motion']['acceleration'];slope=a.get('shrunk_acceleration')
            point.update(source_acceleration_gate=a['reason'],source_slope_candidate=slope,
                source_force_active=r['force']['active'],source_mode=r['structure']['mode'],
                revised_acceleration_gate=s['image_motion']['acceleration']['reason'],
                old_current_acceleration=r['force']['acceleration_px_s2'],new_current_acceleration=s['force']['acceleration_px_s2'])
            changes.append(point)
        result[name]=dict(comparison=comparison,origins=changes,control=score(control,labels),candidate=score(candidate,labels),
            body_and_source_validity_unchanged=all(a['body']==b['body'] and a['valid']==b['valid'] for a,b in zip(control,candidate)),
            rebased_ticks=sum(bool((r['image_motion'].get('position_history_gauge') or {}).get('applied')) for r in candidate),
            revisions={k:revisions(v) for k,v in [('control',control),('candidate',candidate)]})
        run=OUT/(('RAW_side_deadline_rebase8s' if name=='side' else 'RAW_other_deadline_rebase6s') if deadline else
            ('RAW_side_axis_rebase8s' if name=='side' else 'RAW_other_axis_rebase6s'))
        if (run/'summary.json').exists():
            actual=load_rows(run/'predictions.jsonl');base,_=replay(actual,False,**OPTIONS)
            changed,_=replay(actual,False,position_history_rebase='stable-axis',**OPTIONS)
            difference=max(float(np.max(abs(np.asarray(a['physical'][h])-b['physical'][h]))) for a,b in zip(actual,changed) for h in a['physical'])
            assert difference<1e-7,(name,'RAW candidate not reproduced',difference)
            result[name]['actual_RAW']=dict(actual_score=score(actual,labels),runtime=json.loads((run/'summary.json').read_text()),
                matched_original_origins=matched(control,actual,lambda t:reference_at(labels,t)),
                same_new_observations=matched(base,changed,lambda t:reference_at(labels,t)),
                replay_max_difference=difference)
    (OUT/('scores_deadline.json' if deadline else 'scores.json')).write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:dict(frozen_same_origin=v['comparison']['new_matched_error'],
        RAW_same_origin=v.get('actual_RAW',{}).get('matched_original_origins',{}).get('new_matched_error'),
        RAW_same_observations=v.get('actual_RAW',{}).get('same_new_observations',{}).get('new_matched_error')) for k,v in result.items()},indent=2))


if __name__=='__main__':main()
