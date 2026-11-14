"""Cheap frozen-fit diagnosis of polarity cancellation, no model search."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from pure_visibility170 import window
from replay_joint236 import source
from polarity_phase_validation import validate

if __name__=='__main__':
    OUT=ROOT.parent/'output/polarity295_20261109';OUT.mkdir(exist_ok=True)
    assert not (OUT/'results.json').exists(),'Completed evidence must not be overwritten'
    results={}
    for n in ('13','15','16'):
        old=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
        periods=[json.loads(s) for s in (ROOT.parent/f'output/joint242_20261108/recording{n}_10_16.01/periods.jsonl').read_text('utf8').splitlines()]
        cache=ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}'
        out=[];counters=dict(proposals=0,original_qualified=0,split_qualified=0,recovered=0,lost=0,original_multi_frames=0,split_multi_frames=0)
        for row,period in zip(old,periods):
            t=row['t'];mid=(round(t*1e6)-33333+33333//2)/1e6
            eligible=[x for x in old if mid-.12-1e-8<=x['t']<=mid+1e-9];velocity=np.zeros(2)
            if eligible:
                mo=eligible[-1]['motion'] or {};shift=mo.get('estimate') or {};dt=mo.get('dt_s')
                if dt and mo.get('current_observation') and not shift.get('peak_at_search_boundary') and shift.get('score',0)>0:
                    velocity=np.array([shift['dx'],shift['dy']])/dt
            events=window(cache,t);records=[]
            for record in period['meta'].get('records',[]):
                arm=record.get('result',{}).get('arms',{}).get('joint_pose',{})
                if 'author_state' not in arm:continue
                seed=record['seed'];a=np.deg2rad(seed['angle']);co,si=np.cos(a),np.sin(a)
                d=np.c_[events['x'],events['y']]-[seed['cx'],seed['cy']]-(events['t']/1e6-t)[:,None]*velocity
                u=np.c_[(d[:,0]*co+d[:,1]*si)/(seed['major']/2),(-d[:,0]*si+d[:,1]*co)/(seed['minor']/2)]
                keep=np.all(abs(u)<=2.,axis=1);native=events[keep];xy=np.rint(128+80*u[keep]).astype(np.int32)
                early=native['t']<round(mid*1e6)
                control=record['result']['control'];m=control['angular_harmonic'];sign=1 if control['candidate_signed_phase_hz']>0 else -1
                result=validate(arm['author_state'],native,xy,early,angular_order=m,rotation_sign=sign)
                g=arm['inferred_geometry']
                if g.get('invalid'):continue
                scales=np.array([g['major']/seed['major'],g['minor']/seed['minor']])
                good=not g['fov_clipped'] and scales.min()>=.65 and scales.max()<=1.5 and np.linalg.norm([g['cx']-seed['cx'],g['cy']-seed['cy']])<=max(8.,.6*max(g['major'],seed['major']))
                qualified=bool(result['qualified'] and good)
                records.append(dict(original_qualified=record['qualified'],split_qualified=qualified,validation=result,
                    original_pooled_train=arm['train'].get('concentration'),original_pooled_late=arm['validation'].get('concentration')))
                counters['proposals']+=1;counters['original_qualified']+=bool(record['qualified']);counters['split_qualified']+=qualified
                counters['recovered']+=qualified and not record['qualified'];counters['lost']+=record['qualified'] and not qualified
            counters['original_multi_frames']+=sum(r['original_qualified'] for r in records)>=2
            counters['split_multi_frames']+=sum(r['split_qualified'] for r in records)>=2
            out.append(dict(t=t,records=records))
        results[n]=dict(summary=counters,records=out);print(n,counters,flush=True)
    protocol=dict(labels_read=False,NN_used=False,new_fits=False,new_candidates=False,
        thresholds_changed=False,only_discovery_phase_offsets_used=True,
        diagnostic_not_end_to_end_tracking=True)
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,protocol=protocol),indent=2),'utf8')
    for f in ('polarity_phase_validation.py','tools/rescore_polarities295.py'):
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
