"""POST-HOC nondeployable error budget. Oracle rows are NOT forecasts/targets met.

Centered reference derivatives use FUTURE anchors; they only ask how much
present-state error could matter. Sparse weak outline references are noisy,
so this is not an irreducible forecast bound or proof of future unpredictability.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from physical_publication import evaluate
from conditional_unknown_force import plain
from extend_inverse321 import references
from compare_final_pipeline import reference_at,stats

def adjust(row,dp=None,dv=None,da=None):
    r=copy.deepcopy(row)
    def change(p):
        for k,d in [('position',dp),('velocity',dv),('acceleration',da)]:
            if d is not None:p['coefficient'][k]=(np.asarray(p['coefficient'][k])+d).tolist()
        if da is not None:p['force']['acceleration_px_s2']=(np.asarray(p['force']['acceleration_px_s2'])+da).tolist()
        for part in p['force'].get('response_model_mixture',[]):change(part)
    change(r);return r

if __name__=='__main__':
    out=ROOT.parent/'output/budget390_20261110';out.mkdir(exist_ok=False)
    cases={}
    for n in ('13','15','16','11','3','4'):
        rows=[json.loads(s) for s in (ROOT.parent/f'output/force357_20261109/{n}.jsonl').read_text('utf8').splitlines()]
        ref=ROOT.parent/'output/reference323_20261109/references.json' if n=='4' else references(n)
        truth=sorted(json.loads(ref.read_text('utf8'))['records'].values(),key=lambda r:r['t']);records=[]
        for t in np.round(np.arange(10.8,15.6001,.2),1):
            r=max((r for r in rows if r['t']<=t+1e-9),key=lambda r:r['t']);target=reference_at(truth,t+.4);now=reference_at(truth,t)
            if not r['valid'] or t-r['t']>.04 or target is None:continue
            h=float(t-r['t']+.4);predict=np.asarray(evaluate(r,[h])[str(h)]);base=np.asarray(evaluate(plain(r),[h])[str(h)]);delta=predict-base
            scale=max(0.,float(delta@(target-base))/float(delta@delta)) if float(delta@delta)>1e-12 else 1.
            rec=dict(t=float(t),actual_FDE=float(np.linalg.norm(predict-target)),actual_current_position_error=None if now is None else float(np.linalg.norm(now-np.asarray(r['coefficient']['position']))),
                known_future_normal_amplitude_oracle_error=float(np.linalg.norm(base+scale*delta-target)),oracle_normal_amplitude_scale=scale,
                centered_derivative_oracles_use_future_data=True)
            before=reference_at(truth,t-.2);after=reference_at(truth,t+.2)
            if now is not None:
                dp=now-np.asarray(r['coefficient']['position']);rec['oracle_position_FDE']=float(np.linalg.norm(np.asarray(evaluate(adjust(r,dp=dp),[h])[str(h)])-target))
                if before is not None and after is not None:
                    v=(after-before)/.4;a=(after-2*now+before)/.04
                    dv=v-np.asarray(r['coefficient']['velocity']);da=a-np.asarray(r['coefficient']['acceleration'])
                    rec.update(centered_reference_velocity=v.tolist(),centered_reference_acceleration=a.tolist(),
                        velocity_state_disagreement_px_s=float(np.linalg.norm(dv)),acceleration_state_disagreement_px_s2=float(np.linalg.norm(da)))
                    for name,change in [('position_velocity',dict(dp=dp,dv=dv)),('position_velocity_acceleration',dict(dp=dp,dv=dv,da=da))]:
                        rec['oracle_'+name+'_FDE']=float(np.linalg.norm(np.asarray(evaluate(adjust(r,**change),[h])[str(h)])-target))
            records.append(rec)
        keys=['actual_FDE','oracle_position_FDE','oracle_position_velocity_FDE','oracle_position_velocity_acceleration_FDE','known_future_normal_amplitude_oracle_error']
        cases[n]=dict(records=records,summary={k:dict(actual_on_same_subset=stats([r['actual_FDE'] for r in records if k in r]),diagnostic=stats([r[k] for r in records if k in r])) for k in keys},
            oracle_amplitude_scale_percentiles=np.percentile([r['oracle_normal_amplitude_scale'] for r in records],[0,50,95,100]).tolist())
        print(n,{k:(v['diagnostic']['n'],round(v['actual_on_same_subset']['mean'],2),round(v['diagnostic']['mean'],2)) for k,v in cases[n]['summary'].items()},flush=True)
    (out/'results.json').write_text(json.dumps(dict(cases=cases,posthoc_only=True,oracles_not_deployable=True,not_a35px_achievement=True,weak_outline_reference_derivatives_are_noisy=True,not_an_irreducible_error_bound=True),indent=2),'utf8')
