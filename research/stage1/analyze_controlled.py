"""Seed-paired analysis, with no tuning of the frozen experimental protocol."""
from pathlib import Path
import json,sys
import numpy as np
from mechanism import METHODS,HORIZONS,SCALES
from run_study import OUT,verify,save,csvsave

def recovery(times,excess,end,dt,threshold=1.,average=.5,hold=1.):
    window=round(average/dt);sustain=round(hold/dt)
    means=np.convolve(excess,np.ones(window)/window,mode='valid')
    for i,t in enumerate(times[:len(means)]):
        if t<end-1e-8 or i+sustain>len(means):continue
        if np.all(means[i:i+sustain]<=threshold):return float(t-end)
    return None

def main():
    protocol=verify();rows=[];recoveries=[];prefix_checks=negative_checks=trials=0
    for directory in sorted((OUT/'controlled').glob('*/*/seed_*')):
        seed=int(directory.name.split('_')[1]);world=directory.parent.name;mode=directory.parent.parent.name
        clean=np.load(directory/'clean.npz');truth=np.load(directory/'truth.npz');times=clean['times'];dt=times[1]-times[0]
        for scenario in protocol['scenarios']:
            q=np.load(directory/f'{scenario}.npz');trials+=1
            prefix=times<protocol['fault_start_s']
            np.testing.assert_array_equal(q['prediction'][prefix],clean['prediction'][prefix]);prefix_checks+=1
            if scenario=='id_permutation_only':
                np.testing.assert_array_equal(q['prediction'],clean['prediction']);negative_checks+=1
            for hi,h in enumerate(HORIZONS):
                lag=round(h/dt);N=len(times)-lag;t=times[:N]
                actual=truth['position'][lag:]
                errors=np.linalg.norm(q['prediction'][:N,:,hi]-actual[:,None,:],axis=2)
                baseline=np.linalg.norm(clean['prediction'][:N,:,hi]-actual[:,None,:],axis=2)
                drift=np.linalg.norm((q['parameters'][:N]-clean['parameters'][:N])/SCALES,axis=2)
                residual_drift=np.linalg.norm(q['disturbance'][:N]-clean['disturbance'][:N],axis=2)
                true_parameter_error=np.linalg.norm((q['parameters'][:N]-truth['true_parameters'])/SCALES,axis=2)
                for window,(lo,up) in protocol['evaluation_windows'].items():
                    keep=(t>=lo-1e-9)&(t<up-1e-9)
                    for j,method in enumerate(METHODS):
                        rows.append(dict(mode=mode,world=world,seed=seed,scenario=scenario,method=method,horizon_s=float(h),window=window,
                            anchors=int(keep.sum()),mean_error_px=float(errors[keep,j].mean()),p90_error_px=float(np.percentile(errors[keep,j],90)),
                            paired_clean_mean_error_px=float(baseline[keep,j].mean()),
                            paired_excess_error_px=float((errors[keep,j]-baseline[keep,j]).mean()),
                            normalized_parameter_drift=float(drift[keep,j].mean()),
                            residual_drift_px_s2=float(residual_drift[keep,j].mean()),
                            normalized_true_parameter_error=float(true_parameter_error[keep,j].mean()) if method!='cv' else None))
                if h==.4 and scenario!='clean':
                    for j,method in enumerate(METHODS):
                        recovered=recovery(t,errors[:,j]-baseline[:,j],protocol['fault_end_s'],dt)
                        recoveries.append(dict(mode=mode,world=world,seed=seed,scenario=scenario,method=method,
                            recovered=recovered is not None,recovery_s=recovered,followup_s=float(t[-1]-protocol['fault_end_s']),
                            definition='Offline paired excess error, forward0.5s average<=1px sustained1s; not parameter identification or an online flag.'))
    csvsave(OUT/'controlled_trial_metrics.csv',rows);csvsave(OUT/'controlled_recovery.csv',recoveries)
    rng=np.random.default_rng(20261016);aggregate=[]
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ('mode','world','scenario','method','horizon_s','window'));groups.setdefault(key,[]).append(row)
    for key,values in groups.items():
        excess=np.array([v['paired_excess_error_px'] for v in values]);n=len(values)
        boot=np.mean(excess[rng.integers(0,n,(2000,n))],axis=1) if n>1 else None
        aggregate.append(dict(zip(('mode','world','scenario','method','horizon_s','window'),key),
            seeds=n,mean_error_px=float(np.mean([v['mean_error_px'] for v in values])),
            paired_clean_mean_error_px=float(np.mean([v['paired_clean_mean_error_px'] for v in values])),
            paired_excess_error_px=float(excess.mean()),
            excess_ci95_low_px=float(np.quantile(boot,.025)) if n>1 else None,
            excess_ci95_high_px=float(np.quantile(boot,.975)) if n>1 else None,
            normalized_parameter_drift=float(np.mean([v['normalized_parameter_drift'] for v in values])),
            residual_drift_px_s2=float(np.mean([v['residual_drift_px_s2'] for v in values]))))
    csvsave(OUT/'controlled_aggregate_metrics.csv',aggregate)
    summary=dict(trials=trials,paired_prefix_checks=prefix_checks,id_invariance_checks=negative_checks,
        confidence_unit='20 seed-level paired trial means per primary group, not individual correlated frames; percentile bootstrap, 2000 replicates; no multiple-comparison significance claims.',
        metrics=aggregate,recovery=recoveries,
        limitations=['Observation-level synthetic dynamics, no photometric event simulator.',
          '20 seeds share a prescribed maneuver profile: Monte Carlo robustness, not20 independent real flights.',
          'Oracle state and clean-learner transplantation are causal-mechanism diagnostics, not deployable methods.',
          'Absolute clean RLS error and drift are reported; adaptation is not assumed to improve even clean predictions.',
          'All predictors share current state and constant-current-force rollout; this is an ablation, not original end-to-end v0.3 scores.'])
    save(OUT/'controlled_analysis.json',summary)
    print(json.dumps([r for r in aggregate if r['mode']=='estimated' and r['world']=='matched' and r['horizon_s']==.4 and r['window']=='post_early'
                     and r['method'] in ('fixed','ordinary_rls','quality_rls','ordinary_reset_oracle')]),flush=True)

if __name__=='__main__':main()
