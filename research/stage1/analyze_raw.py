"""RAW deletion diagnostics: clean output is NOT ground truth."""
import csv,json
import numpy as np
from mechanism import METHODS,HORIZONS,SCALES
from run_study import OUT,save,csvsave

def availability(observations):
    # Same causal0.16s tracking coast as the frozen runtime. No error-based mask.
    last=24.;result=[]
    for o in observations:
        if o['detected']:last=o['t']
        result.append(o['t']-last<=.16+1e-8)
    return np.array(result)

def main():
    root=OUT/'raw_interventions';completion=json.loads((root/'completion.json').read_text())
    names=list(completion['conditions']);q={n:np.load(root/n/'forecasts.npz') for n in names}
    obs={n:[json.loads(s) for s in (root/n/'observations.jsonl').read_text().splitlines()] for n in names}
    times=q['clean']['t'];valid={n:availability(obs[n]) for n in names};rows=[];human=[]
    labels=json.loads((OUT.parent/'paper_reproduction_v030/evaluation_data/train6.json').read_text(encoding='utf-8'))['labels']
    for name in names:
        for window,(lo,hi) in {'deletion':[24,25],'early_restore':[25,27],'late_restore':[27,30]}.items():
            selected=(times>lo)&(times<=hi);common=selected&valid[name]&valid['clean']
            for j,method in enumerate(METHODS):
                disagreement=np.linalg.norm(q[name]['prediction'][:,j,-1]-q['clean']['prediction'][:,j,-1],axis=1)
                drift=np.linalg.norm((q[name]['parameters'][:,j]-q['clean']['parameters'][:,j])/SCALES,axis=1)
                rows.append(dict(condition=name,window=window,method=method,frames=int(selected.sum()),
                    available_frames=int((selected&valid[name]).sum()),common_available_frames=int(common.sum()),
                    mean_paired_prediction_disagreement_px=float(disagreement[common].mean()) if common.any() else None,
                    normalized_parameter_drift=float(drift[common].mean()) if common.any() else None,
                    not_ground_truth_error=True))
        for label in labels:
            if label['state']!='present':continue
            target=label['t'];box=np.asarray(label['bbox']);truth=(box[:2]+box[2:])/2
            for h in HORIZONS:
                k=np.searchsorted(times,target-h,side='right')-1
                if k<0 or target>30.4 or target-h>times[-1]+1e-9:continue
                actual_h=target-times[k]
                if actual_h>.411:continue
                # Published curves have discrete horizons. Interpolate predictions
                # only (not truth); small <=10ms excess uses analytic state/model
                # continuation from the final two published endpoints.
                for j,method in enumerate(METHODS):
                    p=np.array([np.interp(actual_h,HORIZONS,q[name]['prediction'][k,j,:,axis]) for axis in range(2)])
                    if actual_h>.4:
                        p+=(actual_h-.4)*(q[name]['prediction'][k,j,-1]-q[name]['prediction'][k,j,-2])/.2
                    human.append(dict(condition=name,method=method,target_time_s=target,anchor_time_s=float(times[k]),
                        requested_horizon_s=float(h),actual_horizon_s=actual_h,available=bool(valid[name][k]),
                        common_available=bool(valid[name][k] and valid['clean'][k]),
                        error_px=float(np.linalg.norm(p-truth)),
                        scoring_note='Existing sparse human boxes; small prediction interpolation/linear endpoint continuation, notGT interpolation. Shared-state ablation, not original v0.3 pipeline score.'))
    csvsave(OUT/'raw_paired_diagnostics.csv',rows);csvsave(OUT/'raw_sparse_endpoint_errors.csv',human)
    summaries=[]
    for name in names:
        for method in METHODS:
            for h in HORIZONS:
                r=[a for a in human if a['condition']==name and a['method']==method and a['requested_horizon_s']==h]
                common=[a for a in r if a['common_available']]
                summaries.append(dict(condition=name,method=method,horizon_s=float(h),targets=len(r),
                    available=sum(a['available'] for a in r),common_available=len(common),
                    common_mean_error_px=float(np.mean([a['error_px'] for a in common])) if common else None))
    save(OUT/'raw_analysis.json',dict(paired_diagnostics=rows,sparse_annotation_metrics=summaries,
        limitations=['One predetermined development clip, no physical pose/RPM truth.',
                    'Paired prediction disagreement measures sensitivity, not prediction accuracy.',
                    'Event deletion does not add occluder-generated events.',
                    'Rectangles remain fixed in sensor coordinates; moving drone can leave the mask.',
                    'Very few existing human timestamps; no significance/generalization claim.',
                    'Ablation uses a shared current state, not the full original coupled-position pipeline.']))
    print(json.dumps([r for r in rows if r['window']=='early_restore' and r['method']=='ordinary_rls']),flush=True)

if __name__=='__main__':main()
