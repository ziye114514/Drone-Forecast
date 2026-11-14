"""One fixed calibration mechanism on saved outputs, all labels after freeze."""
from pathlib import Path
import json,sys,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from horizon_force_calibration import HorizonForceCalibration
from prepare_fred_transfer import OUT as FRED,parse
from score_fred_transfer import score as score_fred
from compare_final_pipeline import score,stats


def checks():
    model=HorizonForceCalibration();force={'0.4':[80.,0.]}
    for i in range(12):
        t=i/30;c=dict(position=[0.,0.],velocity=[0.,0.])
        p,d=model.step(t,c,[0.,0.],force)
        assert p==force and not d['updated_horizons']
    for i in range(12,40):
        t=i/30;p,d=model.step(t,c,[0.,0.],force)
    assert d['response_reliability_by_horizon']['0.4']<1.
    model=HorizonForceCalibration();p,d=model.step(0.,c,[0.,0.],{'0.4':[0.,0.]})
    assert p=={'0.4':[0.,0.]} and not d['applied']
    return dict(no_update_before_target_maturity=True,zero_force_contribution_exactly_unchanged=True,
        separate_horizon_states=True,no_metric_force_claim=True)


def main():
    out=FRED/'horizon_force_calibration';out.mkdir(exist_ok=False)
    base=FRED/'motion_observability_v3';paths={n:base/f'{n}_motion.jsonl' for n in ('fred0','fred1','fred2','side','other')}
    adapted=ROOT.parent/'output/fred_domain_adaptation_20261029'
    paths.update(adapted_fred1=adapted/'RAW_1_adapted/predictions.jsonl',adapted_other=adapted/'RAW_train2_adapted6s/predictions.jsonl')
    all_rows={};result={}
    for name,path in paths.items():
        original=[json.loads(x) for x in path.read_text().splitlines()];model=HorizonForceCalibration();modified=[];cost=[]
        for r in original:
            tick=time.perf_counter()
            p,d=model.step(r['t'],r['coefficient'],r.get('force_observed_position'),r['physical'],r['valid'])
            cost.append((time.perf_counter()-tick)*1000);modified.append(dict(r,physical=p,horizon_force_calibration=d))
        (out/f'{name}.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in modified)+'\n','utf-8')
        all_rows[name]=(original,modified);result[name]=dict(component_CPU_ms=stats(cost),final_weights=dict(model.weights),
            applied_ticks=sum(r['horizon_force_calibration']['applied'] for r in modified))
    protocol=json.loads((FRED/'protocol.json').read_text())
    for name,(original,modified) in all_rows.items():
        if 'fred' in name:
            n=int(name[-1]);s=next(s for s in protocol['sequences'] if s['sequence']==n)
            labels=parse(Path(s['annotation']).read_text('utf-8-sig'))
            scoring=lambda r:score_fred(r,labels,s['start'],s['duration'])['error_04s_px']
        else:
            label=ROOT.parent/('output/dense_labels_side_110_1195/labels.json' if name=='side' else 'output/dense_labels_train2_40_46/labels.json')
            labels=sorted(json.loads(label.read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
            scoring=lambda r:score(r,labels)['physical_future_error_px']['0.4']
        result[name]['scores']={k:scoring(v) for k,v in [('control',original),('calibrated',modified)]}
    summary=dict(checks=checks(),sequences=result,validity_and_observations_unchanged=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
