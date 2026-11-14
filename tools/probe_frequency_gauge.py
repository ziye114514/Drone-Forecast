"""Reuse existing visibility-conditioned selector on latest physical rollout.

Only period/geometry observations are passed to the selector. No trajectory
annotation, learned force value, or future outcome chooses a frequency gauge.
"""
from pathlib import Path
import copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from prepare_fred_transfer import OUT
from visibility_period import VisibilityPeriodProxy,ConservativeSlotPeriodProxy
from axis_amplitude_bridge import AxisAmplitudeBridge
from probe_same_window_motion import replay
from compare_final_pipeline import score


def main():
    out=OUT/'frequency_gauge';out.mkdir(exist_ok=False)
    base=ROOT.parent/'output/motion_consistency_30px_20261028'
    paths=dict(side=base/'RAW_side_release_candidate8s/predictions.jsonl',
        other=ROOT.parent/'output/thrust_rate_review_20261029/experiment/other_baseline.jsonl',
        clear=base/'RAW_clear_release_candidate2s/predictions.jsonl')
    records={};result={}
    for name,path in paths.items():
        source=[json.loads(x) for x in path.read_text().splitlines()]
        records[name]={};result[name]={}
        for tag,cls in [('separate_pool',VisibilityPeriodProxy),('conservative_slots',ConservativeSlotPeriodProxy)]:
            selector=cls();bridge=AxisAmplitudeBridge();rows=[]
            for r in source:
                u,q,d=selector.step(r['t'],r['structure'],r['geometry'].get('rotors',[]),r['rates'],r['period'])
                u,q,a=bridge.step(r['t'],r['structure'],u,q,d)
                rows.append(dict(r,period_selection=d,axis_amplitude_bridge=a,force=dict(r['force'],input_u=u)))
            values,cost=replay(rows,False,curved=True,evidence_gain=True,memory=True,endpoint=True,memory_gaps=True,weak_support=True)
            (out/f'{name}_{tag}.jsonl').write_text('\n'.join(json.dumps(r,allow_nan=False) for r in values)+'\n','utf-8')
            records[name][tag]=values
            result[name][tag]=dict(force_inputs=sum(r['force']['active'] for r in values),
                gain_updates=sum(r['force']['gain_learned'] for r in values),
                qualified_slot_ticks=sum(bool(r['period_selection'].get('slot_normalization_applied') or r['period_selection'].get('full_sum_observed')) for r in values))
    for name,modes in records.items():
        if name=='clear':continue
        path=ROOT.parent/('output/dense_labels_side_110_1195/labels.json' if name=='side' else 'output/dense_labels_train2_40_46/labels.json')
        labels=sorted(json.loads(path.read_text())['records'].values(),key=lambda r:r['t'])
        for tag,rows in modes.items():result[name][tag]['error']=score(rows,labels)['physical_future_error_px']['0.4']
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
