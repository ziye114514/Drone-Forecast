"""Fixed causal inputs; save both forecasts before opening sparse references."""
from pathlib import Path
import sys,json,time,hashlib
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from visibility_period import VisibilityPeriodProxy,LegacyPeriodProxy,ConservativeSlotPeriodProxy
from fast_force_response import FastForceResponse
from summarize_live_pilot import reference_at,stats


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=ROOT.parent/'output/three_continuation_20261026'
    all_rows={};summaries={}
    for name in ['clear','side']:
        path=source/f'raw_release_{name}/predictions.jsonl'
        inputs=[json.loads(x) for x in path.read_text('utf-8').splitlines()]
        selectors=dict(legacy=LegacyPeriodProxy(),visibility=VisibilityPeriodProxy(),slots_only=ConservativeSlotPeriodProxy())
        forces={n:FastForceResponse() for n in selectors};rows=[];timing=[];diff=0.
        for r in inputs:
            arms={}
            for n,m in selectors.items():
                start=time.perf_counter()
                u,q,info=m.step(r['t'],r['structure'],r['geometry'].get('rotors',[]),r['rates'],r['period'])
                if n=='visibility':timing.append((time.perf_counter()-start)*1000)
                pred,force=forces[n].step(r['t'],r['coefficient'],r.get('force_observed_position'),u,q)
                arms[n]=dict(prediction=pred,force=force,period=info)
                if n=='legacy' and pred:
                    diff=max(diff,max(float(np.max(np.abs(np.asarray(p)-r['physical'][h]))) for h,p in pred.items()))
            rows.append(dict(t=r['t'],valid=r['valid'],arms=arms))
        assert diff<1e-8,(name,diff)
        all_rows[name]=rows
        slot_difference=max((float(np.max(np.abs(np.asarray(p)-r['arms']['legacy']['prediction'][h])))
            for r in rows for h,p in r['arms']['slots_only']['prediction'].items()),default=0.)
        summaries[name]=dict(source_ticks=len(rows),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            legacy_prediction_max_difference=diff,
            active_force_counts={n:sum(r['arms'][n]['force']['active'] for r in rows) for n in selectors},
            visibility_sources=dict(Counter(r['arms']['visibility']['period']['source'] for r in rows)),
            initialized_baselines=sum(r['arms']['visibility']['period']['slot_baseline_initialized'] for r in rows),
            slot_only_prediction_max_change_px=slot_difference,
            component_ms=dict(mean=float(np.mean(timing)),p95=float(np.percentile(timing,95))))
    (out/'causal_outputs.json').write_text(json.dumps(all_rows),'utf-8')
    # Labels are accessed only after every prediction is frozen to disk.
    truth=sorted(json.loads((ROOT.parent/'output/dense_labels_side_110_1195/labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    errors={str(h):{n:[] for n in selectors} for h in (.1,.2,.4)};samples=[];last=None
    for r in all_rows['side']:
        slot=int(np.floor(r['t']*5+1e-8))
        if not r['valid'] or slot==last:continue
        last=slot
        for h in (.1,.2,.4):
            ref=reference_at(truth,r['t']+h)
            if ref is None:continue
            sample=dict(t=r['t'],horizon=h,errors={})
            for n in selectors:
                p=r['arms'][n]['prediction'][str(h)]
                error=float(np.linalg.norm(np.asarray(p)-ref));errors[str(h)][n].append(error);sample['errors'][n]=error
            samples.append(sample)
    result=dict(saved_causal_inputs_only=True,RAW_or_GPU_run=False,predictions_saved_before_labels=True,
        no_parameter_search=True,cases=summaries,scores={h:{n:stats(v) for n,v in a.items()} for h,a in errors.items()},
        limitations='Previously exposed sparse bbox references; no clear-window labels or physical RPM/force truth; not end-to-end timing or a blind test')
    (out/'scores.json').write_text(json.dumps(samples,indent=2),'utf-8')
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1])
