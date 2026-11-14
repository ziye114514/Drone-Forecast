"""Reuse the existing extended-state observer on saved causal physical inputs.

The no-force arm separates position-feedback benefits from event-force benefits.
No parameter/window search, RAW replay, GPU or truth enters these observers.
"""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from lumped_disturbance import LumpedDisturbance,forecasts
from fast_force_response import FastForceResponse
from summarize_live_pilot import reference_at,stats


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=ROOT.parent/'output/three_continuation_20261026'
    results={}
    for name in ['side','clear']:
        rows=list(map(json.loads,(source/f'raw_release_{name}'/'predictions.jsonl').read_text('utf-8').splitlines()))
        observers={n:LumpedDisturbance() for n in ['no_input','fixed_gain','online_gain']}
        gain=200.;learner=FastForceResponse();samples=[]
        for r in rows:
            position=r.get('force_observed_position')
            if position is None and r['body'].get('detected'):position=r['body']['center']
            u=None if r['force']['input_u'] is None else np.asarray(r['force']['input_u'])
            arms={}
            for n,m in observers.items():
                known=None if n=='no_input' or u is None else u*(gain if n=='online_gain' else 200.)
                coefficient,diag=m.step(r['t'],position,known)
                arms[n]=dict(prediction=forecasts(coefficient),coefficient=coefficient,observer=diag)
            previous_gain=gain
            _,learning=learner.step(r['t'],arms['online_gain']['coefficient'],position,u,.35)
            gain=learner.gain
            if u is not None and observers['online_gain'].x is not None:
                # Gain changes are a decomposition change, not measured thrust.
                observers['online_gain'].x[4:6]-=(gain-previous_gain)*u
                observers['online_gain'].previous_u=gain*u
            samples.append(dict(t=r['t'],valid=r['valid'],arms=arms,
                original=r['physical'],gain_learning=learning))
        results[name]=samples
    (out/'causal_outputs.json').write_text(json.dumps(results),'utf-8')
    truth=sorted(json.loads((ROOT.parent/'output/dense_labels_side_110_1195/labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    errors={str(h):{n:[] for n in ['original','no_input','fixed_gain','online_gain']} for h in (.1,.2,.4)}
    last=None;unready=0;common=0
    for r in results['side']:
        slot=int(np.floor(r['t']*5+1e-8))
        if not r['valid'] or slot==last:continue
        last=slot
        if not all(a['observer']['ready'] for a in r['arms'].values()):unready+=1;continue
        common+=1
        for h in (.1,.2,.4):
            ref=reference_at(truth,r['t']+h)
            if ref is None:continue
            for n in errors[str(h)]:
                p=r['original'][str(h)] if n=='original' else r['arms'][n]['prediction'][str(h)]
                errors[str(h)][n].append(float(np.linalg.norm(p-ref)))
    result=dict(saved_inputs_only=True,no_parameter_search=True,predictions_saved_before_labels=True,
        source='raw_release_side/clear; native causal inputs already frozen',common_side_slots=common,
        side_original_slots_unready_for_observer=unready,
        ready_counts={name:{n:sum(r['arms'][n]['observer']['ready'] for r in rows) for n in observers} for name,rows in results.items()},
        scores={h:{n:stats(v) for n,v in arm.items()} for h,arm in errors.items()},
        final_learned_gain={n:rows[-1]['gain_learning']['effective_gain_px_s2'] for n,rows in results.items()},
        limitations='Position observer benefit is not event-force benefit: compare fixed/online gain with no_input. Sparse previously exposed bbox references; no physical force truth.')
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1])
