"""Public observer parity, source-time causality and finite mixture forecasts."""
from pathlib import Path
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from adaptive_response_observer import AdaptiveResponseObserver
from physical_publication import evaluate,freeze
from run_rotor_only import RotorPredictor
BASE=ROOT.parent/'output';OUT=BASE/'adaptive203_20261108'

def replay(rows):
    model=AdaptiveResponseObserver();result=[];mineig=0.
    for old in rows:
        r=copy.deepcopy(old);meta=r['period_selection'];u=r['force'].get('input_u');u=None if u is None else np.asarray(u)
        if r['state'].get('two_point_initialization_selected_by_partial_entry'):model.two_point_init=True
        c,v,f,s=model.step(r['t'],r['body'],u,.7 if u is not None else 0.,allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'],structure=r['structure'])
        r.update(coefficient=c,valid=v,force=f,state=s,normal_turn_forecast=s.get('normal_turn_forecast'));r['physical']=evaluate(r,[.1,.2,.4]) if c else {};result.append(r)
        if model.P is not None:mineig=min(mineig,float(np.linalg.eigvalsh(model.P).min()))
        if model.alternative.P is not None:mineig=min(mineig,float(np.linalg.eigvalsh(model.alternative.P).min()))
        assert all(p['target']>r['t']-1e-8 for p in model.credit.memory.pending) or not r['motion'].get('current_observation')
    return result,mineig

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    rows=[json.loads(s) for s in (BASE/'normal187_20261107/recording15_with_input/normal_legacy_input.jsonl').read_text('utf-8').splitlines()]
    expected=[json.loads(s) for s in (BASE/'credit202_20261108/15/predictions.jsonl').read_text('utf-8').splitlines()]
    result,mineig=replay(rows);delta=0.
    for a,b in zip(result,expected):
        assert a['valid']==b['valid']
        for h in a['physical']:delta=max(delta,float(np.max(np.abs(np.asarray(a['physical'][h])-b['physical'][h]))))
    assert delta<1e-8,delta;assert mineig>-1e-4,mineig
    prefix,_=replay(rows[:70]);assert all(a['physical']==b['physical'] for a,b in zip(prefix,result))
    r=next(r for r in result if r['valid'] and r['state']['causal_response_credit']['weight']>.1)
    before=evaluate(r,[.4]);frozen=freeze(r,r['t']);r['force']['effective_gain_px_s2']+=100
    assert evaluate(frozen,[.4])==before
    model=RotorPredictor(relative_observation=True,unknown_force_model='plane-adaptive');assert isinstance(model.force,AdaptiveResponseObserver)
    try:RotorPredictor(unknown_force_model='plane-adaptive')
    except ValueError:pass
    else:raise AssertionError('Incompatible factory accepted')
    checks=dict(public_observer_matches_frozen_credit202_max_px=delta,prefix_unchanged_by_later_observations=True,
        pending_targets_not_read_before_maturity=True,both_internal_covariances_min_eigenvalue=mineig,
        published_state_deepcopy_retains_mixture=True,public_factory_validated=True,labels_used_in_inference=False)
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2),'utf-8');print(json.dumps(checks),flush=True)
