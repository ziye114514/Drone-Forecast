"""Bounded checks of the delivered mode and frozen default reproduction."""
from pathlib import Path
import sys,json,numpy as np,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_rotor_only import RotorPredictor
from layout_reference_observer import LayoutReferenceObserver
from normal_response_observer import NormalResponseObserver
from compare_final_pipeline import reference_at
from replay_normal187 import mechanism

if __name__=='__main__':
    out=ROOT.parent/'output/rootfix191_20261107';out.mkdir(exist_ok=True);checks=mechanism()
    common=dict(relative_observation=True,reference_mode='fixed',gap_prediction='short')
    a=RotorPredictor(**common);b=RotorPredictor(**common,unknown_force_model='plane')
    assert type(a.force) is LayoutReferenceObserver and type(b.force) is NormalResponseObserver
    rejected=0
    for kw in [dict(unknown_force_model='plane'),dict(**common,unknown_force_model='plane',legacy_visibility=True)]:
        try:RotorPredictor(**kw)
        except ValueError:rejected+=1
    assert rejected==2
    checks.update(default176_preserved=True,plane_factory_wired=True,invalid_modes_rejected_before_GPU=True)
    truth=[dict(t=t,state='present',box=[0,0,20,20]) for t in [20.,20.2,20.4]]
    assert np.allclose(reference_at(truth,20.2+1e-14),[10,10])
    assert reference_at(truth,20.3) is None
    checks['exact_left_timestamp_fixed_without_interpolating_large_gaps']=True
    path=ROOT.parent/'output/unseen191_20261107/recording16/control176.jsonl'
    rows=[json.loads(s) for s in path.read_text('utf-8').splitlines()];m=LayoutReferenceObserver();maximum=0.
    for r in rows:
        if r['state'].get('two_point_initialization_selected_by_partial_entry'):m.two_point_init=True
        meta=r['period_selection'];u=r['force'].get('input_u');u=None if u is None else np.asarray(u)
        c,v,f,s=m.step(r['t'],r['body'],u,.7 if u is not None else 0.,
            allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=r['motion'])
        assert v==r['valid']
        if c is not None:
            for key in ['position','velocity','acceleration']:
                maximum=max(maximum,float(np.max(np.abs(np.asarray(c[key])-r['coefficient'][key]))))
    assert maximum<1e-8,maximum
    checks['recording16_default_replay_max_state_delta']=maximum
    freeze=json.loads((ROOT.parent/'output/unseen191_20261107/predecode_freeze.json').read_text('utf-8'))
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in freeze['source_sha256'].items())
    checks['test_source_hashes_still_frozen']=True
    (out/'checks.json').write_text(json.dumps(checks,indent=2),'utf-8');print(json.dumps(checks),flush=True)
