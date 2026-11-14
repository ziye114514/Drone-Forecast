"""Checks specific to the new gap prior, partition and entry-point controls."""
from pathlib import Path
import sys,json,copy,subprocess,ast,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from coherent_gap_observer import ShortGapObserver,CoherentGapObserver
from relative_rotor_observer import RelativeRotorForceObserver
from run_rotor_only import RotorPredictor
from physical_publication import evaluate
OUT=ROOT.parent/'output/gap173_20261107'

if __name__=='__main__':
    checks={}
    for visibility,mode,expected in [(False,None,ShortGapObserver),(False,'coherent',CoherentGapObserver),
            (False,'legacy',RelativeRotorForceObserver),(True,None,RelativeRotorForceObserver)]:
        m=RotorPredictor(relative_observation=True,legacy_visibility=visibility,gap_prediction=mode,reference_mode='legacy')
        assert type(m.force) is expected
    try:RotorPredictor(gap_prediction='short')
    except ValueError:pass
    else:raise AssertionError('Explicit mode without relative observations must not be ignored')
    checks['factory_default_and_explicit_controls']=True
    m=ShortGapObserver();m._initialize(np.array([100.,200.]));m.t=0.;m.seen=0.;m.clone_t=0.;m.observations=6
    m.x[2:6]=[40.,100.,200.,-700.];m.u=np.array([.3,-.4]);m.components={'visible':m.u.copy()};m._clone_position()
    other=copy.deepcopy(m);other._set_input(np.zeros(2),transfer_delta=-m.u);other.components={}
    for t in (.033333,.066666,.1):
        a,va,fa,sa=m.step(t,dict(detected=False),None,input_meta={})
        b,vb,fb,sb=other.step(t,dict(detected=False),None,input_meta={})
        for key in ('position','velocity','acceleration'):np.testing.assert_allclose(a[key],b[key],atol=1e-8)
        pa=evaluate(dict(coefficient=a,force=fa),[.4]);pb=evaluate(dict(coefficient=b,force=fb),[.4])
        np.testing.assert_allclose(pa['0.4'],pb['0.4'],atol=1e-8)
        assert m.seen==0. and m.observations==6 and np.linalg.eigvalsh(m.P).min()>-1e-5
    checks['hidden_partition_preserves_gap_prediction']=True
    checks['gap_does_not_create_observations_or_advance_seen']=True
    checks['covariance_min_eigenvalue']=float(np.linalg.eigvalsh(m.P).min())
    for name in ('run_rotor_only.py','run_fast_structure.py'):
        p=subprocess.run([sys.executable,str(ROOT/name),'--help'],capture_output=True,text=True,timeout=45)
        assert p.returncode==0 and '--rotor-gap-prediction' in p.stdout,(name,p.stderr[-500:])
    checks['both_cli_controls_present']=True
    for name in ('coherent_gap_observer.py','singer_position_state.py','rotor_force_observer.py','adaptive_force_memory.py',
        'run_rotor_only.py','run_fast_structure.py','tools/live_gap173.py','tools/replay_gap173.py','tools/replay_memory173.py','tools/probe_native_rims174.py'):
        ast.parse((ROOT/name).read_text('utf-8-sig'),filename=name)
    checks['python_syntax']=True
    summary=json.loads((OUT/'integrated/scores.json').read_text('utf-8'))
    checks['full_front_control_reproduces172']=all(v['control']['prior_or_replay_max_delta']<1e-8 for v in summary.values())
    checks['full_front_candidate_reproduces_observer_replay']=all(v['short']['prior_or_replay_max_delta']<1e-8 for v in summary.values())
    checks['goal30px30Hz_achieved']=False
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2),'utf-8');print(json.dumps(checks))
