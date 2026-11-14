"""Clock advancement, immutable handoff and explicit missing-state behavior."""
from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from physical_publication import freeze,evaluate,publication


def checks():
    row=dict(t=1.,valid=True,coefficient=dict(position=[100.,200.],velocity=[10.,-5.],acceleration=[0.,0.]),
        force=dict(active=False,input_u=None,effective_gain_px_s2=200.,acceleration_px_s2=[0.,0.],acceleration_correlation_s=.35))
    state=freeze(row,1.);row['coefficient']['position'][0]=-10000.
    result=publication(state,1.05,1.05,1)
    assert result['valid'] and np.allclose(result['physical']['0.4'],[104.5,197.75])
    assert abs(result['target_times']['0.4']-1.45)<1e-12
    assert not result['current_image_observation'] and result['source_t']==1.
    assert not publication(state,1.2,1.2,2)['valid']
    assert not publication(state,.99,.99,3)['valid']
    assert not publication(None,1.05,1.05,1)['valid']
    state['last_image_t']=.8;assert not publication(state,1.05,1.05,4)['valid']
    state['force']['acceleration_px_s2']=[1000.,-500.]
    expected=np.array([100.,200.])+.4*np.array([10.,-5.])+.35*(.4+.35*np.expm1(-.4/.35))*np.array([1000.,-500.])
    assert np.allclose(evaluate(state,[.4])['0.4'],expected)
    return dict(exact_actual_time_target=True,original_state_immutable=True,stale_or_future_or_missing_state_invalid=True,
        last_actual_image_age_checked=True,Singer_integral_preserved=True,repeated_model_output_is_not_current_detection=True)


if __name__=='__main__':
    import json
    print(json.dumps(checks(),indent=2))
