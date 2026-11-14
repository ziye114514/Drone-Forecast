"""Tracked point changes must not inject velocity, thrust or unknown force."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from rotor_force_observer import RotorForceObserver
from relative_rotor_observer import RelativeRotorForceObserver
from physical_publication import evaluate
from probe_observed_force_partition import mechanism_checks

if __name__=='__main__':
    result={}
    for cls in [RotorForceObserver,RelativeRotorForceObserver]:
        model=cls();model._initialize(np.array([200.,100.]));model.x[2:6]=[150.,-20.,12.,-5.]
        before=model.x.copy();shift=np.array([21.,-13.])
        model._rebase_observation_reference(shift)
        np.testing.assert_array_equal(model.x[2:7],before[2:7])
        np.testing.assert_allclose(model.x[:2]-before[:2],shift,atol=1e-12)
        if cls is RelativeRotorForceObserver:np.testing.assert_allclose(model.x[:2]-model.x[7:9],before[:2]-before[7:9])
        assert np.linalg.eigvalsh(model.P).min()>-1e-5
        result[cls.__name__]=dict(position_shift_only=True,velocity_and_force_unchanged=True,covariance_psd=True)
    basic=mechanism_checks()
    result['existing_force_partition']={k:v for k,v in basic.items() if k not in ('visibility_log','step_diagnostics')}
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
