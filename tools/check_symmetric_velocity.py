"""Known two-Gaussian symmetric CI before the fixed experiment."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from live_local_flow import velocity_coefficient
OUT=ROOT.parent/'output/symmetric_velocity_20261026';OUT.mkdir(exist_ok=False)
c=dict(position=[40.,60.],velocity=[100.,-20.],acceleration=[0.,0.],conditional_scale_rate=None)
obs=dict(valid=True,velocity=[110.,-30.],velocity_covariance=(np.eye(2)*25.).tolist(),midpoint_t=1.)
new,audit=velocity_coefficient(c,np.eye(2)*100.,obs,1.,strategy='symmetric')
assert np.allclose(new['velocity'],[108.,-28.],atol=1e-10) and audit['baseline_weight']==.5
assert new['position']==c['position'] and new['acceleration']==c['acceleration'] and c['velocity']==[100.,-20.]
absent,none=velocity_coefficient(c,np.eye(2)*100.,dict(valid=False,reason='absent'),1.,strategy='symmetric')
assert absent is c and not none['applied']
checks=dict(known_symmetric_precision_weighting_exact=True,no_position_or_acceleration_change=True,invalid_exact_fallback=True)
(OUT/'mechanism_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
design=dict(omega=.5,source='https://stonesoup.readthedocs.io/en/v1.7/_modules/stonesoup/updater/chernoff.html',
    method='Established Gaussian CI formula, own small NumPy implementation; no dependency installation or claim to run Stone Soup.',
    alpha_sweep=False,selection='One fixed symmetric setting from public implementation default; test alone and with the already-fixed spatial covariance proxy.',
    parent='accumulated_v4_local_flow',cache_only_no_RAW_GPU=True,globally_blind=False)
(OUT/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8');print(json.dumps(checks),flush=True)
