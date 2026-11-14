"""Activate the already-equivalent packaged core through compatibility imports."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/portable_position_core_20261025'
check=json.loads((OUT/'source_equivalence.json').read_text(encoding='utf-8'))
assert check['all_coefficients_and_forecasts_exact'] and check['rows']==632
prefix='from pathlib import Path\nimport sys\nROOT=Path(__file__).resolve().parent\nsys.path.insert(0,str(ROOT/"src"))\n'
changes={
 'live_motion.py':'"""Source-checkout compatibility imports for the installable position core."""\n'+prefix+
    'from event_drone_forecast.position import LiveMotion,predictions\nfrom event_drone_forecast.position.kalman import SimpleKF\nfrom event_drone_forecast.position.robust import robust_acceleration,cap\nfrom event_drone_forecast.position.rollout import forecast as curved_forecast\n',
 'live_measurement_kf.py':'"""Compatibility imports; no historical research module is loaded."""\n'+prefix+
    'from event_drone_forecast.position.kalman import MeasurementKF,SimpleKF,predict,update,coefficient,PARAMS\n',
 'live_covariance.py':'"""Compatibility imports for optional continuous-time covariance."""\n'+prefix+
    'from event_drone_forecast.position.covariance import ContinuousKF,process_noise,REFERENCE_DT,SPECTRAL_DENSITY,author_Q\n'}
for name,text in changes.items():
    assert (ROOT/name).read_bytes()==(OUT/'before'/name).read_bytes(),name
    (ROOT/name).write_text(text,encoding='utf-8')
gain=ROOT/'live_gain.py';old=gain.read_text(encoding='utf-8')
assert old.count('from turn154 import forecast as base_forecast')==1
gain.write_text(old.replace('from turn154 import forecast as base_forecast','from event_drone_forecast.position.rollout import forecast as base_forecast'),encoding='utf-8')
print('Activated packaged core; historical source modules unchanged',flush=True)
