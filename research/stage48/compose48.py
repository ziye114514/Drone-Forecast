"""Better whole-UAV positions with unchanged causal propeller physics experts."""
from pathlib import Path
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage28'),str(ROOT/'research/stage29')]
from axis28 import Predictor
from select29 import Selector,MODELS,prior
METHODS=['legacy_rotors','slotfix_rotors','associated_slotfix']
PARAMS=dict(association_margin_fraction=.25,require_all_frequency_tracks_inside_body=True,
    position='Identical frozen45 body_first native20 position/detection/confidence/bbox on all methods. Original34 real RAW rotor sensing is independent10ms and sampled at exact current20ms cutoff, never a later sample.',
    physics='Execute original28 canonical_static, original27 projected thrust/RLS, original29 delayed expert selection. No force/position hyperparameter tuning. On replacement calibration epoch change, reset only physical features/parameters, keep identical position/history/selector state.',
    interpretation='Body box supplies position and optional association gate, never a rotor center/normal/RPM. Parameters are effective pixel dynamics, not identified true mass, thrust coefficient or wind.')

def empty_sensing():return dict(rotors=[],four_visible=False,four_ellipses=False,four_frequencies=False,thrust_ratio=None,sum_frequency_squared=None,differential_proxy=0.,frequency_quality=0.)
def observation(position,geometry):
    result=copy.deepcopy(position);result['geometry_quality']=geometry.get('geometry_quality',0.) if geometry.get('detected') else 0.;return result
def associate(sensed,position):
    result=copy.deepcopy(sensed);rotors=result['rotors'];box=position.get('bbox') if position.get('detected') else None
    if box is None:inside=[]
    else:
        b=np.array(box,float);margin=(b[2:]-b[:2])*PARAMS['association_margin_fraction'];low=b[:2]-margin;high=b[2:]+margin
        inside=[r for r in rotors if np.all(np.array([r['cx'],r['cy']])>=low) and np.all(np.array([r['cx'],r['cy']])<=high)]
    missing=len(rotors)-len(inside);result['rotors']=inside
    if missing or len(inside)<4:
        result.update(four_frequencies=False,thrust_ratio=None,sum_frequency_squared=None,frequency_quality=0.)
    result['four_visible']=len(inside)==4 and all(r['observed_now'] for r in inside);result['four_ellipses']=result['four_visible'] and all(r['mode']=='ellipse' for r in inside)
    return result,dict(tracks_before=len(rotors),tracks_inside=len(inside),dropped=missing,position_current=bool(position.get('detected')))
