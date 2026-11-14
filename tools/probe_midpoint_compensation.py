"""Single fixed causal mean-compensation probe; not a deployment recipe.

Reuses the fixed observation/publication scorer. Joint covariance is not yet
propagated: this probe tests the temporal mechanism before implementing it.
"""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'tools/score_flow_acceleration.py').read_text('utf-8')
source=source.replace('import json,sys,time','import json,sys,time,copy')
source=source.replace("OUT=WORK/'output/flow_acceleration_20261026';assert not (OUT/'comparison.json').exists()",
    "OUT=WORK/'output/midpoint_compensation_20261026';OUT.mkdir(exist_ok=False)")
source=source.replace("NAMES=['v5','flow_derivative','position_consistent']",
    "NAMES=['v5','flow_derivative','position_consistent','midpoint_consistent']")
source=source.replace('v5,_=velocity_coefficient(', 'v5,velocity_audit=velocity_coefficient(')
source=source.replace('        details.append(dict(t=t,', '''        joint=copy.deepcopy(states[key]['position_consistent'])
        active=audit['position_consistent'].get('applied') and velocity_audit.get('applied')
        if active and velocity_audit['baseline_weight']==0.:
            lag=t-observation['midpoint_t'];assert 0.<=lag<=.08
            joint['velocity']=(np.asarray(observation['velocity'])+lag*np.asarray(joint['acceleration'])).tolist()
            counts['midpoint_consistent']['updated_midpoint_compensation']+=1
        else:counts['midpoint_consistent']['unchanged']+=1
        states[key]['midpoint_consistent']=joint
        details.append(dict(t=t,''')
source=source.replace('parameters_swept=False,RAW_or_GPU_started=False,globally_blind=False,',
    'parameters_swept=False,RAW_or_GPU_started=False,globally_blind=False,joint_covariance_propagated=False,production_promoted=False,')
source=source.replace('current_p_v_validity_unchanged=True',
    'current_position_validity_unchanged=True,midpoint_variant_velocity_changed=True')
source=source.replace("scope='One fixed velocity-derivative fit and a position-consistency-qualified variant. Same current v5 observations/publications; no new timed RAW deployment.'",
    "scope='Fixed midpoint compensation using the already qualified new acceleration; current position/eligibility unchanged. Mean-only causal mechanism probe: joint covariance remains unimplemented, no production promotion.'")
exec(compile(source,str(ROOT/'tools/score_flow_acceleration.py'),'exec'))
