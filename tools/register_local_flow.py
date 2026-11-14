"""Register the cumulative flow candidate after fixed replay guards."""
from pathlib import Path
import copy,json
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;OUT=WORK/'output/local_displacement_crop_20261026'
r=json.loads((OUT/'comparison.json').read_text('utf-8'));policy=json.loads((ROOT/'configs/accumulation_policy.json').read_text('utf-8'))
failures=[]
for case in ['side','train2']:
    for horizon,scores in r['results'][case]['scores'].items():
        for field,key in [('mean','mean_regression_allowance'),('p95','p95_regression_allowance')]:
            rule=policy['accuracy'][key];a=scores['mainline_v1'][field];b=scores['klt'][field]
            if b>a+max(rule['absolute_px'],rule['relative_fraction']*a):failures.append(f'{case}:{horizon}:{field}')
if r['development_macro04s']['klt']>=r['development_macro04s']['mainline_v1']:failures.append('no_development_macro_gain')
assert not failures,failures
path=ROOT/'configs/live_tracks.json';m=json.loads(path.read_text('utf-8'));assert m['recommended']=='accumulated_v1'
for track in m['tracks'].values():track['parameters'].setdefault('local_flow','none')
v=copy.deepcopy(m['tracks']['accumulated_v1']);v['parameters']['local_flow']='klt'
v.update(status='candidate_fixed_replay_pass_pending_actual_RAW',parent='accumulated_v1',
    evidence=str(OUT/'comparison.json'),note='Qualified local ON/OFF feature displacement modifies rollout velocity only; no extra neural model.')
m['tracks']['accumulated_v4_local_flow']=v
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'retention_decision.json').write_text(json.dumps(dict(candidate='accumulated_v4_local_flow',automatic_replay_policy_pass=True,
    guard_failures=failures,macro04s_gain_px=r['development_macro04s']['mainline_v1']-r['development_macro04s']['klt'],
    original_and_v1_preserved=True,default_promoted=False,actual_RAW_required=True),indent=2),encoding='utf-8')
print('registered accumulated_v4_local_flow; default remains accumulated_v1 until actual RAW')
