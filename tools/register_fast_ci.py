"""Register a semantics-preserving solver acceleration; no accuracy claim."""
from pathlib import Path
import copy,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/fast_ci_20261026'
r=json.loads((OUT/'comparison.json').read_text('utf-8'))
assert r['actual_coefficients_and_audits_exact'] and r['known_endpoint_and_complementary_axis_cases_exact']
assert r['timing']['dominance_shortcut']['median_ms']<r['timing']['frozen_optimizer']['median_ms']
path=ROOT/'configs/live_tracks.json';m=json.loads(path.read_text('utf-8'));assert m['recommended']=='accumulated_v4_local_flow'
v=copy.deepcopy(m['tracks']['accumulated_v4_local_flow'])
v.update(status='candidate_exact_solver_acceleration_pending_RAW',parent='accumulated_v4_local_flow',evidence=str(OUT/'comparison.json'),
    note='Same point predictions; PSD precision dominance chooses the same CI endpoint directly, otherwise original optimizer. No extra model.')
m['tracks']['accumulated_v5_fast_ci']=v
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Registered v5; default remains v4 until RAW entry smoke.')
