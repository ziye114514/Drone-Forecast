"""Register the same-value overlap candidate after paired image mechanism work."""
from pathlib import Path
import copy,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent/'output/support_overlap_20261026'
r=json.loads((OUT/'comparison.json').read_text('utf-8'))
assert r['all_NN_boxes_and_support_hashes_exact']
assert r['summary']['gpu-overlap']['median_ms']<r['summary']['serial']['median_ms']
p=ROOT/'configs/live_tracks.json';m=json.loads(p.read_text('utf-8'));assert m['recommended']=='accumulated_v5_fast_ci'
for v in m['tracks'].values():v['parameters'].setdefault('flow_preparation','serial')
v=copy.deepcopy(m['tracks']['accumulated_v5_fast_ci']);v['parameters']['flow_preparation']='gpu-overlap'
v.update(parent='accumulated_v5_fast_ci',status='candidate_matched_image_overlap_pending_RAW',evidence=str(OUT/'comparison.json'),
    note='Same current native support, prepared after sending GPU request before receiving its reply. No future frame or prediction formula change.')
m['tracks']['accumulated_v6_support_overlap']=v
p.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Registered v6; recommended remains v5 pending actual RAW.')
