from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from score_nn_rotor166 import load,score,paired
OUT=ROOT.parent/'output/source_force168_20261105/native_cache_tracked'
if __name__=='__main__':
    meta=json.loads((OUT/'replay_control.json').read_text('utf-8'));assert meta['original_prediction_parity_max_px']<1e-6
    truth=sorted(json.loads((ROOT.parent/'output/dense_labels_side_110_1195/labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    queries=np.arange(110.4,112.0001,.2)
    arms={a:score(load(OUT/path),queries,truth) for a,path in [('native_actual','predictions.jsonl'),('zero_same_observations','replay_zero_matched.jsonl')]}
    result=dict(arms=arms,numeric_value=paired(arms['zero_same_observations'],arms['native_actual']),
        raw_decoding_included=False,input='native EventCD NPY packets, streaming full front and observer')
    (OUT/'scores.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps({a:v['error'] for a,v in arms.items()}))
