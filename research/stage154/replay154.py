"""One fixed curved-rollout comparison. Uses no new motion observations."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
from turn154 import forecast,PARAMS
OUT=ROOT.parent/'output/stage154_curved_rollout_20261024';OUT.mkdir(exist_ok=True)
summary={}
for name,label in [('side','dense_labels_side_110_1195'),('train2','dense_labels_train2_40_46')]:
    rows=[json.loads(x) for x in (ROOT.parent/'output/stage149_inertia_scale_20261024'/name/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    by_t={round(r['t'],6):r for r in rows};refs=json.loads((ROOT.parent/'output'/label/'labels.json').read_text(encoding='utf-8'))['records'].values()
    paired={str(h):[] for h in (.1,.2,.4)}
    for ref in refs:
        if ref.get('state')!='present':continue
        truth=(np.array(ref['box'][:2])+ref['box'][2:])/2
        for h in (.1,.2,.4):
            r=by_t.get(round(ref['t']-h,6))
            if r is None or r['coefficients'].get('optimized_geometry') is None:continue
            c=r['coefficients']['optimized_geometry'];base=r['predictions']['optimized_geometry'][str(h)];new=forecast(c,h)
            paired[str(h)].append(dict(t=ref['t'],baseline_error=float(np.linalg.norm(np.asarray(base)-truth)),
                turn_error=float(np.linalg.norm(np.asarray(new)-truth)),prediction=new,published=bool(r['valid'])))
    summary[name]={h:dict(n=len(v),baseline=float(np.mean([q['baseline_error'] for q in v])),
        turn=float(np.mean([q['turn_error'] for q in v])),p90_baseline=float(np.percentile([q['baseline_error'] for q in v],90)),
        p90_turn=float(np.percentile([q['turn_error'] for q in v],90))) for h,v in paired.items()}
    (OUT/(name+'_paired.json')).write_text(json.dumps(paired,indent=2),encoding='utf-8')
(OUT/'scores154.json').write_text(json.dumps(dict(parameters=PARAMS,scores=summary,coefficient_replay_only=True),indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
