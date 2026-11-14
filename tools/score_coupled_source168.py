from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from replay_coupled_source168 import OUT
from score_nn_rotor166 import load,score,paired

if __name__=='__main__':
    info=json.loads((OUT/'summary.json').read_text('utf-8'));assert not info['source_changed_during_run'];result={}
    for clip in ['pure_side','pure_train2']:
        name='dense_labels_side_110_1195' if clip=='pure_side' else 'dense_labels_train2_40_46'
        truth=sorted(json.loads((ROOT.parent/'output'/name/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        sets={'fixed':np.arange(110.4,112.0001,.2),'chronological_extension':np.arange(112.2,113.2001,.2)} if clip=='pure_side' else {'fixed':np.arange(43.2,44.4001,.2)}
        result[clip]={}
        for section,queries in sets.items():
            arms={a:score(load(OUT/f'{clip}_{a}.jsonl'),queries,truth) for a in ['legacy_exact','zero_matched','tracked']}
            result[clip][section]=dict(arms=arms,old_vs_tracked=paired(arms['legacy_exact'],arms['tracked']),
                numeric_value=paired(arms['zero_matched'],arms['tracked']))
            print(clip,section,json.dumps({a:v['error'] for a,v in arms.items()}),flush=True)
    (OUT/'scores.json').write_text(json.dumps(result,indent=2),'utf-8')
