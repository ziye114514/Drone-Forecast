"""Score stored force-only modifications after inference; all fixed queries."""
from pathlib import Path
import sys,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from score_nn_rotor166 import QUERIES,load,score,paired
from visible_force_dynamics import evaluate_visible

BASE=ROOT.parent/'output';OUT=BASE/'visible_force167_20261105/source_gain';PREV=BASE/'nn_rotor166_20261105'
if __name__=='__main__':
    info=json.loads((OUT/'summary.json').read_text('utf-8'));assert not info['source_changed_during_run'];summary={}
    for clip in ['clear','side','train2','pure_side','pure_train2','pure_best_side','pure_best_side_2px']:
        if clip=='clear':
            with zipfile.ZipFile(ROOT.parent/'archive/output_history_20261028.zip') as z:d=json.loads(z.read('paper_reproduction_v030/evaluation_data/train6.json').decode('utf-8'))
            truth=[dict(r,box=r.get('bbox')) for r in d['labels']]
        else:
            name='dense_labels_side_110_1195' if 'side' in clip else 'dense_labels_train2_40_46'
            truth=list(json.loads((BASE/name/'labels.json').read_text('utf-8'))['records'].values())
        truth=sorted(truth,key=lambda r:r['t']);queries=np.arange(110.4,112.0001,.2) if clip in ('pure_side','pure_best_side','pure_best_side_2px') else QUERIES[clip.replace('pure_','')]
        arms={}
        for mode in ['zero','normal_only','full_fixed','full_learned','normal_only_learned','rate_only_learned']:
            arms[mode]=score(load(OUT/f'{clip}_{mode}.jsonl'),queries,truth,sparse=clip=='clear',evaluator=evaluate_visible)
        old_path=Path(info[clip]['source']) if clip.startswith('pure_') else PREV/f'{clip}_preserved_legacy_exact.jsonl'
        arms['old_full']=score(load(old_path),queries,truth,sparse=clip=='clear')
        summary[clip]=dict(arms=arms,comparisons=dict(learned_vs_zero=paired(arms['zero'],arms['full_learned']),
            learned_vs_old=paired(arms['old_full'],arms['full_learned']),learned_vs_fixed=paired(arms['full_fixed'],arms['full_learned']),
            rate_at_fixed_gain=paired(arms['normal_only'],arms['full_fixed']),
            rate_at_learned_gain=paired(arms['normal_only_learned'],arms['full_learned']),
            direction_at_learned_gain=paired(arms['rate_only_learned'],arms['full_learned'])))
        print(clip,json.dumps({k:v['error'] for k,v in arms.items()}),flush=True)
    (OUT/'scores.json').write_text(json.dumps(summary,indent=2),'utf-8')
