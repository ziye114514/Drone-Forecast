"""Score native increments on the verified original NN motion backbone."""
from pathlib import Path
import sys,json,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from score_nn_rotor166 import QUERIES,score,paired,load

BASE=ROOT.parent/'output';OUT=BASE/'nn_rotor166_20261105'
if __name__=='__main__':
    summary={}
    for clip,queries in QUERIES.items():
        if clip=='clear':
            with zipfile.ZipFile(ROOT.parent/'archive/output_history_20261028.zip') as z:d=json.loads(z.read('paper_reproduction_v030/evaluation_data/train6.json').decode('utf-8'))
            truth=[dict(r,box=r.get('bbox')) for r in d['labels']]
        else:
            name='dense_labels_side_110_1195' if clip=='side' else 'dense_labels_train2_40_46'
            truth=list(json.loads((BASE/name/'labels.json').read_text('utf-8'))['records'].values())
        truth=sorted(truth,key=lambda r:r['t']);arms={}
        for mode in ['legacy_exact','native_four','native_visible','native_visible_no_gain']:
            arms[mode]=score(load(OUT/f'{clip}_preserved_{mode}.jsonl'),queries,truth,sparse=clip=='clear')
        summary[clip]=dict(arms=arms,comparisons=dict(
            full_vs_old=paired(arms['legacy_exact'],arms['native_visible']),
            visibility_vs_four_only=paired(arms['native_four'],arms['native_visible'])))
        print(clip,json.dumps({k:v['error'] for k,v in arms.items()}),flush=True)
    (OUT/'preserved_scores.json').write_text(json.dumps(summary,indent=2),'utf-8')
