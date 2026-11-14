"""All fixed queries; no source-only subset chosen after inspecting errors."""
from pathlib import Path
import sys,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_source_force168 import OUT,ARMS
from score_nn_rotor166 import QUERIES,load,score,paired
from visible_force_dynamics import evaluate_visible

BASE=ROOT.parent/'output'
if __name__=='__main__':
    info=json.loads((OUT/'summary.json').read_text('utf-8'));assert not info['source_changed_during_run'];summary={}
    for case in ['clear','side','train2','pure_side','pure_train2','pure_side_extension']:
        clip='pure_side' if case=='pure_side_extension' else case
        if clip=='clear':
            with zipfile.ZipFile(ROOT.parent/'archive/output_history_20261028.zip') as z:d=json.loads(z.read('paper_reproduction_v030/evaluation_data/train6.json').decode('utf-8'))
            truth=[dict(r,box=r.get('bbox')) for r in d['labels']]
        else:
            name='dense_labels_side_110_1195' if 'side' in clip else 'dense_labels_train2_40_46'
            truth=list(json.loads((BASE/name/'labels.json').read_text('utf-8'))['records'].values())
        truth=sorted(truth,key=lambda r:r['t']);queries=np.arange(110.4,112.0001,.2) if clip=='pure_side' else QUERIES[clip.replace('pure_','')]
        if case=='pure_side_extension':queries=np.arange(112.2,113.2001,.2)
        arms={a:score(load(OUT/f'{clip}_{a}.jsonl'),queries,truth,sparse=clip=='clear',evaluator=evaluate_visible) for a in ARMS}
        old_path=Path(info[clip]['source']) if clip.startswith('pure_') else BASE/'nn_rotor166_20261105'/f'{clip}_preserved_legacy_exact.jsonl'
        arms['old_full']=score(load(old_path),queries,truth,sparse=clip=='clear')
        summary[case]=dict(arms=arms,comparisons=dict(
            source_only=paired(arms['legacy_algebra'],arms['tracked_algebra']),
            combined_vs_legacy=paired(arms['legacy_algebra'],arms['tracked_innovation']),
            numeric_rotor_value=paired(arms['zero_innovation'],arms['tracked_innovation']),
            continuity_under_same_unknown=paired(arms['legacy_innovation'],arms['tracked_innovation']),
            future_frequency_trend=paired(arms['tracked_innovation_normal'],arms['tracked_innovation']),
            complete_vs_old_system=paired(arms['old_full'],arms['tracked_innovation'])))
        print(case,json.dumps({k:round(v['error']['mean'],5) if v['error']['n'] else None for k,v in arms.items()}),flush=True)
    (OUT/'scores.json').write_text(json.dumps(summary,indent=2),'utf-8')
