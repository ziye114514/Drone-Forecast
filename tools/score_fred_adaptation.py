"""Score isolated adapted detector after RAW outputs are frozen."""
from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'tools')]
from prepare_fred_transfer import parse,OUT as FRED
from score_fred_transfer import score,reference
from compare_final_pipeline import stats,score as manual_score,reference_at
OUT=ROOT.parent/'output/fred_domain_adaptation_20261029'


def matched(a,b,target_at):
    last=None;origins=[]
    for r in a:
        slot=int(np.floor(r['t']*5+1e-8))
        if not r['valid'] or slot==last:continue
        last=slot;truth=target_at(r['t']+.4)
        if truth is not None:origins.append((r,truth))
    records=[]
    for r,truth in origins:
        s=min(b,key=lambda s:abs(s['t']-r['t']))
        if abs(s['t']-r['t'])>1e-7:raise ValueError('Time grids differ')
        records.append(dict(t=r['t'],available=bool(s['valid']),
            old_error_px=float(np.linalg.norm(np.array(r['physical']['0.4'])-truth)),
            new_error_px=None if not s['valid'] else float(np.linalg.norm(np.array(s['physical']['0.4'])-truth))))
    pairs=[r for r in records if r['available']]
    return dict(original_origins=len(origins),new_available=len(pairs),
        old_matched_error=stats([r['old_error_px'] for r in pairs]),
        new_matched_error=stats([r['new_error_px'] for r in pairs]),records=records)


def main():
    suffix=sys.argv[1] if len(sys.argv)>1 else 'adapted'
    if not suffix.replace('_','').isalnum():raise ValueError('Invalid suffix')
    protocol=json.loads((FRED/'protocol.json').read_text());result={}
    for s in protocol['sequences']:
        n=s['sequence'];run=OUT/f'RAW_{n}_{suffix}'
        if not (run/'summary.json').exists():continue
        rows=[json.loads(x) for x in (run/'predictions.jsonl').read_text().splitlines()]
        prior=FRED/'motion_observability_v3'/f'fred{n}_motion.jsonl'
        old=[json.loads(x) for x in prior.read_text().splitlines()]
        labels=parse(Path(s['annotation']).read_text('utf-8-sig'))
        def target(t):
            b=reference(labels,t)
            return None if b is None else (b[:2]+b[2:])/2
        result[str(n)]=dict(score=score(rows,labels,s['start'],s['duration']),matched=matched(old,rows,target),
            runtime=json.loads((run/'summary.json').read_text()))
    for name,folder,truth in [('other',f'RAW_train2_{suffix}6s','dense_labels_train2_40_46'),
        ('side',f'RAW_side_{suffix}8s','dense_labels_side_110_1195')]:
        run=OUT/folder
        if not (run/'summary.json').exists():continue
        rows=[json.loads(x) for x in (run/'predictions.jsonl').read_text().splitlines()]
        old=[json.loads(x) for x in (FRED/'motion_observability_v3'/f'{name}_motion.jsonl').read_text().splitlines()]
        labels=sorted(json.loads((ROOT.parent/'output'/truth/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        result[name]=dict(score=manual_score(rows,labels),matched=matched(old,rows,lambda t:reference_at(labels,t)),
            runtime=json.loads((run/'summary.json').read_text()))
    (OUT/('scores.json' if suffix=='adapted' else f'scores_{suffix}.json')).write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:{a:b for a,b in v['matched'].items() if a!='records'} for k,v in result.items()},indent=2))


if __name__=='__main__':main()
