"""Post-inference fixed-query ablation; no interpolation of sparse references."""
from pathlib import Path
import sys,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from physical_publication import evaluate
from compare_final_pipeline import reference_at,stats
from nn_rotor_fusion import ARMS

BASE=ROOT.parent/'output';OUT=BASE/'nn_rotor166_20261105'
QUERIES={'side':np.arange(110.4,114.0001,.2),'train2':np.arange(43.2,44.4001,.2),
    'clear':np.array([26.908891573668,27.583922299469,28.258953025269,28.908982613077,29.584013338878,30.234042926686])-.4}


def load(path):return list(map(json.loads,path.read_text('utf-8').splitlines()))


def score(rows,queries,truth,sparse=False,evaluator=evaluate):
    records=[]
    for query in queries:
        available=[r for r in rows if r['t']<=query+1e-9];target_t=float(query+.4)
        record=dict(query_t=float(query),target_t=target_t,valid=False)
        if sparse:
            exact=[r for r in truth if abs(r['t']-target_t)<1e-7 and r.get('state')=='present' and r.get('box')]
            target=(np.asarray(exact[0]['box'][:2])+exact[0]['box'][2:])/2 if exact else None
        else:target=reference_at(truth,target_t)
        if not available or target is None:
            records.append(dict(record,reason='missing_source_or_reference'));continue
        row=available[-1];age=query-row['t'];good=bool(row['valid'] and age<=.04)
        record.update(valid=good,source_t=row['t'],source_age_s=float(age),mode=(row.get('structure') or {}).get('mode'),
            current_force=bool(row['force'].get('active')),reason='scored' if good else 'invalid_or_old_source')
        if good:
            prediction=np.asarray(evaluator(row,[age+.4])[str(float(age+.4))])
            record.update(error_px=float(np.linalg.norm(prediction-target)),prediction=prediction.tolist(),target=target.tolist())
        records.append(record)
    return dict(eligible=len(queries),available=sum(r['valid'] for r in records),
        error=stats([r['error_px'] for r in records if r['valid']]),records=records)


def paired(a,b):
    pairs=[(x,y) for x,y in zip(a['records'],b['records']) if x['valid'] and y['valid']]
    delta=[x['error_px']-y['error_px'] for x,y in pairs]
    return dict(common=len(pairs),mean_reduction_px=float(np.mean(delta)) if delta else None,
        improved=sum(d>1e-6 for d in delta),worsened=sum(d< -1e-6 for d in delta),unchanged=sum(abs(d)<=1e-6 for d in delta),
        baseline_common=stats([x['error_px'] for x,y in pairs]),candidate_common=stats([y['error_px'] for x,y in pairs]))


if __name__=='__main__':
    result={};protocol=json.loads((OUT/'protocol.json').read_text('utf-8'))
    assert 'source_changed_during_run' in protocol and not protocol['source_changed_during_run'],'Wait for complete inference'
    for clip,queries in QUERIES.items():
        if clip=='clear':
            with zipfile.ZipFile(ROOT.parent/'archive/output_history_20261028.zip') as z:
                data=json.loads(z.read('paper_reproduction_v030/evaluation_data/train6.json').decode('utf-8'))
            truth=[dict(r,box=r.get('bbox')) for r in data['labels']]
        else:
            name='dense_labels_side_110_1195' if clip=='side' else 'dense_labels_train2_40_46'
            data=json.loads((BASE/name/'labels.json').read_text('utf-8'));truth=list(data['records'].values())
        truth=sorted(truth,key=lambda r:r['t']);arms={};positions={};rows_by_arm={}
        for arm in ARMS:
            rows_by_arm[arm]=load(OUT/f'{clip}_{arm}.jsonl');arms[arm]=score(rows_by_arm[arm],queries,truth,sparse=clip=='clear')
        old_path=Path(protocol['clips'][clip]['original_NN_run'])/'predictions.jsonl';old=load(old_path)
        reproduction=[]
        for r in old:
            if r.get('coefficient') is not None and '0.4' in r['physical']:
                reproduction.append(float(np.linalg.norm(np.asarray(evaluate(r,[.4])['0.4'])-r['physical']['0.4'])))
        assert max(reproduction,default=0.)<1e-6,('Legacy rollout unsupported',clip,max(reproduction))
        arms['saved_old_NN_pipeline']=score(old,queries,truth,sparse=clip=='clear')
        # Every arm must consume exactly the same original NN observations.
        for arm,rows in rows_by_arm.items():
            assert [r['t'] for r in rows]==[r['t'] for r in rows_by_arm['NN_motion']]
            assert [r['original_NN_position'] for r in rows]==[r['original_NN_position'] for r in rows_by_arm['NN_motion']]
        result[clip]=dict(arms=arms,comparisons={
            'geometry_vs_motion':paired(arms['NN_motion'],arms['NN_geometry']),
            'full_vs_motion':paired(arms['NN_motion'],arms['NN_visible_force']),
            'full_vs_geometry':paired(arms['NN_geometry'],arms['NN_visible_force']),
            'visibility_vs_four_only':paired(arms['NN_four_force'],arms['NN_visible_force']),
            'full_vs_saved_old_pipeline':paired(arms['saved_old_NN_pipeline'],arms['NN_visible_force'])},
            original_NN_observations_identical=True,legacy_saved_rollout_max_difference_px=max(reproduction,default=0.))
        print(clip,json.dumps({k:v['error'] for k,v in arms.items()}),flush=True)
    summary=dict(clips=result,source_time_only=True,actual_publication_or_full_pipeline_FPS_measured=False,
        sparse_clear_targets_exact_not_interpolated=True,independent_heldout_test=False,parameter_sweep=False,
        positive_gain_is_usefulness_evidence_not_a_novelty_proof=True)
    (OUT/'scores.json').write_text(json.dumps(summary,indent=2),'utf-8')
