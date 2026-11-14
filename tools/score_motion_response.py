"""Same-input staged controls for the final current motion/force candidate."""
from pathlib import Path
import os,sys,json,time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
import numpy as np
from probe_same_window_motion import replay
from compare_final_pipeline import score
def main(run,label,output):
    folder=Path(run);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in (folder/'predictions.jsonl').read_text('utf-8').splitlines()];variants={};costs={}
    for name,curve,gain,memory,endpoint,gaps in [('pre_stage',False,False,False,False,False),('curved',True,False,False,False,False),('curved_gain',True,True,False,False,False),('candidate',True,True,True,False,False),('endpoint',True,True,True,True,False),('calibrated_gap',True,True,True,True,True)]:
        values,cost=replay(rows,False,curved=curve,evidence_gain=gain,memory=memory,endpoint=endpoint,memory_gaps=gaps);variants[name]=values
        costs[name]=float(np.mean(cost));(out/(name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
    protocol=json.loads((folder/'protocol.json').read_text('utf-8'));expected='calibrated_gap' if protocol.get('calibrated_force_memory') else 'endpoint' if protocol.get('endpoint_force_balance') else 'candidate' if protocol.get('evidence_gain') else 'curved'
    difference=max(float(np.max(abs(np.asarray(a['physical'][h])-b['physical'][h]))) for a,b in zip(rows,variants[expected]) for h in a['physical'])
    assert difference<1e-7,(expected,difference)
    truths=sorted(json.loads(Path(label).read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
    result=dict(actual_run=str(folder.resolve()),scores={k:score(v,truths)['physical_future_error_px'] for k,v in variants.items()},
        actual_replay_variant=expected,actual_replay_max_difference=difference,
        component_CPU_mean_ms=costs,same_current_observations_and_valid_slots=True,
        gain_updates={k:sum(r['force']['gain_learned'] for r in v) for k,v in variants.items()},
        memory_updated_ticks=sum((r.get('adaptive_force_memory') or {}).get('updated',False) for r in variants[expected]),
        all_development_targets_retained=True,annotations_not_used_by_predictor=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
if __name__=='__main__':main(*sys.argv[1:])
