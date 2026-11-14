"""Fixed development inputs; online response learned only from matured observations."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from adaptive_force_memory import AdaptiveForceMemory
from compare_final_pipeline import score
def main(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);base=ROOT.parent/'output';result={}
    for name,folder,label in [('side','RAW_side_curved8s','dense_labels_side_110_1195'),('train2','RAW_train2_curved6s','dense_labels_train2_40_46')]:
        rows=[json.loads(s) for s in (base/'motion_consistency_30px_20261028'/folder/'predictions.jsonl').read_text('utf-8').splitlines()]
        model=AdaptiveForceMemory();values=[]
        for r in rows:
            p,d=model.step(r['t'],r['coefficient'],r.get('force_observed_position'),r['physical'],r['force'],r.get('normal_turn_forecast'),r['valid'])
            values.append(dict(r,physical=p,adaptive_force_memory=d))
        (out/(name+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in values)+'\n','utf-8')
        truth=sorted(json.loads((base/label/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        result[name]=dict(scores={k:score(v,truth)['physical_future_error_px'] for k,v in [('current',rows),('adaptive_force_memory',values)]},
            learned_ticks=sum(r['adaptive_force_memory']['updated'] for r in values),last_weight=values[-1]['adaptive_force_memory']['weight'],same_valid_slots=True,RAW_FPS_measured=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
