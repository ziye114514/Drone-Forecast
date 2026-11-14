"""One frozen response fit: aligned vs zero vs causal past-delayed input."""
from pathlib import Path
import sys,json,hashlib,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from calibrated_force_head import features_at,fit_head,predict_head
OUT=ROOT.parent/'output/decision169_20261105'
ARMS=('aligned','zero','past_delayed')


def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0]))
        if p.name in ('heldout_targets.json','labels.json') or 'evaluation_data' in p.parts or p.name.endswith('.zip'):
            raise RuntimeError('Test or original coordinate targets forbidden during fit/prediction')


if __name__=='__main__':
    sys.addaudithook(guard);snapshot=OUT/'head_source_snapshot';snapshot.mkdir(exist_ok=False)
    names=['calibrated_force_head.py','tools/run_decision169.py'];hashes={}
    for n in names:
        data=(ROOT/n).read_bytes();p=snapshot/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);hashes[n]=hashlib.sha256(data).hexdigest()
    schedule=json.loads((OUT/'schedule.json').read_text('utf-8'));train=json.loads((OUT/'calibration_targets.json').read_text('utf-8'))
    models={};predictions={};features_hash={}
    for name,s in schedule.items():
        path=OUT/f'evidence_v2/{name}_features.jsonl';rows=list(map(json.loads,path.read_text('utf-8').splitlines()));models[name]={};predictions[name]={}
        features_hash[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        usable=[r for r in train[name] if features_at(rows,r['query_t']) is not None]
        for arm in ARMS:
            delay=.4 if arm=='past_delayed' else 0.
            records=[features_at(rows,r['query_t'],delay) for r in usable]
            model=fit_head(records,[r['target'] for r in usable],zero=arm=='zero');models[name][arm]=model
            outputs=[]
            for q in s['test_queries']:
                r=features_at(rows,q,delay)
                outputs.append(dict(query_t=q,target_t=q+.4,valid=r is not None,
                    prediction=None if r is None else predict_head(r,model,zero=arm=='zero'),
                    baseline=None if r is None else r['baseline'],source_t=None if r is None else r['source_t'],
                    input_u=None if r is None else r['input_u']))
            predictions[name][arm]=outputs
        print(name,json.dumps({a:dict(gain=m['gain'],train=m['training_examples']) for a,m in models[name].items()}),flush=True)
    (OUT/'frozen_models.json').write_text(json.dumps(models,indent=2),'utf-8')
    (OUT/'heldout_predictions.json').write_text(json.dumps(predictions,indent=2),'utf-8')
    changed=[n for n,h in hashes.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h];assert not changed
    (OUT/'fit_completion.json').write_text(json.dumps(dict(source_sha256=hashes,features_sha256=features_hash,
        source_changed_during_run=changed,manual_calibration=True,heldout_target_read=False,NN_trained=False,
        zero_and_delayed_controls_fixed=True,parameters_swept=False,test_after_freeze=True),indent=2),'utf-8')
