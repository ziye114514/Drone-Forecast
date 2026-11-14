"""Conditional effective gain, not physical total thrust identification.

Reuses the joint observer; only validated LOCAL pooled evidence is permitted
to learn the coefficient. No parameter sweep and no new trajectory branch.
"""
from pathlib import Path
import copy,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from rotor_force_observer import RotorForceObserver
from physical_publication import evaluate
from score_rotor164 import score


def replay(source,out,learn):
    model=RotorForceObserver();result=[];qualified=0
    for old in source:
        r=copy.deepcopy(old);meta=r['period_selection'];u=r['force']['input_u']
        local=r.get('local_temporal_phase',{})
        eligible=bool(learn and local.get('qualified') and meta.get('pooled_estimator_source')=='local_pixel_first_harmonic')
        qualified+=eligible
        c,v,f,s=model.step(r['t'],r['body'],None if u is None else np.asarray(u),.7 if u is not None else 0.,
            allow_gain_learning=eligible or meta.get('learning_input_current',True),input_meta=meta)
        if not learn and c is not None:
            np.testing.assert_allclose(c['position'],r['coefficient']['position'],atol=1e-8,rtol=0)
            np.testing.assert_allclose(c['velocity'],r['coefficient']['velocity'],atol=1e-8,rtol=0)
        r.update(coefficient=c,valid=v,force=f,state=s)
        r['physical']=evaluate(r,[.1,.2,.4]) if c else {};result.append(r)
        if model.P is not None:assert np.linalg.eigvalsh(model.P).min()>-1e-5
    out.write_text(''.join(json.dumps(r)+'\n' for r in result),'utf-8')
    return dict(current_local_qualified=qualified,gain_updates=sum(r['force'].get('gain_learned',False) for r in result),
        final_gain=model.default_gain if model.x is None else float(model.x[6]))


if __name__=='__main__':
    folder=ROOT.parent/'output/native_plane165_20261104';inputs={
        'cold':ROOT.parent/'output/side_temporal164_20261101/raw_side_final/predictions.jsonl',
        'warm':ROOT.parent/'output/side_temporal164_20261101/warm_subpixel_source/predictions.jsonl'}
    result={};paths={}
    for name,path in inputs.items():
        source=[json.loads(s) for s in path.read_text('utf-8').splitlines()]
        for arm in [False,True]:
            key=f'{name}_{arm}';dest=folder/f'mixed_gain_{key}.jsonl'
            result[key]=replay(source,dest,arm);paths[key]=dest
    labels=ROOT.parent/'output/dense_labels_side_110_1195/labels.json'
    for key,path in paths.items():
        result[key].update(fixed=score(path,labels,np.arange(110.4,112.0001,.2)),
            extension=score(path,labels,np.arange(112.2,114.0001,.2)))
    (folder/'mixed_gain_scores.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps({k:{'updates':v['gain_updates'],'gain':v['final_gain'],
        'scores':{n:v[n]['error'] for n in ['fixed','extension']}} for k,v in result.items()}))
