"""Single mechanism trial on fixed current observations; labels after replay."""
from pathlib import Path
import sys,copy,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from rotor_force_observer import RotorForceObserver
from relative_rotor_observer import RelativeRotorForceObserver
from physical_publication import evaluate
from score_rotor164 import score
from score_publication165 import score as publication_score

if __name__=='__main__':
    out=ROOT.parent/'output/native_plane165_20261104';results={}
    for name,source in [('cold','raw_native_join_nodrop'),('warm','raw_entry_fast'),('bounded','raw_native_join_paced')]:
        rows=list(map(json.loads,(out/source/'predictions.jsonl').read_text('utf-8').splitlines()))
        paths={}
        for relative in [False,True]:
            model=(RelativeRotorForceObserver if relative else RotorForceObserver)();output=[];used=0
            for old in rows:
                r=copy.deepcopy(old);meta=r['period_selection'];u=r['force']['input_u']
                if r['state'].get('two_point_initialization_selected_by_partial_entry'):model.two_point_init=True
                c,v,f,s=model.step(r['t'],r['body'],None if u is None else np.asarray(u),.7 if u is not None else 0.,
                    allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,
                    motion=r['motion'] if relative else None)
                if not relative and c is not None:
                    np.testing.assert_allclose(c['position'],r['coefficient']['position'],atol=1e-8,rtol=0)
                    np.testing.assert_allclose(c['velocity'],r['coefficient']['velocity'],atol=1e-8,rtol=0)
                used+=s.get('native_relative_displacement_observed',False)
                if model.P is not None:assert np.linalg.eigvalsh(model.P).min()>-1e-5
                r.update(coefficient=c,valid=v,force=f,state=s);r['physical']=evaluate(r,[.1,.2,.4]) if c else {};output.append(r)
            path=out/f'{name}_relative_{relative}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in output),'utf-8');paths[relative]=path
            results[f'{name}_{relative}']=dict(relative_updates=used)
        labels=ROOT.parent/'output/dense_labels_side_110_1195/labels.json'
        for relative,path in paths.items():
            results[f'{name}_{relative}'].update(fixed=score(path,labels,np.arange(110.4,112.0001,.2)),
                extension=score(path,labels,np.arange(112.2,114.0001,.2)))
            if name=='bounded':
                results[f'{name}_{relative}']['actual_clock_counterfactual']=publication_score(out/source,labels,np.arange(110.4,113.0001,1/30),predictions=path)
                results[f'{name}_{relative}']['actual_clock_counterfactual']['runtime_of_new_observer_not_remeasured']=True
    (out/'relative_observer_scores.json').write_text(json.dumps(results,indent=2),'utf-8')
    print(json.dumps({k:dict(updates=v['relative_updates'],fixed=v['fixed']['error'],extension=v['extension']['error']) for k,v in results.items()}))
