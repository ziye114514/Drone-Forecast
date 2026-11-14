"""One fault-specific fixed-observation replay; retain control parity and all queries."""
from pathlib import Path
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from relative_rotor_observer import RelativeRotorForceObserver
from physical_publication import evaluate
from score_rotor164 import score
OUT=ROOT.parent/'output/motion172_20261107'

if __name__=='__main__':
    OUT.mkdir(exist_ok=False);summary={}
    for case,source,labels,queries in [
        ('side',ROOT.parent/'output/geometry170_20261106/batched_run/side_fixed.jsonl','dense_labels_side_110_1195',np.arange(110.4,114.0001,.2)),
        ('train2',ROOT.parent/'output/geometry170_20261106/batched_regional/train2_fixed.jsonl','dense_labels_train2_40_46',np.arange(43.2,44.4001,.2))]:
        rows=[json.loads(s) for s in source.read_text('utf-8').splitlines()];summary[case]={}
        for guard in [False,True]:
            model=RelativeRotorForceObserver();output=[];rejected=[];parity=0.
            for old in rows:
                r=copy.deepcopy(old);body=r['body'];motion=r.get('motion') or {};shift=motion.get('estimate') or {}
                if r['state'].get('two_point_initialization_selected_by_partial_entry'):model.two_point_init=True
                invalid=bool(body.get('detected') and body.get('measurement_source')=='filtered_native_translation' and shift.get('peak_at_search_boundary'))
                if guard and invalid:
                    body.update(center_provisional=True,low_confidence=True,motion_reference_advanced_without_measurement=True,
                                invalid_displacement_not_an_absolute_fix=True)
                    rejected.append(r['t'])
                meta=r['period_selection'];u=r['force']['input_u']
                c,v,f,s=model.step(r['t'],body,None if u is None else np.asarray(u),.7 if u is not None else 0.,
                    allow_gain_learning=meta.get('learning_input_current',True),input_meta=meta,motion=motion)
                r.update(coefficient=c,valid=v,force=f,state=s);r['physical']=evaluate(r,[.1,.2,.4]) if c else {}
                if not guard and c:
                    for key in ['position','velocity']:parity=max(parity,float(np.max(abs(np.asarray(c[key])-old['coefficient'][key]))))
                if model.P is not None:assert np.linalg.eigvalsh(model.P).min()>-1e-5
                output.append(r)
            if not guard:assert parity<1e-8,parity
            arm='guard' if guard else 'control';path=OUT/f'{case}_{arm}.jsonl'
            path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in output),'utf-8')
            summary[case][arm]=dict(control_parity_max=parity,rejected_cutoffs=rejected,
                score=score(path,ROOT.parent/f'output/{labels}/labels.json',queries),fixed_observation_replay_not_full_front_reexecution=True)
        print(case,{a:dict(error=v['score']['error'],available=v['score']['available'],rejected=v['rejected_cutoffs']) for a,v in summary[case].items()},flush=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
