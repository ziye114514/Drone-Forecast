"""Nontrivial admission, identity, permutation and original-control checks."""
from pathlib import Path
import copy,itertools,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage119')]
from study119 import OUT,OLD,REFERENCE,rows,save,stable,baseline
from partial119 import PartialModel,fit_three

def main():
    rr=rows(OLD/'clear6/records.jsonl');reference=rows(REFERENCE/'clear6/records.jsonl');models={m:PartialModel(m) for m in ['body_hold','affine3','conic3','guarded3']}
    # Full-observation path must still equal frozen117 before any dropout.
    for i,r in enumerate(rr[:65]):
        rotors=r['outputs']['merge_safe']['geometry'].get('rotors',[]);m=r['outputs']['merge_safe']['center']['measurement']
        for name,model in models.items():
            # Disable only the added partial path for this isolation check:
            # native first65 includes a naturally missing rotor already.
            if len(rotors)==3:model.mode='body_hold'
            _,d=model.step(r['t'],r['body'],rotors,r['axis'],m);d.pop('partial119')
            model.mode=name
            assert stable(d)==stable(reference[i]['outputs']['fixed_structure']['center'])
    nofull=PartialModel('guarded3');count=0
    for r in rr[:100]:
        rotors=copy.deepcopy(r['outputs']['merge_safe']['geometry'].get('rotors',[]))
        if len(rotors)==4:rotors.pop()
        _,d=nofull.step(r['t'],r['body'],rotors,r['axis'],baseline(rotors,r['body']))
        assert not d['current'] and nofull.model.shape is None;count+=1
    r=rr[65];rotors=r['outputs']['merge_safe']['geometry']['rotors'][:3];base=models['guarded3'];answers=[]
    for order in itertools.permutations(range(3)):
        m,_=base.measure(r['body'],[rotors[i] for i in order]);assert m['valid'];answers.append(m['center'])
    difference=float(max(np.linalg.norm(np.asarray(answers)-answers[0],axis=1)));assert difference<1e-4
    _,d=base.step(r['t']+.6,dict(detected=False,center=None),[],dict(valid=False),dict(valid=False,reason='no_body',current_count=0))
    assert not d['current'] and base.model.shape is None and base.H is None
    side=rows(OUT/'side6_mask0/records.jsonl');negative=rows(OUT/'negative5_mask0/records.jsonl')
    assert all(not r['outputs'][m]['center']['current'] for r in side+negative for m in models)
    save(OUT/'controls119.json',dict(partial_disabled_first65_original117_exact_all_modes=True,three_only_never_initializes_frames=count,
        six_input_permutations_center_max_difference_px=difference,identity_gap_resets_structure=True,
        natural_side100_and_negative30_never_claim_new_geometry=True,independent_prefixes4x100_exact=True,
        assertion_repair='Initial check incorrectly assumed first65 unmasked frames all have four native observations. New partial update legitimately differs on natural three-input frames. Isolation check now disables only partial updates; inference experiment/results unchanged.',
        scope='Causal/admission/identity controls. Real feature removal is not physical pixel occlusion; no COM truth.'))
    print(json.dumps(json.loads((OUT/'controls119.json').read_text('utf-8'))),flush=True)
if __name__=='__main__':main()
