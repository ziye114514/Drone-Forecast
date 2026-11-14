"""Same learned memory and observations; fixed-up plane diagnostic control."""
from pathlib import Path
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from identifiable_tangent_memory import IdentifiableTangentMemory,adapt
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score
from run_rotor_only import serial

if __name__=='__main__':
    OUT=ROOT.parent/'output/plane_ablation291_20261108';OUT.mkdir(exist_ok=False);results={}
    for n in ['13','15','16','11','3']:
        src=ROOT.parent/'output/frozen_tangent287_20261108/control.jsonl' if n=='3' else source(n)
        rows=[json.loads(s) for s in src.read_text('utf8').splitlines()];model=IdentifiableTangentMemory();outputs=[]
        for r in rows:
            control=copy.deepcopy(r);n0=control['structure'].get('axis_xy')
            if n0 is not None:control['structure']['axis_xy']=[0.,-float(np.linalg.norm(n0))]
            info=model.observe(control);s=adapt(r,info);s['physical']=evaluate(s,[.1,.2,.4]) if s['coefficient'] else {};outputs.append(s)
        path=OUT/f'{n}_fixed_up.jsonl';path.write_text(''.join(json.dumps(r,default=serial)+'\n' for r in outputs),'utf8')
        if n=='3':ref=ROOT.parent/'output/reference290_20261108/references.json'
        else:
            folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16','11':'regional212_20261108/reference11'}[n];ref=ROOT.parent/f'output/{folder}/references.json'
        origins=np.round(np.arange(10.8,15.6001,.2),1)
        actual=ROOT.parent/f'output/identifiable289_20261108/{n}_tangent.jsonl'
        results[n]=dict(original=score(src,ref,origins),measured_plane=score(actual,ref,origins),fixed_up_control=score(path,ref,origins))
        print(n,{k:round(v['error']['mean'],3) for k,v in results[n].items()})
    (OUT/'results.json').write_text(json.dumps(dict(scores=results,scope='regression-plane ablation only; detector/force/current state unchanged; fixed-up is not selected default'),indent=2),'utf8')
