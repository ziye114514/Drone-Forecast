"""Which existing future-force term causes the directional error?

Fixed2x2 ablation ONLY of the two existing rollout additions. Every estimated
current position, velocity, acceleration, geometry and measured input is kept.
This is the same force state, not a new detector or prediction model search.
"""
from pathlib import Path
import sys,json,copy,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score
from run_rotor_only import serial


def remove(row,turn=False,rate=False):
    r=copy.deepcopy(row)
    def change(packet):
        if turn and packet.get('normal_turn_forecast'):packet['normal_turn_forecast']['applied']=False
        force=packet['force']
        if rate and force.get('external_unknown_rate_px_s3') is not None:force['external_unknown_rate_px_s3']=[0.,0.]
        for component in force.get('response_model_mixture',[]):change(component)
    change(r);r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};return r


if __name__=='__main__':
    OUT=ROOT.parent/'output/future_parts267_20261108';OUT.mkdir(exist_ok=False);results={}
    for n in ['13','15','16']:
        rows=[json.loads(s) for s in source(n).read_text('utf-8').splitlines()];files={}
        for name,turn,rate in [('original',False,False),('no_future_normal_turn',True,False),('no_future_external_rate',False,True),('no_future_extras',True,True)]:
            outputs=[remove(r,turn,rate) for r in rows]
            assert all(a['coefficient']==b['coefficient'] for a,b in zip(outputs,rows))
            path=OUT/f'{n}_{name}.jsonl';path.write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in outputs),'utf-8');files[name]=path
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]={name:score(path,ref,origins) for name,path in files.items()}
        print(n,{k:v['error'] for k,v in results[n].items()},flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,current_states_identical=True,NN_used=False,GPU_used=False,
        no_new_hyperparameters=True,no_recording_specific_rules=True,heldout1_opened=False),indent=2),'utf-8')
