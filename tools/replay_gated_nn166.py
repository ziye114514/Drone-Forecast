"""Preserve original NN rollout; earn rotor correction weight causally."""
from pathlib import Path
import sys,json,hashlib,os,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src')]
from nn_rotor_gate import CausalRotorResidualGate
from nn_rotor_residual import PreservedNNRotorResidual
from run_rotor_only import serial

OUT=ROOT.parent/'output/nn_rotor166_20261105'
MODES=('native_pose','native_four','native_visible')
SOURCES=('nn_rotor_gate.py','nn_rotor_residual.py','tools/replay_gated_nn166.py',
    'adaptive_force_memory.py','physical_publication.py','rotor_thrust_bank.py',
    'fast_force_response.py','endpoint_force_balance.py','normal_turn_forecast.py')


def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0]))
        if p.name=='labels.json' or 'evaluation_data' in p.parts or p.name=='output_history_20261028.zip':
            raise RuntimeError('Annotation access forbidden during replay')


def run(old,observations,mode):
    candidate=PreservedNNRotorResidual(mode);gate=CausalRotorResidualGate();rows=[];cost=[]
    for observation in observations:
        r=old[observation['t']];tick=time.perf_counter();c=candidate.step(r,observation)
        eligible=bool(observation.get('current_pose')) if mode=='native_pose' else bool(c['force']['active'])
        out=gate.step(r,c,eligible);cost.append((time.perf_counter()-tick)*1000)
        assert out['coefficient']==r['coefficient'] and out['valid']==r['valid']
        rows.append(out)
    return rows,cost


if __name__=='__main__':
    sys.addaudithook(guard);snapshot=OUT/'gated_source_snapshot';snapshot.mkdir(exist_ok=False)
    hashes={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        hashes[name]=hashlib.sha256(b).hexdigest()
    summary={};info=json.loads((OUT/'summary.json').read_text('utf-8'))
    for clip,config in info.items():
        observations=list(map(json.loads,(OUT/f'{clip}_observations.jsonl').read_text('utf-8').splitlines()))
        old={r['t']:r for r in map(json.loads,(Path(config['original_NN_run'])/'predictions.jsonl').read_text('utf-8').splitlines())}
        entry={}
        for mode in MODES:
            rows,cost=run(old,observations,mode)
            # Re-executing only an input prefix must reproduce its predictions.
            prefix,_=run(old,observations[:len(observations)//2],mode)
            assert [r['physical'] for r in prefix]==[r['physical'] for r in rows[:len(prefix)]]
            (OUT/f'{clip}_gated_{mode}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rows),'utf-8')
            g=[r['rotor_residual_fusion'] for r in rows]
            entry[mode]=dict(frames=len(rows),applied=sum(r['applied'] for r in g),
                updates=sum(r['updated'] for r in g),max_weight=max(r['weight'] for r in g),
                mean_candidate_and_gate_ms=float(np.mean(cost)),prefix_invariant=True)
        summary[clip]=entry;print(clip,json.dumps(entry),flush=True)
    summary['source_sha256']=hashes
    summary['source_changed_during_run']=[n for n,h in hashes.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    assert not summary['source_changed_during_run']
    (OUT/'gated_summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
