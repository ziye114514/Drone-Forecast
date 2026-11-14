"""Verify the old force rollout, then change only native rotor inputs/pose."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'src')]
from nn_rotor_residual import PreservedNNRotorResidual
from run_rotor_only import serial

OUT=ROOT.parent/'output/nn_rotor166_20261105'
MODES=('legacy_exact','native_four','native_visible','native_visible_no_gain')

if __name__=='__main__':
    info=json.loads((OUT/'summary.json').read_text('utf-8'));summary={}
    for clip,config in info.items():
        observations=list(map(json.loads,(OUT/f'{clip}_observations.jsonl').read_text('utf-8').splitlines()))
        all_old=list(map(json.loads,(Path(config['original_NN_run'])/'predictions.jsonl').read_text('utf-8').splitlines()))
        old={r['t']:r for r in all_old};outputs={};entry={}
        for mode in MODES:
            model=PreservedNNRotorResidual(mode);rows=[];delta=[]
            for observation in observations:
                r=old[observation['t']];p=model.step(r,observation);rows.append(p)
                if mode=='legacy_exact':
                    delta.extend(float(np.linalg.norm(np.asarray(v)-r['physical'][h])) for h,v in p['physical'].items())
                assert p['coefficient']==r['coefficient'] and p['valid']==r['valid']
            outputs[mode]=rows
            entry[mode]=dict(active=sum(r['force']['active'] for r in rows),
                disturbance_updates=sum(r['force']['disturbance_updated'] for r in rows),
                gain_updates=sum(r['force']['gain_learned'] for r in rows))
            if mode=='legacy_exact':
                entry['legacy_max_prediction_difference_px']=max(delta,default=0.)
                assert max(delta,default=0.)<1e-6,(clip,max(delta))
            (OUT/f'{clip}_preserved_{mode}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rows),'utf-8')
        summary[clip]=entry;print(clip,json.dumps(entry),flush=True)
    summary['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['nn_rotor_residual.py','tools/replay_preserved_nn166.py']}
    (OUT/'preserved_summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
