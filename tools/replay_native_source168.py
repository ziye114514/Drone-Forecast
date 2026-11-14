"""Matched zero-input control on the actual native-cache run; no labels."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from replay_coupled_source168 import replay
from replay_gated_nn166 import guard
from run_rotor_only import serial
OUT=ROOT.parent/'output/source_force168_20261105/native_cache_tracked'
if __name__=='__main__':
    sys.addaudithook(guard);protocol=json.loads((OUT/'protocol.json').read_text('utf-8'));assert not protocol['source_changed_during_run']
    source=[json.loads(s) for s in (OUT/'predictions.jsonl').read_text('utf-8').splitlines()]
    outputs,parity=replay(source,True)
    for arm,rows in outputs.items():
        (OUT/f'replay_{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rows),'utf-8')
    (OUT/'replay_control.json').write_text(json.dumps(dict(original_prediction_parity_max_px=parity,
        NN_used=False,control_is_fixed_observation_replay_not_separate_live_run=True,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),'utf-8')
    print('Native run parity:',parity)
