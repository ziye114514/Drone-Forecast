"""Apply the same force mechanism to the stronger existing pure-rotor state."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_force167 import replay,pure_observation,OUT
from replay_gated_nn166 import guard
from run_rotor_only import serial

if __name__=='__main__':
    sys.addaudithook(guard)
    source=ROOT.parent/'output/native_plane165_20261104/cold_relative_True.jsonl'
    base=[json.loads(s) for s in source.read_text('utf-8').splitlines()]
    outputs,entry=replay(base,[pure_observation(r) for r in base],True)
    prefix,_=replay(base[:len(base)//2],[pure_observation(r) for r in base[:len(base)//2]],True)
    for mode,rows in outputs.items():
        assert [r['physical'] for r in prefix[mode]]==[r['physical'] for r in rows[:len(prefix[mode])]]
        (OUT/f'pure_best_side_2px_{mode}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rows),'utf-8')
    summary=json.loads((OUT/'summary.json').read_text('utf-8'))
    summary['pure_best_side_2px']=dict(arms=entry,NN_used=False,source=str(source),prefix_invariant=True,
        additional_sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['visible_force_dynamics.py','tools/run_force167.py','tools/probe_force167_best.py']})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
    print(json.dumps(entry))
