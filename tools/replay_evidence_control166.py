"""One matched-availability control; do not rerun native GPU extraction."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]
from replay_gated_nn166 import run,guard
from run_rotor_only import serial

OUT=ROOT.parent/'output/nn_rotor166_20261105'
if __name__=='__main__':
    sys.addaudithook(guard);info=json.loads((OUT/'summary.json').read_text('utf-8'));summary={}
    names=('nn_rotor_gate.py','nn_rotor_residual.py','tools/replay_evidence_control166.py')
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    snapshot=OUT/'evidence_source_snapshot';snapshot.mkdir(exist_ok=False)
    for n in names:
        p=snapshot/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    for clip,config in info.items():
        observations=list(map(json.loads,(OUT/f'{clip}_observations.jsonl').read_text('utf-8').splitlines()))
        old={r['t']:r for r in map(json.loads,(Path(config['original_NN_run'])/'predictions.jsonl').read_text('utf-8').splitlines())}
        rows,cost=run(old,observations,'native_evidence_only')
        prefix,_=run(old,observations[:len(observations)//2],'native_evidence_only')
        assert [r['physical'] for r in prefix]==[r['physical'] for r in rows[:len(prefix)]]
        full=list(map(json.loads,(OUT/f'{clip}_gated_native_visible.jsonl').read_text('utf-8').splitlines()))
        assert [r['rotor_residual_fusion']['current_evidence'] for r in rows]==[r['rotor_residual_fusion']['current_evidence'] for r in full]
        (OUT/f'{clip}_gated_native_evidence_only.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rows),'utf-8')
        g=[r['rotor_residual_fusion'] for r in rows]
        summary[clip]=dict(applied=sum(r['applied'] for r in g),updates=sum(r['updated'] for r in g),max_weight=max(r['weight'] for r in g),
            mean_ms=sum(cost)/len(cost),availability_identical_to_full=True,prefix_invariant=True)
        print(clip,json.dumps(summary[clip]),flush=True)
    summary['source_sha256']=hashes
    summary['source_changed_during_run']=[n for n,h in hashes.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    assert not summary['source_changed_during_run']
    (OUT/'evidence_summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
