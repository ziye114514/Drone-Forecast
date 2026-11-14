"""One physical route to future-force memory; no0.4s-error weight learning."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from tangential_response_memory import TangentialResponseMemory,adapt
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score
from run_rotor_only import serial

if __name__=='__main__':
    OUT=ROOT.parent/'output/tangent286_20261108';OUT.mkdir(exist_ok=False)
    files=['tangential_response_memory.py','tools/replay_tangent286.py'];hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(change='learn projected tangential velocity-dependent unknown-force response, use as conditional future memory',
        no_recording_specific_rules=True,no_parameter_search=True,source_sha256=hashes,
        source_motion_and_plane_only=True,NN_used=False,GPU_used=False,
        independent_mechanical_force_GT_unavailable=True,development=['13','15','16','11'],
        old_baseline_not_replaced=True,primary_source='https://rpg.ifi.uzh.ch/docs/RAL18_Faessler.pdf',
        distinction='paper3D rotor-drag motivation only; this own2D proxy is not its controller or parameter-identification reproduction')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    result={};checks={}
    for n in protocol['development']:
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()];memory=TangentialResponseMemory();outputs=[];parity=0.;used=0
        for old in rows:
            info=memory.observe(old);r=adapt(old,info);used+=bool(info.get('qualified'))
            control=evaluate(old,[.1,.2,.4]) if old['coefficient'] else {}
            for h,p in control.items():parity=max(parity,float(np.max(abs(np.asarray(p)-old['physical'][h]))))
            r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};outputs.append(r)
        assert parity<1e-7,parity
        path=OUT/f'{n}_tangent.jsonl';path.write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in outputs),'utf8')
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16','11':'regional212_20261108/reference11'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        result[n]=dict(control=score(source(n),ref,origins),tangent=score(path,ref,origins))
        checks[n]=dict(control_max_px=parity,qualified_memory_frames=used,last_response=outputs[-1]['tangential_response_memory'])
        print(n,{k:v['error'] for k,v in result[n].items()},'last response',checks[n]['last_response'].get('effective_response_s_inv'),flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==v for f,v in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(scores=result,checks=checks,protocol=protocol),indent=2),'utf8')
