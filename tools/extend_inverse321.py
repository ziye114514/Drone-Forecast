"""Freeze one mechanism; check five existing recordings without per-case picks."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from inverse_normal_response import InverseNormalResponse
from tangential_response_memory import adapt
from replay_joint236 import source
from physical_publication import evaluate
from score_rotor164 import score

def sources(n):
    return ROOT.parent/'output/frozen_tangent287_20261108/control.jsonl' if n=='3' else source(n)

def references(n):
    return ROOT.parent/('output/reference290_20261108/references.json' if n=='3' else f'output/'+{
        '13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15',
        '16':'unseen191_20261107/reference16','11':'regional212_20261108/reference11'}[n]+'/references.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='inverse326_20261109');args=parser.parse_args()
    out=ROOT.parent/'output'/args.batch;out.mkdir(exist_ok=False)
    files=['inverse_normal_response.py','identifiable_tangent_memory.py','tangential_response_memory.py','physical_publication.py','tools/extend_inverse321.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    protocol=dict(no_recording_specific_rules=True,no_parameter_search=True,NN_used=False,RAW_FPS_measured=False,
        all_cases_previously_seen_development=True,no_new_heldout_claim=True,all_inference_before_reference_reads=True,
        baseline_and_current_state_preserved=True,source_sha256=hashes)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8');cases={};parity={}
    for n in ('13','15','16','11','3'):
        rows=[json.loads(s) for s in sources(n).read_text('utf8').splitlines()];m=InverseNormalResponse();outputs={k:[] for k in ('control','tangent289','inverse320','combined')};delta=0.
        for row in rows:
            inverse=m.step(row);info=m.last_tangent_info
            packet=dict(control=row,tangent289=adapt(row,info),inverse320=inverse,combined=adapt(inverse,info))
            for arm,r in packet.items():
                r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};outputs[arm].append(r)
                assert r['coefficient']==row['coefficient'];assert r['force']['acceleration_px_s2']==row['force']['acceleration_px_s2']
        for arm,rr in outputs.items():(out/f'{n}_{arm}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr),'utf8')
        print(n,'all inference frozen',flush=True)
    for n in ('13','15','16','11','3'):
        origins=np.round(np.arange(10.8,15.6001,.2),1);ref=references(n)
        cases[n]={a:score(out/f'{n}_{a}.jsonl',ref,origins) for a in outputs};print(n,{a:v['error'] for a,v in cases[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (out/'results.json').write_text(json.dumps(dict(cases=cases,protocol=protocol),indent=2),'utf8')
