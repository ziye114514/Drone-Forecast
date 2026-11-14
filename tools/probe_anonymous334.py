"""Reuse existing anonymous single-ring observer to inspect detection gaps.

No full five-circle initialization, NN, label or saved phase result. This is
the existing235/298 CPU author interface, not another phase implementation.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from joint_phase_bank import JointPhaseBank
from run_rotor_only import serial

if __name__=='__main__':
    out=ROOT.parent/'output/anonymous334_20261109';out.mkdir(exist_ok=False)
    files=['joint_phase_bank.py','joint_ring_phase.py','angular_phase_candidates.py','tools/probe_anonymous334.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    protocol=dict(source_sha256=hashes,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        no_full_rotor_template_required=True,no_recording_specific_rules=True,no_parameter_search=True,
        exact_native_packets=True,current_late_validation_required=True,no_reference_read=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    src=ROOT.parent/'output/frozen324_20261109';rows=list(map(json.loads,(src/'control.jsonl').read_text('utf8').splitlines()))
    manifest=json.loads((src/'cache/manifest.json').read_text('utf8'))['frames']
    model=JointPhaseBank(ROOT.parent/'output/helix_cpu232_20261108/build/Release/helix_reference.dll',temporal_hypothesis=True)
    results=[];timing=[]
    for i,(row,packet) in enumerate(zip(rows,manifest)):
        assert abs(row['t']-packet['t'])<1e-9
        e=np.load(src/'cache'/packet['file']);tick=time.perf_counter()
        u,q,meta=model.step(row['t'],e,row['geometry'].get('rotors',[]),row['motion']);timing.append(time.perf_counter()-tick)
        r=dict(t=row['t'],u=None if u is None else u.tolist(),quality=q,meta=meta,
            original_current_detection=bool(row['body'].get('detected')),original_valid=row['valid']);results.append(r)
        with (out/'periods.jsonl').open('a',encoding='utf8') as fh:fh.write(json.dumps(r,default=serial,allow_nan=False)+'\n')
        if i%20==0:print(i,row['t'],'original detection',r['original_current_detection'],'independent phase',meta['qualified_current_rings'],flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    gained=[dict(t=r['t'],rings=r['meta']['qualified_current_rings']) for r in results if not r['original_current_detection'] and r['meta']['qualified_current_rings']]
    summary=dict(frames=len(results),original_current_detections=sum(r['original_current_detection'] for r in results),
        phase_confirmed_frames=sum(r['meta']['qualified_current_rings']>0 for r in results),
        phase_confirmed_during_detection_gap=gained,mean_CPU_phase_ms=float(1000*np.mean(timing)),
        prediction_gain_proven=False,no_force_inputs_invented=True)
    (out/'results.json').write_text(json.dumps(summary,indent=2),'utf8');print(summary,flush=True)
