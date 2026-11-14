"""One completeness-conditioned mechanical scale bridge, no label tuning."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from full_rotor_response_bridge import FullRotorResponseBridge
from rotor_torque_innovation import RotorTorqueInnovation,adapt
from continuous_phase_force import torque_period
from physical_publication import evaluate
from replay_joint236 import source
from score_rotor164 import score

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase-batch',default='continuous305_20261109');p.add_argument('--output-batch',default='response312_20261109');p.add_argument('--redistribution-only',action='store_true');a=p.parse_args()
    OUT=ROOT.parent/'output'/a.output_batch;OUT.mkdir(exist_ok=False)
    files=['full_rotor_response_bridge.py','rotor_torque_innovation.py','rotor_frequency_drive.py','tools/replay_full_response312.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    for f in files:
        d=OUT/'source_snapshot'/f;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes((ROOT/f).read_bytes())
    protocol=dict(source_sha256=hashes,phase_batch=a.phase_batch,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        no_parameter_search=True,no_recording_specific_rules=True,inference_before_any_reference_reads=True,
        current_position_velocity_acceleration_unchanged=True,hidden_frequency_or_force_not_filled=True,
        gain_from_estimated_image_response_not_true_motor_force=True,only_four_current_sources_calibrate=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8');results={};diagnostics={}
    for n in ('13','15','16'):
        rows=[json.loads(s) for s in source(n).read_text('utf8').splitlines()]
        folder=ROOT.parent/'output'/a.phase_batch/f'recording{n}'
        pp=folder/('continuous_periods.jsonl' if (folder/'continuous_periods.jsonl').exists() else 'periods.jsonl')
        periods=[json.loads(s) for s in pp.read_text('utf8').splitlines()]
        bridge=FullRotorResponseBridge(redistribution_only=a.redistribution_only);single=RotorTorqueInnovation(frequency_changes_only=True,redistribution_only=a.redistribution_only);out={k:[] for k in ('bridge','default_scale')}
        for row,period in zip(rows,periods):
            period=torque_period(row,period);r=bridge.step(row,period);s=adapt(row,single.observe(row,period))
            for k,v in [('bridge',r),('default_scale',s)]:
                v['physical']=evaluate(v,[.1,.2,.4]) if v['coefficient'] else {};out[k].append(v)
        for k,rr in out.items():(OUT/f'{n}_{k}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr),'utf8')
        infos=[r['full_rotor_response_bridge'] for r in out['bridge']]
        diagnostics[n]=dict(calibration_updates=sum(i['calibration_updated'] for i in infos),
            calibrated_parameter_frames=sum(i['calibration'] is not None for i in infos),
            last_calibration=next((i['calibration'] for i in infos[::-1] if i['calibration'] is not None),None))
        print(n,diagnostics[n],flush=True)
    for n in ('13','15','16'):
        folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
        ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
        results[n]={k:score(OUT/f'{n}_{k}.jsonl',ref,origins) for k in out}
        print(n,{k:v['error'] for k,v in results[n].items()},flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(dict(cases=results,diagnostics=diagnostics,protocol=protocol),indent=2),'utf8')
