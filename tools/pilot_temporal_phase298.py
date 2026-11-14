"""Past per-source frequency seeds a second CURRENT discovery fit; late frozen."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from joint_phase_bank import JointPhaseBank
from rotor_torque_partition import RotorTorquePartition
from rotor_torque_response import adapt
from replay_joint236 import source
from pure_visibility170 import window
from physical_publication import evaluate
from score_rotor164 import score

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recording',required=True,choices=('13','15','16'));args=parser.parse_args();n=args.recording
    OUT=ROOT.parent/f'output/temporal298_20261109/recording{n}';OUT.mkdir(parents=True,exist_ok=False)
    files=['joint_phase_bank.py','joint_ring_phase.py','rotor_torque_partition.py','rotor_torque_response.py','tools/pilot_temporal_phase298.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(source_sha256=hashes,recording=n,NN_used=False,GPU_used=False,RAW_FPS_measured=False,
        one_additional_hypothesis_only=True,selection_uses_only_current_discovery=True,
        late_half_updates_author_tracker=False,qualification_thresholds_changed=False,
        past_frequency_not_promoted_to_current_without_late_confirmation=True,no_labels_during_inference=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    for f in files:
        p=OUT/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    src=source(n);rows=[json.loads(s) for s in src.read_text('utf8').splitlines()]
    cache=ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}'
    phase=JointPhaseBank(ROOT.parent/'output/helix_cpu232_20261108/build/Release/helix_reference.dll',temporal_hypothesis=True)
    torque=RotorTorquePartition();periods=[];outputs=[];timings=[]
    for i,row in enumerate(rows):
        t=row['t'];tick=time.perf_counter();u,q,meta=phase.step(t,window(cache,t),row['geometry'].get('rotors',[]),row['motion'])
        period=dict(t=t,u=None if u is None else u.tolist(),quality=q,meta=meta);periods.append(period)
        info=torque.observe(row,period);r=adapt(row,info);r['physical']=evaluate(r,[.1,.2,.4]) if r['coefficient'] else {};outputs.append(r)
        timings.append(1000*(time.perf_counter()-tick))
        with (OUT/'periods.jsonl').open('a',encoding='utf8') as fh:fh.write(json.dumps(period,allow_nan=False)+'\n')
        if i%15==0:print(n,i,'rings',meta['qualified_current_rings'],'calibration',info.get('gain_zscore'),flush=True)
    path=OUT/'torque.jsonl';path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in outputs),'utf8')
    folder={'13':'separated201_20261107/reference13','15':'motion181_20261107/reference_review15','16':'unseen191_20261107/reference16'}[n]
    ref=ROOT.parent/f'output/{folder}/references.json';origins=np.round(np.arange(10.8,15.6001,.2),1)
    infos=[r['rotor_torque_response'] for r in outputs]
    result=dict(frames=len(rows),ring_histogram={str(k):sum(p['meta']['qualified_current_rings']==k for p in periods) for k in range(5)},
        prior_candidate_trials=sum(r['temporal_hypothesis_selection']['prior_candidate_tried'] for p in periods for r in p['meta']['records']),
        prior_selected=sum(r['temporal_hypothesis_selection']['selected']=='past_frequency_current_discovery_refit' for p in periods for r in p['meta']['records']),
        longest_current_torque_block=max(i.get('current_block_frames',0) for i in infos),
        best_gain_zscore=max(i.get('gain_zscore',0) for i in infos),
        gain_qualified_frames=sum(i.get('calibration_qualified',False) for i in infos),torque_qualified_frames=sum(i['qualified'] for i in infos),
        CPU_ms_mean=float(np.mean(timings)),control=score(src,ref,origins),torque=score(path,ref,origins))
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
    (OUT/'results.json').write_text(json.dumps(result,indent=2),'utf8');print({k:v for k,v in result.items() if k not in ('control','torque')},flush=True)
