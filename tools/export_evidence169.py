"""One annotation-free export of standalone rotor evidence and force features."""
from pathlib import Path
import sys,json,copy,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools'),str(ROOT/'src')]
from rotor_evidence import RotorEvidence
from local_period_tracks import LocalPeriodTracks
from nn_rotor_residual import ObservedForceGauge
from run_force167 import pure_observation
from replay_gated_nn166 import guard
from run_rotor_only import serial

BASE=ROOT.parent/'output';OUT=BASE/'decision169_20261105/evidence_v2'


if __name__=='__main__':
    sys.addaudithook(guard);OUT.mkdir(exist_ok=False);snap=OUT/'source_snapshot';snap.mkdir(exist_ok=False)
    names=['rotor_evidence.py','local_period_tracks.py','nn_rotor_residual.py','tools/run_force167.py','tools/export_evidence169.py']
    hashes={}
    for n in names:
        b=(ROOT/n).read_bytes();p=snap/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);hashes[n]=hashlib.sha256(b).hexdigest()
    sources={};configs=json.loads((BASE/'nn_rotor166_20261105/summary.json').read_text('utf-8'))
    for clip,config in configs.items():
        obs=[json.loads(s) for s in (BASE/f'nn_rotor166_20261105/{clip}_observations.jsonl').read_text('utf-8').splitlines()]
        base=[json.loads(s) for s in (BASE/f'source_force168_20261105/native_selection_fixed/{clip}_zero_algebra.jsonl').read_text('utf-8').splitlines()]
        sources['NN_'+clip]=(base,obs)
    for clip in ['pure_side','pure_train2']:
        base=[json.loads(s) for s in (BASE/f'source_force168_20261105/single_observer_relativefix/{clip}_zero_matched.jsonl').read_text('utf-8').splitlines()]
        obs=[dict(pure_observation(r),local_temporal_phase=r.get('local_temporal_phase'),body=r['body']) for r in base]
        sources[clip]=(base,obs)
    summary={}
    for name,(rows,observations) in sources.items():
        model=RotorEvidence();side=LocalPeriodTracks();gauge=ObservedForceGauge();features=[];packets=[]
        for r,o in zip(rows,observations):
            assert r['t']==o['t'];s=o['structure']
            packet=model.step(r['t'],o['body'],s,o['geometry'].get('rotors',[]),o.get('rates',[]))
            local=side.step(r['t'],o.get('local_temporal_phase') or {})
            components=copy.deepcopy(packet['current_components']);source='current_individual_regions'
            if not components and local['qualified'] and packet['axis_xy'] is not None:
                components={'local_change:'+str(local['source_epoch']):(local['frequency_hz']**2/360000.*np.asarray(packet['axis_xy'])).tolist()}
                source='normalized_local_change_not_total_thrust'
            u=np.sum(list(components.values()),axis=0) if components else None
            value,transition=gauge.step(u,dict(observed_components=components))
            packets.append(dict(packet,local_period_proxy=local,feature_source=source,feature_partition_transition=transition))
            features.append(dict(t=r['t'],valid=r['valid'],coefficient=r.get('coefficient'),
                u=None if value is None else value.tolist(),current_input_available=value is not None,
                evidence_branch=packet['branch'],contribution_count=packet['current_contribution_count'],
                geometry_template_mode=s.get('mode'),source=source,NN_auxiliary=name.startswith('NN_')))
        (OUT/f'{name}_features.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in features),'utf-8')
        (OUT/f'{name}_evidence.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in packets),'utf-8')
        summary[name]=dict(frames=len(rows),template_free_four_sum=sum(p['complete_current_frequency_sum'] for p in packets),
            four_sum_without_template=sum(p['complete_current_frequency_sum'] and not p['template_confirmed'] for p in packets),
            current_individual_contribution_frames=sum(p['current_contribution_count']>0 for p in packets),
            current_any_input=sum(r['current_input_available'] for r in features),
            current_geometry_branches={b:sum(p['branch']==b for p in packets) for b in sorted(set(p['branch'] for p in packets))},
            source_fields='native candidate arcs + current frequency tests; not independent physical truth')
        print(name,json.dumps(summary[name]),flush=True)
    summary['source_sha256']=hashes;summary['source_changed_during_run']=[n for n,h in hashes.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    assert not summary['source_changed_during_run']
    (OUT/'evidence_summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
