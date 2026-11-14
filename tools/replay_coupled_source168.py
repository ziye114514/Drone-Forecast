"""Use the existing single joint state observer, without a second force filter."""
from pathlib import Path
import sys,json,copy,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from local_period_tracks import LocalPeriodTracks
from rotor_thrust_bank import RotorThrustBankProxy
from rotor_force_observer import RotorForceObserver
from relative_rotor_observer import RelativeRotorForceObserver
from physical_publication import evaluate
from run_force167 import pure_observation
from replay_gated_nn166 import guard
from run_rotor_only import serial

BASE=ROOT.parent/'output';OUT=BASE/'source_force168_20261105/single_observer_relativefix'
SOURCES=['tools/replay_coupled_source168.py','tools/run_force167.py','local_period_tracks.py',
    'rotor_thrust_bank.py','rotor_force_observer.py','relative_rotor_observer.py','physical_publication.py']


def replay(source,relative):
    kind=RelativeRotorForceObserver if relative else RotorForceObserver
    models={a:kind() for a in ['legacy_exact','zero_matched','tracked']}
    track=LocalPeriodTracks();proxy=RotorThrustBankProxy(vector_pool=True);result={a:[] for a in models};parity=[]
    for old in source:
        t=old['t'];obs=pure_observation(old);local=track.step(t,old.get('local_temporal_phase') or {})
        pooled=obs['pooled'] if old['structure'].get('mode') in ('four','three','partial_two') else local
        u,q,meta=proxy.step(t,old['structure'],old.get('geometry',{}).get('rotors',[]),old.get('rates',[]),pooled)
        if not obs['current_pose']:u=None;q=0.;meta=dict(meta,observed_components={})
        for arm,model in models.items():
            if old['state'].get('two_point_initialization_selected_by_partial_entry'):model.two_point_init=True
            if arm=='legacy_exact':U=old['force']['input_u'];M=old['period_selection'];Q=.7 if U is not None else 0.
            else:
                U=u;M=copy.deepcopy(meta);Q=q
                if arm=='zero_matched' and U is not None:
                    U=np.zeros(2);M['observed_components']={k:[0.,0.] for k in M['observed_components']}
            c,v,f,s=model.step(t,old['body'],None if U is None else np.asarray(U),Q,
                allow_gain_learning=M.get('learning_input_current',True) if arm=='legacy_exact' else False,
                input_meta=M,motion=old['motion'] if relative else None)
            r=copy.deepcopy(old);r.update(coefficient=c,valid=v,force=f,state=s,period_selection=M,
                local_source_tracking=local,source168_arm=arm,whole_body_NN_used=False,single_original_observer=True)
            r['physical']=evaluate(r,[.1,.2,.4]) if c else {};result[arm].append(r)
            if arm=='legacy_exact' and c:
                parity.extend(float(np.linalg.norm(np.asarray(r['physical'][h])-old['physical'][h])) for h in r['physical'])
        if model.P is not None:assert np.linalg.eigvalsh(model.P).min()>-1e-5
    assert max(parity,default=0.)<1e-6,max(parity,default=0.)
    return result,max(parity,default=0.)


if __name__=='__main__':
    sys.addaudithook(guard);OUT.mkdir(exist_ok=False);snap=OUT/'source_snapshot';snap.mkdir();hashes={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();p=snap/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);hashes[name]=hashlib.sha256(b).hexdigest()
    summary={}
    for clip,path,relative in [('pure_side',BASE/'native_plane165_20261104/cold_relative_True.jsonl',True),
        ('pure_train2',BASE/'native_plane165_20261104/train2_cache_border/predictions.jsonl',True)]:
        source=[json.loads(s) for s in path.read_text('utf-8').splitlines()];outputs,parity=replay(source,relative)
        prefix,_=replay(source[:len(source)//2],relative)
        for arm,rows in outputs.items():
            assert [r['physical'] for r in prefix[arm]]==[r['physical'] for r in rows[:len(prefix[arm])]]
            (OUT/f'{clip}_{arm}.jsonl').write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rows),'utf-8')
        summary[clip]=dict(source=str(path),legacy_parity_max_px=parity,prefix_invariant=True,
            active={a:sum(r['force']['active'] for r in v) for a,v in outputs.items()},NN_used=False)
        print(clip,json.dumps(summary[clip]),flush=True)
    summary['source_sha256']=hashes
    summary['source_changed_during_run']=[n for n,h in hashes.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    assert not summary['source_changed_during_run']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
