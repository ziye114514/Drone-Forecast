from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage149'),str(ROOT/'research/stage148'),str(ROOT/'src')]
from inertia149 import InertialBank
from dynamics149 import RelativeScale,ProjectedExpert
from temporal148 import TemporalBank

OUT=ROOT.parent/'output/stage149_inertia_scale_20261024'
OUT.mkdir(exist_ok=True)
checks={}
def rotor(x,y=100.,major=40.,angle=0.):
    return dict(cx=float(x),cy=float(y),major=float(major),minor=float(major*.5),
        ratio=.5,angle=float(angle),mode='ellipse',inferred_shape=False)
def body(x=100.,y=100.):
    return dict(detected=True,center=[float(x),float(y)])
def step(bank,t,rs,b=None):
    return bank.step(t,b or body(),copy.deepcopy(rs))
b=InertialBank()
ids=[]
for i in range(8): ids.append(step(b,i*.02,[rotor(70+2*i),rotor(130+2*i)],body(100+2*i))[0]['visual_track_id'])
assert len(set(ids))==1
for i in range(8,17): assert step(b,i*.02,[],body(100+2*i))==[]
coast=copy.deepcopy(b.coasting)
assert len(coast)==2 and all(not c['current_observation'] and c['frequency_hz'] is None for c in coast)
r=step(b,.34,[rotor(104),rotor(164)],body(134))
assert r[0]['visual_track_id']==ids[0] and r[0]['association_state']=='relock_pending'
r=step(b,.36,[rotor(106),rotor(166)],body(136))
assert r[0]['visual_track_id']==ids[0] and r[0]['association_state']=='measured'
assert b.diagnostic['relocks_confirmed']==2
checks['body_translation_gap_200ms_confirmed_relock']=True
checks['coast_not_measurement_or_frequency']=True
before=copy.deepcopy(r)
again=step(b,.36,[rotor(106),rotor(166)],body(136))
assert [q['temporal_observations'] for q in again]==[q['temporal_observations'] for q in before]
checks['same_timestamp_no_double_update']=True
step(b,.9,[])
after=step(b,.92,[rotor(106)])
assert after[0]['visual_track_id']!=ids[0]
checks['expired_track_not_forced_relock']=True
b=InertialBank()
for i in range(10): step(b,.02*i,[rotor(70+i*2)],body())
old_id=b.tracks[0]['track'].id
step(b,.28,[])
coast=next(c for c in b.coasting if c['visual_track_id']==old_id)
assert coast['center'][0]>88
r=step(b,.30,[rotor(100)],body())
assert r[0]['visual_track_id']==old_id
checks['relative_rotor_inertia_with_stationary_body']=True
b=InertialBank()
for i in range(4): step(b,.02*i,[rotor(70)],body())
old=b.tracks[0]['track'].id
r=rotor(70);r.update(mode='line',ratio=.05,minor=2.)
assert step(b,.12,[r])[0]['visual_track_id']!=old
checks['disk_line_group_no_forced_motor_identity']=True
b=InertialBank()
for i in range(6): step(b,.02*i,[rotor(76),rotor(84)])
r=step(b,.12,[rotor(80)])
assert r[0]['association_ambiguous']
checks['ambiguous_crossing_rejected']=True

# Known injected gaps: compare identity continuity, not a claimed natural-ID truth.
def artificial(bank):
    id0=None; resumed=[]
    for i in range(35):
        rs=[] if 10<=i<20 else [rotor(60+1.5*i),rotor(160+1.5*i)]
        got=step(bank,.02*i,rs,body(110+1.5*i))
        if i==9: id0=got[0]['visual_track_id']
        if i in (20,21): resumed.append(got[0]['visual_track_id']==id0)
    return resumed
checks['known_gap_old148']=artificial(TemporalBank())
checks['known_gap_new149']=artificial(InertialBank())
assert checks['known_gap_old148']==[False,False] and checks['known_gap_new149']==[True,True]

# Relative scale and no use of a lone line/hidden rotor.
scale=RelativeScale()
for i in range(20):
    t=.02*i; rs=[]
    for j in range(3):
        r=rotor(50+j*60,major=40*np.exp(.2*t))
        r.update(current_observation=True,visual_track_id=j,temporal_observations=10,association_ambiguous=False)
        rs.append(r)
    s=scale.step(t,dict(observed_rotors=rs))
assert s['valid'] and abs(s['log_scale_rate_per_s']-.2)<1e-6
checks['known_relative_scale_rate']=True
assert not scale.step(.40,dict(observed_rotors=[]))['valid']
checks['missing_scale_not_fabricated']=True

# Projection identity: normalized X is stationary, apparent scale expands.
expert=ProjectedExpert()
g=dict(observed_rotors=[],axis_xy=[0.,-1.],confidence=.8)
co=dict(position=[640.,360.],velocity=[0.,0.])
base={str(h):[640.,360.] for h in (.1,.2,.4)}
for i in range(30): result=expert.step(.02*i,body(640,360),g,dict(valid=False),co,base)
assert result['updates']==0 and not result['physical']
checks['no_feature_no_physics_update']=True
expert=ProjectedExpert()
result=expert.step(0.,body(640,360),g,dict(valid=True,relative_f2_proxy=1.),co,base)
assert result['updates']==0
expert.step(.02,dict(detected=False),g,dict(valid=False),co,base)
assert expert.scaled.updates==0
checks['not_yet_arrived_target_not_learned']=True
for i in range(2,15): expert.step(.02*i,dict(detected=False),g,dict(valid=False),co,base)
assert expert.scaled.updates==0
checks['missing_target_not_learned']=True

# Reuse recorded detector/rotor observations for an explicitly separate replay.
replays={}
oldout=ROOT.parent/'output/stage148_temporal_rotors_20261024'
for name in ['clear','side','train2','fred0']:
    source=oldout/(name+'_joint')/'predictions.jsonl'
    if not source.exists(): continue
    banks={'148':TemporalBank(),'149':InertialBank()}; totals={key:dict(new=0,matched=0,coast_ticks=0,relocks=0) for key in banks}
    last_body=None
    for line in source.read_text(encoding='utf8').splitlines():
        row=json.loads(line); t=row['t']; currentbody=row['body']
        if currentbody.get('detected'):
            if last_body is not None and t-last_body>.5:
                banks={'148':TemporalBank(),'149':InertialBank()}
            last_body=t
        raw=row['rotor_branch']['geometry'].get('temporal148',{}).get('raw_rotors',[])
        for key,bank in banks.items():
            result=bank.step(t,currentbody,raw); d=bank.diagnostic
            totals[key]['new']+=d['new_tracks']; totals[key]['matched']+=d['matched']
            totals[key]['coast_ticks']+=bool(d.get('coasting_tracks'))
            totals[key]['relocks']+=d.get('relocks_confirmed',0)
            assert len(result)==len(raw) and all(q['current_observation'] for q in result)
    replays[name]=totals
checks['recorded_observation_replay_current_counts_preserved']=True
(OUT/'controls149.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
(OUT/'observation_replay149.json').write_text(json.dumps(replays,indent=2),encoding='utf8')
print(json.dumps(dict(checks=checks,observation_replay_only=replays),ensure_ascii=False))
