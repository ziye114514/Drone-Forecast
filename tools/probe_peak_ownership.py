"""Mechanism checks and a real-RAW ownership audit of the spatial variance.

The synthetic part fixes causal behaviour before any real data is read: a
dominance change must not be reported as a rotation change, a harmonic must not
become a rotor identity, and one component must never resolve four slots.

The real part re-decodes the same cutoffs with the public rolling reader only to
place events inside the same 160 ms window. It does not re-run or modify the
forecaster, and no annotation or future outcome enters an attribution.
"""
from pathlib import Path
from collections import deque
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from live_decode import native_batches
from side_moment_period import binned_moment,component
from spatial_peak_ownership import GRID,PeakOwnershipAudit,power_spectrum
from visibility_period import VisibilityPeriodProxy

DTYPE=[('x','u2'),('y','u2'),('p','i2'),('t','i8')]
OUT=ROOT.parent/'output/peak_ownership_20261029'
BASE=ROOT.parent/'output'
FOUR=(190.,240.,290.,340.)
STEP=1/30.
PINNED=26.499735


def timeline(widths,span,dt=.00025,repeats=8,y=300):
    """Continuous-phase line events; widths(t) returns (center_x,frequency,amplitude)."""
    rows=[];phase={}
    for t in np.arange(0.,span,dt):
        for j,(cx,f,a) in enumerate(widths(float(t))):
            phase[j]=phase.get(j,0.)+np.pi*f*dt
            r=a*np.cos(phase[j])
            for x in (cx-r,cx+r):
                for _ in range(repeats):rows.append((round(float(x)),y,1,round(float(t)*1e6)))
    return np.array(rows,dtype=DTYPE)


def audit_sequence(events,slots_fn,span,center=500.,axis=(1.,0.)):
    audit=PeakOwnershipAudit();out=[]
    for cut in np.arange(.16,span+.0001,STEP):
        times,value,valid,counts=binned_moment(events,float(cut),center,axis,(0.,0.))
        out.append(audit.step(float(cut),times,value,valid,slots_fn(float(cut))))
    return out


def checks():
    results={}
    one=timeline(lambda t:((300.,200.,40.),),.32)
    single=audit_sequence(one,lambda t:{0:200.},.32)
    results['single_component_owned']=dict(ok=single[-1]['corroborated_slots']==[0] and
        single[-1]['state'] in ('stable_owner','owner_initialized'),states=[r['state'] for r in single],
        frequencies=[round(r['dominant_frequency_hz'],2) for r in single])
    balanced=(60.,55.,50.,45.)
    four=timeline(lambda t:tuple((300.+150.*j,FOUR[j],balanced[j]) for j in range(4)),.32)
    order=audit_sequence(four,lambda t:{i:f for i,f in enumerate(FOUR)},.32)
    results['four_components_resolved']=dict(ok=order[-1]['corroborated_slots']==[0,1,2,3] and
        order[-1]['resolved_slot_fraction']==1.,corroborated=order[-1]['corroborated_slots'],
        frequencies=[round(p['frequency_hz'],2) for p in order[-1]['peaks']])
    lazy=(60.,50.,30.,20.)
    skew=timeline(lambda t:tuple((300.+150.*j,FOUR[j],lazy[j]) for j in range(4)),.32)
    weak=audit_sequence(skew,lambda t:{i:f for i,f in enumerate(FOUR)},.32)
    results['weak_component_masked_by_leakage']=dict(ok=len(weak[-1]['corroborated_slots'])<4,
        corroborated=weak[-1]['corroborated_slots'],resolved=weak[-1]['resolved_slot_fraction'],
        note='unwindowed Lomb-Scargle leakage hides the weakest rotor; documented, not silently repaired')
    permuted=audit_sequence(four,lambda t:{3:FOUR[3],2:FOUR[2],1:FOUR[1],0:FOUR[0]},.32)
    results['identity_free_square_sum']=dict(ok=permuted[-1]['corroborated_slots']==[0,1,2,3] and
        abs(sum(f*f for f in FOUR)-sum(f*f for f in
            (float(v) for v in permuted[-1]['slot_frequencies'].values())))<1e-9,
        note='sum of squared slot frequencies is invariant to slot labels')
    lone=audit_sequence(timeline(lambda t:((300.,240.,40.),),.28),
        lambda t:{0:190.,1:240.,2:290.,3:340.},.28)
    results['single_component_cannot_corroborate_four']=dict(ok=lone[-1]['resolved_slot_fraction']==.25 and
        lone[-1]['corroborated_slots']==[1],resolved=lone[-1]['resolved_slot_fraction'])
    switch_time=.42
    def amplitudes(t):
        a=(60.,50.,40.,35.) if t<switch_time else (35.,60.,40.,50.)
        return tuple((300.+150.*j,FOUR[j],a[j]) for j in range(4))
    switch=audit_sequence(timeline(amplitudes,.88),lambda t:{i:f for i,f in enumerate(FOUR)},.88)
    post=[r['state'] for r in switch[8:]]
    results['dominance_switch_is_not_rotation']=dict(ok='dominance_switch' in post and
        'rotation_change_of_same_owner' not in post and switch[-1]['dominant_model_id']==1,
        states=post[:8],final_owner=switch[-1]['dominant_model_id'])
    def turning(t):
        f=190. if t<switch_time else 215.
        return tuple((300.+150.*j,f if j==0 else FOUR[j],(120.,45.,40.,35.)[j]) for j in range(4))
    def turn_slots(t):
        # The slot estimator reports the new speed only after it is observed.
        return {0:190. if t<switch_time+.10 else 215.,1:FOUR[1],2:FOUR[2],3:FOUR[3]}
    spin=audit_sequence(timeline(turning,.88),turn_slots,.88)
    post_spin=[r['state'] for r in spin[8:]]
    results['rotation_change_of_same_owner']=dict(ok='rotation_change_of_same_owner' in post_spin and
        'dominance_switch' not in post_spin and spin[-1]['dominant_model_id']==0,
        states=post_spin,final_owner=spin[-1]['dominant_model_id'])
    def smearing(t):
        f=190. if t<switch_time else 215.
        return tuple((300.+150.*j,f if j==0 else FOUR[j],balanced[j]) for j in range(4))
    smear=audit_sequence(timeline(smearing,.88),turn_slots,.88)
    post_smear=[r['state'] for r in smear[8:]]
    results['rotation_change_hidden_by_dominance_transfer']=dict(
        ok='dominance_switch' in post_smear and 'rotation_change_of_same_owner' not in post_smear,
        states=post_smear[:8],
        note='with comparable amplitudes the changing rotor loses the peak while its energy smears; '
             'the audit reports the dominance transfer and never invents a rotation change')
    harmonic=audit_sequence(timeline(lambda t:((300.,240.,40.),),.26),lambda t:{0:120.},.26)
    results['harmonic_is_not_identity']=dict(ok=harmonic[-1]['state']=='harmonic_not_identity' and
        harmonic[-1]['dominant_model_id'] is None,state=harmonic[-1]['state'],
        mode=harmonic[-1]['attribution'][0]['mode'])
    half=audit_sequence(timeline(lambda t:((300.,120.,40.),),.26),lambda t:{0:240.},.26)
    results['half_is_not_identity']=dict(ok=half[-1]['state']=='harmonic_not_identity',
        state=half[-1]['state'],mode=half[-1]['attribution'][0]['mode'])
    tied=audit_sequence(timeline(lambda t:((300.,231.5,40.),),.26),lambda t:{0:230.,1:233.},.26)
    results['ambiguous_two_slots']=dict(ok=tied[-1]['state']=='ambiguous_ownership',state=tied[-1]['state'])
    both=audit_sequence(timeline(lambda t:((300.,240.,40.),),.26),lambda t:{0:120.,1:240.},.26)
    results['direct_match_beats_harmonic']=dict(ok=both[-1]['dominant_model_id']==1 and
        both[-1]['state'] in ('owner_initialized','stable_owner'),state=both[-1]['state'])
    rows=[]
    for t in np.arange(0.,.16,1/30.):
        for x in (300,400):
            for _ in range(int(8+6*np.sin(2*np.pi*350*t))):rows.append((x,300,1,round(float(t)*1e6)))
    times,value,valid,counts=binned_moment(np.array(rows,dtype=DTYPE),.16,350.,(1.,0.),(0.,0.))
    quiet=PeakOwnershipAudit().step(.16,times,value,valid,{0:350.})
    results['common_multiplicity_creates_no_owner']=dict(ok=quiet['state']=='no_periodic_component',
        state=quiet['state'],reason=quiet['reason'])
    return results


def real_pass(specs,stride=1):
    mapper=VisibilityPeriodProxy();records={}
    for name,raw,path,label in specs:
        rows=[json.loads(x) for x in Path(path).read_text('utf-8').splitlines()][::stride]
        rows=sorted(rows,key=lambda r:r['t'])
        row_times=np.array([r['t'] for r in rows])
        begin=min(r['t'] for r in rows)-.05;duration=max(r['t'] for r in rows)-begin+.06
        audit=PeakOwnershipAudit();queue=deque();found=[];started=time.perf_counter();seen=0
        last_cutoff=-np.inf
        for t,e in native_batches(Path(raw),begin,duration,False,33333,reader_backend='rolling'):
            queue.append(e.copy())
            while queue and (not len(queue[0]) or queue[0]['t'][-1]<(t-.17)*1e6):queue.popleft()
            k=int(np.searchsorted(row_times,t))
            near=[j for j in (k-1,k) if 0<=j<len(row_times)]
            if not near:continue
            j=min(near,key=lambda i:abs(row_times[i]-t))
            if abs(row_times[j]-t)>1/60.:continue
            row=rows[j];cutoff=float(row_times[j]);seen+=1
            if cutoff<=last_cutoff:continue
            last_cutoff=cutoff
            all_events=np.concatenate(queue)
            base=dict(t=cutoff,clip=name,mode=row['structure'].get('mode'),
                body_detected=bool(row['body'].get('detected')),
                force_active=bool(row['force'].get('active')),
                force_source=row['period_selection'].get('source'))
            if not base['body_detected'] or row['structure'].get('axis_xy') is None:
                found.append(dict(base,state='no_current_body_or_plane',reason='no_current_body_or_plane',
                    claimed_slots=[],slot_frequencies={},corroborated_slots=[],resolved_slot_fraction=0.,
                    spectrum_qualified=False,dominant_model_id=None,dominant_frequency_hz=None,
                    peaks=[],attribution=[]))
                continue
            box=np.asarray(row['body']['bbox']);center=np.asarray(row['body']['center'])
            velocity=row['coefficient']['velocity'];normal=np.asarray(row['structure']['axis_xy'])
            axis=np.array([-normal[1],normal[0]])
            keep=((all_events['x']>=box[0])&(all_events['x']<box[2])&
                  (all_events['y']>=box[1])&(all_events['y']<box[3]))
            times,value,valid,counts=binned_moment(all_events[keep],cutoff,center,axis,velocity)
            slots,mapinfo=mapper._slots(row['structure'],row['geometry'].get('rotors',[]),row['rates'])
            result=audit.step(cutoff,times,value,valid,slots)
            record=dict(base,**result,mapping=mapinfo)
            pinned=abs(row['t']-PINNED)<.02
            if pinned or sum('single_peak_component' in r for r in found)<3:
                record['single_peak_component']=component(times,value,valid)
            if pinned or (result['state'] in ('owner_initialized','dominance_switch',
                    'rotation_change_of_same_owner') and sum('spectrum' in r for r in found)<3):
                power,info=power_spectrum(times,value,valid)
                record['spectrum']=dict(grid_hz=[float(x) for x in GRID],
                    power=None if power is None else [float(x) for x in power])
            found.append(record)
            if seen%40==0:print(name,seen,'ticks',round(time.perf_counter()-started,1),'s',flush=True)
        records[name]=found
        print(name,'rows',len(rows),'records',len(found),'elapsed',
            round(time.perf_counter()-started,1),flush=True)
    return records


def summarize(records):
    import collections
    summary={}
    for name,found in records.items():
        owned=[r for r in found if r.get('dominant_model_id') is not None]
        errors=[abs(np.log(r['dominant_frequency_hz']/float(r['slot_frequencies'][str(r['dominant_model_id'])])))
            for r in owned if str(r['dominant_model_id']) in r.get('slot_frequencies',{})]
        summary[name]=dict(ticks=len(found),states=dict(collections.Counter(r['state'] for r in found)),
            ticks_with_slot_claims=sum(bool(r.get('claimed_slots')) for r in found),
            owned_ticks=len(owned),
            dominant_owner_histogram=dict(collections.Counter(str(r['dominant_model_id']) for r in owned)),
            mean_resolved_slot_fraction=float(np.mean([r.get('resolved_slot_fraction',0.) for r in found])),
            max_peak_to_slot_relative_error=None if not errors else float(np.expm1(max(errors))),
            median_peak_to_slot_relative_error=None if not errors else float(np.expm1(np.median(errors))),
            force_active_ticks=sum(r.get('force_active',False) for r in found),
            qualified_spectra=sum(bool(r.get('spectrum_qualified')) for r in found))
    return summary


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    result=dict(checks=checks(),default_enabled=False,forecaster_modified=False,
        annotations_used=False,mechanical_RPM_measured=False,thrust_measured=False)
    (OUT/'mechanism_checks.json').write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({k:v['ok'] for k,v in result['checks'].items()}),flush=True)
    specs=[('clear','E:/dataset/train/6/media/event.raw',
            BASE/'motion_consistency_30px_20261028/RAW_clear_release_candidate2s/predictions.jsonl','none'),
        ('side','E:/dataset/train/6/media/event.raw',
            BASE/'motion_consistency_30px_20261028/RAW_side_release_candidate8s/predictions.jsonl',
            'dense_labels_side_110_1195'),
        ('other','E:/dataset/train/2/aligned/event.raw',
            BASE/'thrust_rate_review_20261029/experiment/other_baseline.jsonl','dense_labels_train2_40_46')]
    records=real_pass(specs)
    for name,found in records.items():
        (OUT/f'audit_{name}.jsonl').write_text(
            '\n'.join(json.dumps(r,allow_nan=False) for r in found)+'\n','utf-8')
    result['summary']=summarize(records)
    result['records']={name:len(found) for name,found in records.items()}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),'utf-8')
    print(json.dumps(result['summary'],indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
