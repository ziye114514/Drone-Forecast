"""Why do side-view origins lose the force input, and can a bounded weak
observation recover them without hurting the others?

The input chain has a natural hierarchy, and a weaker prior must sit BELOW the
existing hold, not above it:

   1 qualified period (existing selector)
   2 recent held amplitude (existing bridge,0.12s)
   3 unqualified pooled estimate, only when it is young, consistent with the
     last accepted frequency, has enough pixels, and never counts as a learning
     observation  <- the only new ingredient tested here

Variants differ only in that hierarchy; the replay, labels and scoring are the
frozen ones, and the base arm must reproduce the published number first.
"""
from pathlib import Path
import json,sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from visibility_period import LegacyPeriodProxy,force_input
from probe_same_window_motion import replay
from compare_final_pipeline import score,reference_at
from axis_amplitude_bridge import current_axis_qualified

BASE=ROOT.parent/'output'
OUT=BASE/'peak_ownership_20261029'
OPTIONS=dict(curved=True,evidence_gain=True,memory=True,endpoint=True,memory_gaps=True,weak_support=True)
CLIPS=[('side','motion_consistency_30px_20261028/RAW_side_release_candidate8s/predictions.jsonl',
        'dense_labels_side_110_1195'),
       ('other','thrust_rate_review_20261029/experiment/other_baseline.jsonl','dense_labels_train2_40_46')]
WEAK=dict(max_age_s=.3,rel_tol=.08,min_pixels=2)


class Bridge:
    """The existing recent-amplitude bridge with a configurable maximum age."""
    def __init__(self,max_hold_s=.12):
        self.last=None;self.last_t=None;self.epoch=None;self.max_hold_s=max_hold_s

    def step(self,t,structure,u,quality,period):
        epoch=structure.get('geometry_epoch')
        if (self.last_t is not None and t-self.last_t>.5) or (self.epoch is not None and epoch!=self.epoch):
            self.last=None
        self.last_t=t;self.epoch=epoch
        relative=period.get('relative_f2')
        if u is not None and quality>=.35 and relative is not None:
            complete=period.get('learning_input_current',True)
            if complete:self.last=(t,float(relative))
            return u,quality,dict(applied=False,amplitude_current=complete,learning_input_current=complete,
                reason='current_complete_input' if complete else 'current_axis_with_partial_rotor_prior',
                partial_estimate_does_not_refresh_complete_amplitude=not complete)
        current_axis=current_axis_qualified(structure)
        if self.last is None or not current_axis or t-self.last[0]>self.max_hold_s+1e-9:
            return u,quality,dict(applied=False,amplitude_current=False,learning_input_current=True,
                reason='no_recent_amplitude_or_current_axis')
        revised,revised_quality=force_input(structure,self.last[1],True)
        if revised is None:
            return u,quality,dict(applied=False,amplitude_current=False,learning_input_current=True,
                reason='current_axis_unqualified')
        return revised,revised_quality,dict(applied=True,amplitude_current=False,learning_input_current=False,
            reason='current_axis_with_recent_period_amplitude',period_age_s=t-self.last[0],
            maximum_period_hold_s=self.max_hold_s,held_relative_f2=self.last[1],
            current_frequency_invented=False,hidden_rotor_frequencies_added=0,
            held_amplitude_excluded_from_gain_and_input_integral_learning=True)


def weak_input(t,structure,info,reference,last_accepted,pooled,config):
    weak=pooled.get('frequency_hz');pixels=pooled.get('pixels') or 0
    if (weak is None or reference is None or last_accepted is None or pixels<config['min_pixels']
            or t-last_accepted[0]>config['max_age_s']+1e-9
            or abs(np.log(weak/last_accepted[1]))>config['rel_tol']):
        return None,0.,dict(info,weak_pooled_input=False)
    relative=float((weak/reference)**2)
    revised,quality=force_input(structure,relative,True)
    if revised is None:
        return None,0.,dict(info,weak_pooled_input=False)
    return revised,quality,dict(info,source='weak_pooled_consistent_with_recent_accepted',
        frequency_hz=float(weak),relative_f2=relative,coherent=True,
        learning_input_current=False,weak_pooled_input=True,
        weak_pooled_age_s=t-last_accepted[0],weak_pooled_pixels=int(pixels),
        hidden_frequencies_measured=0,mechanical_RPM_measured=False)


def chain(rows,hold_s=.12,weak=None):
    selector=LegacyPeriodProxy();bridge=Bridge(hold_s);last_accepted=None;out=[];census=collections.Counter()
    for r in rows:
        structure=r['structure']
        u,q,info=selector.step(r['t'],structure,r['geometry'].get('rotors',[]),r['rates'],r['period'])
        if u is not None and info.get('frequency_hz') is not None:
            last_accepted=(r['t'],float(info['frequency_hz']))
        u,q,bridge_info=bridge.step(r['t'],structure,u,q,info)
        if u is None and weak is not None:
            u,q,info=weak_input(r['t'],structure,info,selector.reference,last_accepted,r['period'],weak)
            if info.get('weak_pooled_input'):
                bridge_info=dict(bridge_info,learning_input_current=False,weak_pooled_input=True)
        if u is None:
            if structure.get('axis_xy') is None:census['no_axis']+=1
            elif r['period'].get('frequency_hz') is None:census['no_frequency_at_all']+=1
            elif r['period'].get('qualified') is not True:census['weak_unqualified_frequency']+=1
            else:census['gate_or_coherence']+=1
        out.append(dict(r,period_selection=info,axis_amplitude_bridge=bridge_info,
            force=dict(r['force'],input_u=u)))
    return out,census


def controlled():
    structure=dict(axis_xy=[0.,-1.],geometric_weight=.9,mode='axis_only',geometry_epoch=0,
        axis_support=dict(valid=True,confidence=.9))
    def tick(qualified,frequency,pixels):
        return dict(qualified=qualified,frequency_hz=frequency,pixels=pixels,
            reason='qualified_flicker' if qualified else 'weak_or_common_mode')
    def row(t,pooled):
        return dict(t=t,structure=structure,geometry={'rotors':[]},rates=[],period=pooled,force={})
    rows=[row(0.,tick(True,270.,200)),row(1/30,tick(True,270.,200)),
        row(2/30,tick(False,275.,10)),row(.2,tick(False,275.,10)),
        row(.233,tick(False,320.,10)),row(.6,tick(False,275.,10))]
    base,_=chain(rows,weak=None)
    helped,_=chain(rows,weak=WEAK)
    fresh,_=chain([rows[3]],weak=WEAK)
    return dict(
        base_second_tick_has_input=base[1]['force']['input_u'] is not None,
        bridge_still_holds_at_33ms=base[2]['force']['input_u'] is not None and
            base[2]['axis_amplitude_bridge'].get('applied') is True,
        weak_does_not_override_the_hold=not helped[2]['period_selection'].get('weak_pooled_input',False),
        base_gives_nothing_after_120ms=base[3]['force']['input_u'] is None,
        weak_accepts_young_consistent_estimate=helped[3]['force']['input_u'] is not None and
            helped[3]['period_selection'].get('weak_pooled_input') is True,
        weak_input_never_learns=helped[3]['period_selection'].get('learning_input_current') is False,
        inconsistent_weak_rejected=helped[4]['force']['input_u'] is None,
        stale_weak_rejected=helped[5]['force']['input_u'] is None,
        no_recent_acceptance_rejected=fresh[0]['force']['input_u'] is None,
        hidden_frequencies_measured=0)


def main():
    OUT.mkdir(exist_ok=True)
    result=dict(controlled=controlled(),clips={},default_enabled=False,forecaster_modified=False,
        annotations_used_for_decisions=False)
    plans=[('base',.12,None),('weak_last',.12,WEAK),('hold30',.30,None),('hold30_weak_last',.30,WEAK)]
    for name,rel,label in CLIPS:
        rows=[json.loads(x) for x in (BASE/rel).read_text('utf-8').splitlines()]
        truth=sorted(json.loads((BASE/label/'labels.json').read_text('utf-8'))['records'].values(),
            key=lambda r:r['t'])
        variants={};census={}
        for key,hold,weak in plans:
            variants[key],census[key]=chain(rows,hold,weak)
        replayed={key:replay(value,False,**OPTIONS)[0] for key,value in variants.items()}
        entry=dict(modes={},census=census)
        for key,values in replayed.items():
            scored=score(values,truth)
            inputs=sum(1 for r in values if r['force'].get('input_u') is not None)
            weak_ticks=sum(1 for r in values if (r.get('period_selection') or {}).get('weak_pooled_input'))
            mode_entry=dict(force_active_ticks=sum(1 for r in values if r['force'].get('active')),
                input_ticks=inputs,weak_input_ticks=weak_ticks)
            mode_entry['error_0.4']=scored['physical_future_error_px']['0.4']
            entry['modes'][key]=mode_entry
        origin_rows=[];last=None
        for r in replayed['base']:
            slot=int(np.floor(r['t']*5+1e-8))
            if not r['valid'] or slot==last:continue
            last=slot
            target=reference_at(truth,r['t']+.4)
            if target is None or '0.4' not in r['physical']:continue
            errors={}
            for key,values in replayed.items():
                row=min(values,key=lambda x:abs(x['t']-r['t']))
                if '0.4' in row['physical']:
                    errors[key]=float(np.linalg.norm(np.asarray(row['physical']['0.4'])-target))
            origin_rows.append(dict(t=r['t'],base_has_input=r['force'].get('input_u') is not None,
                current_error_px=None if reference_at(truth,r['t']) is None or
                    not r['body'].get('detected') else float(np.linalg.norm(
                        np.asarray(r['body']['center'])-reference_at(truth,r['t']))),
                error=errors))
        entry['origins']=origin_rows
        entry['ticks']=len(rows)
        result['clips'][name]=entry
        print(name,{k:round(v['error_0.4']['mean'],4) for k,v in entry['modes'].items()},
            'inputs',{k:v['input_ticks'] for k,v in entry['modes'].items()},flush=True)
        print('   census base',dict(census['base']),flush=True)
    (OUT/'force_coverage.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),'utf-8')
    print(json.dumps(result['controlled'],ensure_ascii=False))


if __name__=='__main__':main()
