"""Train-only KF process-noise selection, fixed transfer and RPM-Q controls.

All observations/features are existing causal caches. This is not RAW-only
deployment, author-code reproduction, or true-RPM verification.
"""
from pathlib import Path
from collections import deque
from types import FunctionType
import argparse,copy,datetime,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage88'),str(ROOT/'research/stage74')]
from hybrid74 import ControlledFilter
from inventory88 import save,load,sha,parse,unique_rows
import score88
from paired88 import prior
OUT=ROOT.parent/'output/stage90_train_only_kinematic_20261022'
BASE=ROOT.parent/'output/stage88_causal_event_controls_20261022'
FREQ=ROOT.parent/'output/stage89_small_target_periods_20261022'
AUTO=ROOT.parent/'output/stage78_RAW_separated_forecast_20261021'
SIGMAS=[50.,100.,200.,400.,800.,1600.,3200.,6400.]
TRAIN=[0,4,6,10];TEST=[8,9]
METHODS=['cv','trusted_acc','kf_tuned','guarded_q','raw_q','constant_q','high_q','lag_q','shuffle101','shuffle202','shuffle303']


def rows(p):return score88.rows(p)
def sources():return {str(p):sha(p) for p in [Path(__file__),ROOT/'research/stage74/hybrid74.py',ROOT/'research/stage88/score88.py']}


def freeze():
    assert not OUT.exists()
    save(OUT/'protocol90.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        train=TRAIN,test=TEST,baseline_sigma_grid=SIGMAS,position_noise_sigma_px=6.,
        choice='Lowest equal-recording mean0.4s FDE on four TRAIN recordings only. No test/self labels opened in selection.',
        evaluation='88 identical anchors/tolerance/actual-label ADE. Test8/9 previously inspected historically, not blind.',
        Q_rule='Q scale=max(.5,1+2r+max(0,clip(dr/dt,-1,1))); r=clip((frequency-60)/1140,0,1). '
            'Existing ControlledFilter sigma multiplied by sqrt(scale). Paper-inspired, not exact paper reproduction.',
        controls='Guarded histogram and raw argmax separated. Raw/constant/high/lag/shuffle share raw candidate clock. '
            'Constant is median first5 past raw candidate levels; highQ scale4; lag200ms; three seeded past200ms..2s shuffles.',
        frequency_interpretation='Raw histogram failed some synthetic negatives. Its Q arm is an explicit unreliable-input diagnostic, not thrust/RPM.',
        self_transfer='Train-selected kinematic KF only on78 automatic position observations; compare fixed existing sparse targets, no tuning.',
        structural_controls=controls(),no_default_promotion=True))
    print('90 train-only baseline/diagnostic protocol frozen',flush=True)


def verify():
    p=load(OUT/'protocol90.json');assert p['source_hashes']==sources();return p


class Bank:
    def __init__(self,sigma,methods=None):
        self.sigma=sigma;self.names=[m for m in (METHODS if methods is None else methods) if m not in ['cv','trusted_acc']]
        self.filters={m:None for m in self.names};self.last_t=None;self.last_seen=None;self.refs=[];self.pool=deque();self.prev={};self.rng={f'shuffle{s}':np.random.default_rng(s) for s in [101,202,303]}
    def step(self,t,z,frequency,guarded,valid,original):
        if self.last_seen is not None and t-self.last_seen>.5:
            self.__init__(self.sigma,self.names)
        if z is not None:self.last_seen=t
        dt=.02 if self.last_t is None else t-self.last_t;self.last_t=t
        if frequency is not None and len(self.refs)<5:self.refs.append(float(np.clip((frequency-60.)/1140.,0,1)))
        ready=len(self.refs)==5;current=frequency is not None and ready
        while self.pool and t-self.pool[0][0]>2.+1e-8:self.pool.popleft()
        pool=[q for q in self.pool if t-q[0]>=.2-1e-8]
        r=float(np.clip((frequency-60.)/1140.,0,1)) if frequency is not None else 0.
        ref=float(np.median(self.refs)) if ready else 0.
        values=dict(raw_q=r,constant_q=ref,lag_q=pool[-1][1] if pool else ref,guarded_q=r)
        for name,rng in self.rng.items():values[name]=pool[int(rng.integers(len(pool)))][1] if pool else ref
        coeff={m:copy.deepcopy(original['coefficients'][m]) for m in ['cv','trusted_acc'] if valid};scales={}
        for name,f in self.filters.items():
            active=bool(current and (guarded or name!='guarded_q'))
            scale=1.
            if active and name not in ['kf_tuned','high_q']:
                old=self.prev.get(name);derivative=0. if old is None or t-old[0]>.16 else np.clip((values[name]-old[1])/(t-old[0]),-1.,1.)
                scale=max(.5,1+2*values[name]+max(0.,derivative));self.prev[name]=(t,values[name])
            elif active and name=='high_q':scale=4.
            if f is None:
                if z is not None:f=ControlledFilter(z);self.filters[name]=f
            else:
                f.dt=dt;f.a=np.zeros(2);f.sigma=self.sigma*np.sqrt(scale);f.predict()
                if z is not None:f.update(z)
            if valid:
                assert f is not None
                coeff[name]=dict(position=f.x[:2].tolist(),velocity=f.x[2:].tolist(),acceleration=[0.,0.])
            scales[name]=scale
        if frequency is not None:self.pool.append((t,r))
        return dict(t=t,valid=valid,coefficients=coeff,diagnostics=dict(fresh=current,scales=scales))


def predict_seq(seq,sigma,destination,baseline_only=False):
    reference=rows(BASE/'fred'/f'sequence{seq}'/'forecasts.jsonl');features=rows(FREQ/f'sequence{seq}'/'candidates.jsonl')
    assert len(reference)==len(features)==1500
    destination.mkdir(parents=True,exist_ok=False);bank=Bank(sigma,['kf_tuned'] if baseline_only else None);elapsed=[]
    with (destination/'forecasts.jsonl').open('x',encoding='utf-8') as f:
        for old,feature in zip(reference,features):
            assert old['t']==feature['t'];audit=old['input_audit'];b=audit['bbox'];z=None
            if audit['current_label_t'] is not None:z=((np.array(b[:2])+b[2:])/2).tolist()
            raw=feature['histogram'];frequency=raw['frequency_hz'] if raw['raw_qualified'] else None
            started=time.perf_counter();value=bank.step(old['t'],z,frequency,raw['guarded_qualified'],old['valid'],old);elapsed.append((time.perf_counter()-started)*1000)
            f.write(json.dumps(value,allow_nan=False)+'\n')
    save(destination/'completion.json',dict(sequence=seq,ticks=len(reference),source_hashes_unchanged=True,
        sigma=sigma,uses_past_provided_boxes=True,cached_native_event_features=True,actual_RAW_run=False,
        predictor_median_ms=float(np.median(elapsed)),predictor_p95_ms=float(np.percentile(elapsed,95))))


def scored_for(folder,cases,methods=METHODS):
    # Reuse88 scorer in an isolated global namespace, no monkey patch.
    scoring=FunctionType(score88.fred_score.__code__,dict(score88.fred_score.__globals__,OUT=folder,METHODS=methods))
    p=load(BASE/'protocol88.json');p['fred_cases']=cases
    return scoring(p)


def annotations_to(folder,cases):
    (folder/'annotations').mkdir(parents=True,exist_ok=True)
    for seq in cases:(folder/'annotations'/f'{seq}.txt').write_bytes((BASE/'annotations'/f'{seq}.txt').read_bytes())


def select():
    verify();assert not (OUT/'selected_sigma.json').exists();results=[]
    for sigma in SIGMAS:
        folder=OUT/'training_grid'/f'sigma{sigma:g}';annotations_to(folder,TRAIN)
        for seq in TRAIN:predict_seq(seq,sigma,folder/'fred'/f'sequence{seq}',True)
        detail,_=scored_for(folder,TRAIN,['kf_tuned'])
        means={str(seq):float(np.mean([r['fde'] for r in detail if r['case']==f'fred{seq}' and r['method']=='kf_tuned' and r['horizon']==.4 and r['valid']])) for seq in TRAIN}
        result=dict(sigma=sigma,by_train_recording=means,macro_fde=float(np.mean(list(means.values()))));results.append(result)
        print('90 TRAIN GRID',result,flush=True)
    chosen=min(results,key=lambda r:r['macro_fde'])
    save(OUT/'selected_sigma.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sigma=chosen['sigma'],grid=results,
        input_training_sequences=TRAIN,test_labels_read=False,self_labels_read=False,selection_complete_before_test_prediction=True))
    print('90 selected solely on TRAIN',chosen['sigma'],flush=True)


def controls():
    c=dict(position=[0.,0.],velocity=[0.,0.],acceleration=[0.,0.]);original=dict(coefficients=dict(cv=c,trusted_acc=c))
    model=Bank(400.,['kf_tuned']);direct=None;trace=[];last_t=None
    for j in range(100):
        t=(j+1)*.02;z=[t*40.,10*np.sin(t)] if j%3 else None
        if z is None and direct is None:continue
        q=model.step(t,z,None,False,True,original)
        if direct is None:direct=ControlledFilter(z)
        else:
            direct.dt=t-last_t;direct.sigma=400.;direct.predict()
            if z is not None:direct.update(z)
        assert q['coefficients']['kf_tuned']['position']==direct.x[:2].tolist()
        assert q['coefficients']['kf_tuned']['velocity']==direct.x[2:].tolist()
        trace.append(q);last_t=t
    return dict(existing_filter_exact100_ticks=True,no_unverified_frequency_changes_baseline=True,
                sigma_choice_no_test_inputs=True,scope='Numerical wiring, not real-world accuracy.')


def evaluate():
    verify();selected=load(OUT/'selected_sigma.json');assert not (OUT/'scores90.json').exists()
    annotations_to(OUT,TRAIN+TEST)
    for seq in TRAIN+TEST:predict_seq(seq,selected['sigma'],OUT/'fred'/f'sequence{seq}')
    details,audit=scored_for(OUT,TRAIN+TEST)
    # Fixed-domain transfer: automatic observations unchanged; no FRED boxes.
    old_scores=load(AUTO/'scoring/analysis.json')['details'];own=[]
    for case in ['clear6','side6','train3','extra1','extra2','extra4']:
        bank=Bank(selected['sigma']);pred={}
        for r in rows(AUTO/case/'observations_forecasts.jsonl'):
            q=r['frontends']['delayed_feedback'];obs=r['observation'];z=obs['center'] if obs['detected'] else None
            value=bank.step(r['t'],z,None,False,q['valid'],q);pred[round(r['t'],8)]=value
        for r in old_scores:
            if r['case']!=case or r['frontend']!='event_force' or r['model']!='mmae_state_fresh' or r['nominal_h']!=.4 or r['target_state']!='present':continue
            v=pred.get(round(r.get('anchor_t',-100),8))
            for method in ['cv','trusted_acc','kf_tuned']:
                entry=dict(case=case,method=method,horizon=.4,anchor_t=r.get('anchor_t'),valid=bool(v and v['valid']))
                if entry['valid']:entry['fde']=float(np.linalg.norm(np.array(prior.evaluate(v['coefficients'][method],r['actual_h']))-r['target_center']))
                own.append(entry)
        save(OUT/'self'/case/'forecasts.json',list(pred.values()))
    save(OUT/'endpoint_details90.json',dict(fred=details,self=own))
    table=score88.summarise(details+own)
    test={m:float(np.mean([r['fde'] for r in details if r['case'] in ['fred8','fred9'] and r['method']==m and r['horizon']==.4 and r['valid']])) for m in METHODS}
    save(OUT/'scores90.json',dict(table=table,selected_sigma=selected['sigma'],test_04_mean_fde=test,fred_audit=audit,
        inference_frozen_before_score=True,selection_used_train_only=True,own_transfer_tuning=False,
        guarded_zero_coverage_test_means_no_RPM_gain=True,raw_q_failed_negative_controls=True,default_promoted=False))
    print('90 fixed test and automatic transfer complete',json.dumps(test),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','select','evaluate']);a=p.parse_args();globals()[a.action]()
