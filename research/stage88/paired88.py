"""Matched, causal periodic-input controls reusing 72/77/84, no new KF engine."""
from pathlib import Path
from collections import deque
import copy, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage84'),str(ROOT/'research/stage72')]
from batched84 import BatchedSeparatedBank
from increment72 import IncrementForecast, Gain, prior

SEEDS=(101,202,303)
ARMS=['event','constant','zero','lag200']+[f'past_shuffle{s}' for s in SEEDS]
METHODS=['cv','trusted_acc']+ARMS
PARAMS=dict(history_reservoir_s=2.,minimum_resampled_age_s=.2,shuffle_seeds=SEEDS,
    lag_s=.2,mode_clock='All controls share EVENT current eligibility and learned-update clock from77.',
    reuse='72 IncrementForecast/Gain; 77 independent state-controlled MMAE; 84 exact projection.',
    controls='Constant=ratio1, zero=no force innovation, lag=latest qualified ratio at least200ms old, '
             'shuffle=uniform sample from qualified ratios200ms..2s old. No future/circular shifts.',
    tuning='No new thresholds, gain bounds, process noises, transitions, or per-recording selector.')


class Paired:
    def __init__(self): self.reset()
    def reset(self):
        self.inc=IncrementForecast(); self.banks={k:BatchedSeparatedBank('mmae') for k in ARMS}
        self.gains={k:Gain() for k in ARMS if k not in ['event','constant','zero']}
        self.rng={f'past_shuffle{s}':np.random.default_rng(s) for s in SEEDS}
        self.reservoir=deque();self.previous_gain={};self.previous_history=np.zeros(2);self.last_seen=None

    def step(self,q,obs,body,activity,quality):
        t=q['t'];z=obs.get('center') if obs.get('detected') else None
        if self.last_seen is not None and t-self.last_seen>.5:self.reset()
        if z is not None:self.last_seen=t
        # Minimal envelope: old output maps are neither needed nor mutated.
        v=dict(t=t,valid=q['valid'],coefficients={k:copy.deepcopy(q['coefficients'][k]) for k in ['cv','trusted_acc'] if q['valid']},
               predictions={},feature_diagnostic=q['feature_diagnostic'])
        inc=self.inc.step(v,obs,body,activity,quality)['increment_diagnostic']
        real=inc['models']['increment_periodic'];old=self.previous_gain.get('event')
        eligible=bool(real['active'] and old is not None and old['updates']>=10 and real['delta'] is not None)
        fresh=bool(eligible and inc['direction']['current'] and inc['features']['periodic']['current'])
        n=self.inc.direction.normal.copy() if inc['direction']['active'] else None
        feedback=bool(q['valid'] and z is not None and obs.get('confidence',0)>.5)
        current=bool(inc['direction']['current'] and inc['direction']['quality']>.45 and inc['features']['periodic']['current'])
        while self.reservoir and t-self.reservoir[0][0]>2.+1e-8:self.reservoir.popleft()
        pool=[(stamp,value) for stamp,value in self.reservoir if t-stamp>=.2-1e-8]
        values={'lag200':pool[-1][1] if pool else 1.}
        sample_times={'lag200':pool[-1][0] if pool else None}
        for name,rng in self.rng.items():
            j=int(rng.integers(len(pool))) if pool else None
            values[name]=pool[j][1] if pool else 1.;sample_times[name]=pool[j][0] if pool else None
        gd={'event':real,'constant':inc['models']['increment_periodic_constant']}
        for name,gain in self.gains.items():
            if q['valid']:
                _,gd[name]=gain.step(t,q['coefficients']['trusted_acc'],n,values[name],real['active'],current,feedback,z)
            else:gd[name]=dict(updates=gain.updates,gain_px_s2=200*gain.theta,delta=None)
        history=np.asarray(q['coefficients']['trusted_acc']['acceleration']) if q['valid'] else np.zeros(2)
        coeff={k:copy.deepcopy(q['coefficients'][k]) for k in ['cv','trusted_acc'] if q['valid']};diag={}
        for name,bank in self.banks.items():
            innovation=np.zeros(2);g=self.previous_gain.get(name);d=gd.get(name,{}).get('delta')
            if fresh and name!='zero' and g is not None and d is not None:innovation=g['gain_px_s2']*np.asarray(d)
            bank.step(t,z,self.previous_history+innovation if fresh else np.zeros(2),history+innovation if fresh else np.zeros(2),fresh)
            if bank.bank is not None:
                c,bd=bank.coefficient()
                if q['valid']:coeff[name]=c if fresh else copy.deepcopy(q['coefficients']['trusted_acc'])
                diag[name]=dict(active=fresh,innovation=innovation.tolist(),mode_probability=bd['mode_probability'],
                    gain_used=None if g is None else g['gain_px_s2'])
            else:diag[name]=dict(active=False)
        if inc['features']['periodic']['current']:
            self.reservoir.append((t,inc['features']['periodic']['ratio']))
        for stamp in sample_times.values():assert stamp is None or stamp<=t-.2+1e-8
        self.previous_gain=copy.deepcopy(gd)
        if q['valid']:self.previous_history=history.copy()
        return dict(t=t,valid=q['valid'],coefficients=coeff,
            diagnostics=dict(fresh=fresh,eligible=eligible,direction=inc['direction'],periodic=inc['features']['periodic'],
                frequency=inc['frequency'],controls=diag,resampled_source_t=sample_times,
                event_gain_updates=real['updates'],event_gain=real['gain_px_s2']))


def minimal_q(q):
    return dict(t=q['t'],valid=q['valid'],coefficients=q['coefficients'],feature_diagnostic=q['feature_diagnostic'])


def controls():
    # Future-suffix changes must not alter any earlier prediction/control draw.
    def run(stop,future):
        m=Paired();result=[]
        for j in range(stop):
            t=(j+1)*.02;f=200+10*np.sin(t*4)+(future if j>=100 else 0)
            c=dict(position=[t*20,0.],velocity=[20.,0.],acceleration=[0.,0.])
            q=dict(t=t,valid=True,coefficients=dict(cv=c,trusted_acc=c),
                   feature_diagnostic=dict(axis_age_s=0.,models={'adaptive_axis_only':{'normal':[.6,-.8,0.]}}))
            obs=dict(detected=True,center=c['position'],confidence=1.)
            a=dict(density=10.,spectral_candidates=[dict(qualified=True,frequency_hz=f,tone_fraction=.8,phase_coherence=.95)])
            result.append(m.step(q,obs,dict(valid=False),a,.9))
        return result
    a=run(160,0);b=run(160,60);prefix=run(100,0)
    assert a[:100]==b[:100]==prefix
    for r in a:
        assert len({v['active'] for v in r['diagnostics']['controls'].values()})==1
    return dict(future_suffix_invariant=True,truncated_prefix_exact=True,shared_publication_clock=True,
                no_future_reservoir_samples=True,synthetic_ticks=160,mechanical_rpm_claim=False)
