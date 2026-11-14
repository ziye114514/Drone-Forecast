"""Untuned extra rate-process checks after the original66/67 diagnostic controls."""
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
import study67 as current
from multipeak67 import analyse,PARAMS


def run():
    current.verify();path=current.OUT/'holdout_rate_controls67.json';assert not path.exists()
    n=PARAMS['window_us']//PARAMS['bin_us'];t=(np.arange(n)+.5)/10000.;signals=[];cases=[]
    for kind in ['new_mixtures','aperiodic_colored_rate','frequency_switch','uniform_flicker']:
        for seed in range(40):
            rng=np.random.default_rng(67100+seed);f=[119.3,167.7,287.1,523.4][seed%4]
            second=f+[19.,43.,81.,137.][seed%4]
            if kind=='new_mixtures':
                ratio=[.35,.7,1.,1.5][(seed//4)%4]
                z=(np.cos(2*np.pi*f*t)+ratio*np.cos(2*np.pi*second*t))/(1+ratio)
                a=3+2.7*z;b=3-2.7*z;truth=[f,second]
            elif kind=='aperiodic_colored_rate':
                z=lfilter([1.],[1.,-.95],rng.normal(size=n));z/=max(z.std(),1e-9)
                a=3+2.7*np.tanh(z);b=6-a;truth=[]
            elif kind=='frequency_switch':
                z=np.cos(2*np.pi*np.where(t<.08,f*t,f*.08+second*(t-.08)))
                a=3+2.7*z;b=3-2.7*z;truth=[f,second]
            else:
                a=3+2.7*np.cos(2*np.pi*100*t);b=3-2.7*np.cos(2*np.pi*100*t);truth=[100.]
            signals.append(np.stack([rng.poisson(a),rng.poisson(b)]));cases.append(dict(kind=kind,seed=seed,
                known_rate_components_hz=truth,rotor_origin_known=False,
                interpretation='Flicker is periodic but not rotor evidence; frequency switch is not stationary.'))
    signals=np.asarray(signals);result=analyse(signals,signals.sum(axis=(1,2)),np.full(len(signals),PARAMS['window_us']))
    for c,r in zip(cases,result):
        c.update(r)
        if c['kind']=='new_mixtures':
            values=[v['frequency_hz'] for v in r['matched_frequencies']]
            c['known_components_matched']=sum(any(abs(h-f)<=r.get('frequency_resolution_hz',6.25) for h in values) for f in c['known_rate_components_hz'])
    summary=[]
    for kind in ['new_mixtures','aperiodic_colored_rate','frequency_switch','uniform_flicker']:
        rr=[r for r in cases if r['kind']==kind]
        summary.append(dict(kind=kind,cases=len(rr),spectral=sum(r['valid'] for r in rr),
            dominant_half=sum(r['stable'] for r in rr),multi_half=sum(r['multi_half_stable'] for r in rr),
            known_components_matched=sum(r.get('known_components_matched',0) for r in rr) if kind=='new_mixtures' else None))
    current.save(path,dict(helper_sha256=current.sha(__file__),parameters_unchanged=True,table=summary,rows=cases,
        scope='Additional unmodified-algorithm rate-process checks, no hardware/rotor truth. '
              'These frequencies/noise processes were not in initial66/67 controls. '
              'Reporting this does not turn known development RAW into held-out data.'))
    print(summary,flush=True)


if __name__=='__main__':run()
