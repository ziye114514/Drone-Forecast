"""Tone-origin counterexamples and temporal guard controls, before RAW scoring."""
import json
import numpy as np
from scipy.signal import lfilter
from activity68 import *

def run():
    result=[];n=1600;t=(np.arange(n)+.5)/10000.
    for kind in ['sinusoid','mixture','colored','burst','frequency_switch','common_flicker']:
        for seed in range(40):
            rng=np.random.default_rng(6800+seed);f=[119.3,167.7,287.1,523.4][seed%4]
            if kind=='sinusoid':z=np.cos(2*np.pi*f*t)
            elif kind=='mixture':z=(np.cos(2*np.pi*f*t)+np.cos(2*np.pi*(f+43)*t))/2
            elif kind=='colored':
                z=lfilter([1],[1,-.95],rng.normal(size=n));z=np.tanh(z/max(z.std(),1e-9))
            elif kind=='burst':z=np.exp(-.5*((t-.08)/.001)**2)
            elif kind=='frequency_switch':z=np.cos(2*np.pi*np.where(t<.08,f*t,f*.08+(f+81)*(t-.08)))
            else:f=100.;z=np.cos(2*np.pi*f*t)
            signal=np.stack([rng.poisson(3+2.7*z),rng.poisson(3-2.7*z)])
            if kind=='common_flicker':background=np.stack([rng.poisson(30+27*z),rng.poisson(30-27*z)])
            else:background=rng.poisson(3,size=(2,n))
            ff,pp=spectrum(signal[None]);q=peak(ff,pp[0]);candidates=[]
            if q:
                noise=max(float(np.median(pp[0,(ff>=60)&(ff<=1200)])),1e-12)
                for freq in q['top_peaks_hz']:
                    tt=tone(signal,freq);bg=tone(background,freq);snr=float(pp[0,np.argmin(abs(ff-freq))]/noise)
                    ok=snr>=8 and tt['tone_fraction']>=.15 and tt['phase_coherence']>=.8 and bg['tone_fraction']<tt['tone_fraction']*.5
                    candidates.append(dict(**tt,snr=snr,background_fraction=bg['tone_fraction'],qualified=bool(ok)))
            result.append(dict(kind=kind,seed=seed,accepted=any(q['qualified'] for q in candidates),candidates=candidates))
    # No future events may be used, and normalisation is stable after warm-up.
    dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);e=np.zeros(1,dtype);e['t']=20000
    refused=False
    try:Activity().step(e,None,.02,[0,0])
    except AssertionError:refused=True
    assert refused
    ref=Reference()
    for j in range(5):q=ref.step(j*.02,4.)
    assert q['active'] and q['ratio']==1. and not ref.step(.3,None)['active']
    summary={kind:dict(n=40,accepted=sum(r['accepted'] for r in result if r['kind']==kind)) for kind in sorted({r['kind'] for r in result})}
    return dict(table=summary,rows=result,future_events_refused=True,reference_expiry_verified=True,
        scope='Known count-rate signals; no hardware or rotor-origin truth. Fixed .15/.8/.5 gates are draft design choices, not tuned using future-position labels.')

if __name__=='__main__':
    import study68 as study
    study.verify();study.save(study.OUT/'controls68.json',run());print('68 controls saved',flush=True)
