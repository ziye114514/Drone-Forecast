"""Mechanism check: the phase module now starts from >=2 qualified visible slots.

Synthetic only. Two static elliptical rings emit alternating-polarity events at
known rates, so the per-pixel temporal spectrum has a known peak. The check asks
four things: partial initialisation is accepted, hidden slots are not reported as
measured, a single slot is still refused, and the four-slot path is unchanged.
"""
import sys
import numpy as np
sys.path.insert(0,r"D:\drone forecast\event_drone_forecast")
from shared_phase_fft import SharedPhaseFFT

W,H=1280,720
DTYPE=[('x','u2'),('y','u2'),('p','i2'),('t','i8')]


def ring_events(center,frequency,cutoff,span=.16,axes=(32.,20.),samples=48):
    rows=[]
    steps=int(span*frequency)
    for k in range(steps):
        t=cutoff-span+k/frequency
        polarity=1 if k%2==0 else 0
        for j in range(samples):
            theta=2*np.pi*j/samples
            x=center[0]+axes[0]*np.cos(theta)
            y=center[1]+axes[1]*np.sin(theta)
            rows.append((int(round(x)),int(round(y)),polarity,int(round(t*1e6))))
    return np.array(rows,dtype=DTYPE)


def conic(center,major,minor,angle=0.):
    return dict(cx=center[0],cy=center[1],major=major,minor=minor,angle=angle,
        ratio=minor/major,mode='ellipse',current_observation=True)


def structure(hubs,observed,mode):
    return dict(mode=mode,model_centers=hubs,observed_model_ids=observed,geometry_epoch=0,
        axis_xy=[0.,-1.],axis_support=dict(valid=True,confidence=.9))


def main():
    hubs=np.array([[300.,200.],[600.,200.],[600.,400.],[300.,400.]])
    body=dict(detected=True,bbox=[240.,140.,660.,460.],center=[450.,300.])
    velocity=[0.,0.]
    cutoffs=[.06,.10,.14,.18,.22]
    events=[ring_events(hubs[0],200.,c) for c in cutoffs]+[ring_events(hubs[1],260.,c) for c in cutoffs]
    results_partial=[]
    module=SharedPhaseFFT()
    for index,cutoff in enumerate(cutoffs):
        window=np.concatenate([ring_events(hubs[0],200.,cutoff),ring_events(hubs[1],260.,cutoff)])
        rotors=[conic(hubs[0],64.,40.),conic(hubs[1],64.,40.)]
        results_partial.append(module.step(window,cutoff,body,
            structure(hubs,[0,1],'partial_two'),rotors,velocity))
    reasons=[r.get('reason') for r in results_partial]
    last=results_partial[-1]
    reported={slot['model_id']:slot for slot in last.get('rotors',[])}
    frequencies={key:(None if value.get('frequency_hz') is None else round(value['frequency_hz'],1))
        for key,value in reported.items()}
    qualified={key:value.get('qualified') for key,value in reported.items()}

    single=SharedPhaseFFT().step(ring_events(hubs[0],200.,.10),.10,body,
        structure(hubs,[0],'partial_two'),[conic(hubs[0],64.,40.)],velocity)

    four_results=[]
    module4=SharedPhaseFFT()
    for cutoff in cutoffs:
        window=np.concatenate([ring_events(hubs[i],200.+20*i,cutoff) for i in range(4)])
        four_results.append(module4.step(window,cutoff,body,structure(hubs,[0,1,2,3],'four'),
            [conic(hubs[i],64.,40.) for i in range(4)],velocity))

    checks=dict(
        partial_initialisation_accepted=reasons[0]!='awaiting_current_partial_calibration',
        init_reason_first=reasons[0],
        hidden_slots_not_measured=all(reported.get(i,{}).get('current_slot_observed') is False
            for i in (2,3) if i in reported),
        single_slot_still_refused=single.get('reason')=='awaiting_current_partial_calibration',
        four_slot_path_unchanged=four_results[-1].get('initialized') is True,
        partial_frequencies_hz=frequencies,
        partial_qualified=qualified,
        partial_measured_slots=sorted(reported),
        partial_reason_last=last.get('reason'),
        four_slot_reason_last=four_results[-1].get('reason'),
        note='synthetic static rings with alternating polarity; the reported frequency is the '
             'blade-phase spectrum peak, not a calibrated RPM')
    print(__import__('json').dumps(checks,ensure_ascii=False,indent=1))


if __name__=='__main__':
    main()
