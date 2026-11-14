"""Freeze separate discovery ON/OFF phase offsets, then validate late events.

An event polarity marks different contrast edges, so their phases need not be
equal. Averaging both polarities before checking coherence can cancel genuine
rotation. Neither late polarity updates its phase, frequency or homography.
The existing 0.5 coherence/agreement/contrast and nine-sector gates are retained
for EACH polarity; this is not a lower-confidence acceptance threshold.
"""
import numpy as np


def validate(fitted, native, xy, early, *, angular_order, rotation_sign):
    first=int(native['t'][early][0]);last=int(native['t'][early][-1])
    inv=np.linalg.inv(np.asarray(fitted['H']))
    point=np.c_[xy,np.ones(len(xy))]@inv.T
    finite=np.isfinite(point).all(1)&(abs(point[:,2])>1e-9)
    normalized=np.zeros((len(xy),2));normalized[finite]=point[finite,:2]/point[finite,2,None]
    radius=np.linalg.norm(normalized,axis=1)
    valid=finite&(radius>=.2)&(radius<=1.5)
    weight=np.exp(-.5*((radius-1)/.15)**2)
    angle=np.arctan2(rotation_sign*normalized[:,1],normalized[:,0])
    opposite=np.arctan2(-rotation_sign*normalized[:,1],normalized[:,0])
    tl=(last-first)/1e6;age=(native['t']-last)/1e6
    phase_last=fitted['phi0']+fitted['angular_velocity_rad_s']*tl+.5*fitted['alpha']*tl*tl
    phi=phase_last+fitted['angular_velocity_rad_s']*age+.5*fitted['alpha']*age*age-angular_order*fitted['q'][1]
    residual=np.exp(1j*(phi-angular_order*angle))
    wrong=np.exp(1j*(phi-angular_order*opposite))
    polarities=[]
    for p in (0,1):
        values=[]
        for mask in (early,~early):
            keep=valid&mask&(native['p']==p);w=weight[keep]
            z=np.sum(w*residual[keep])/max(w.sum(),1e-12)
            other=np.sum(w*wrong[keep])/max(w.sum(),1e-12)
            sectors=len(np.unique(np.floor((angle[keep&(weight>=.1)]+np.pi)*12/(2*np.pi)).astype(int)%12))
            values.append(dict(complex_mean=[float(z.real),float(z.imag)],concentration=float(abs(z)),
                opposite_power=float(abs(other)**2),sectors=sectors,events=int(keep.sum())))
        a,b=values;z=complex(*a['complex_mean']);zz=complex(*b['complex_mean'])
        agreement=float(np.real(zz*np.conj(z))/max(abs(z),1e-12))
        power=abs(zz)**2;other=b['opposite_power'];contrast=float((power-other)/max(power+other,1e-12))
        good=bool(min(a['events'],b['events'])>=32 and a['concentration']>=.5 and b['concentration']>=.5
            and agreement>=.5 and contrast>=.5 and b['sectors']>=9)
        polarities.append(dict(polarity=p,train=a,validation=b,phase_agreement=agreement,
            direction_contrast=contrast,qualified=good))
    return dict(qualified=all(x['qualified'] for x in polarities),polarities=polarities,
        policy='both_current_polarities_with_discovery_frozen_offsets',
        validation_updates_phase_or_pose=False,coherence_threshold=.5,sectors_threshold=9)
