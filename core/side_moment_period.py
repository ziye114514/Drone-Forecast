"""Experimental side-line spatial variance period, independent of hub IDs.

    Var[u] is invariant to common translation and event multiplicity. For an
    ideal projected two-blade diameter it contains cos(2*omega*t). This is an
    observable temporal component, not a verified mechanical-RPM conversion.
    Existing angular-phase/count estimators remain unchanged.
"""
import numpy as np
from scipy.signal import lombscargle


def binned_moment(events,cutoff,center,axis,velocity,span=.16,bin_us=250):
    start=round((cutoff-span)*1e6);end=round(cutoff*1e6)
    selected=events[(events['t']>=start)&(events['t']<end)]
    size=int(np.ceil((end-start)/bin_us));ids=((selected['t']-start)//bin_us).astype(int)
    axis=np.asarray(axis,float);axis/=np.linalg.norm(axis)
    relative=np.column_stack((selected['x'],selected['y'])).astype(float)-np.asarray(center)
    relative-=((selected['t']/1e6-cutoff)[:,None])*np.asarray(velocity)
    u=relative@axis
    counts=np.bincount(ids,minlength=size).astype(float)
    first=np.bincount(ids,weights=u,minlength=size);second=np.bincount(ids,weights=u*u,minlength=size)
    mean=np.divide(first,counts,out=np.zeros(size),where=counts>0)
    variance=np.maximum(0.,np.divide(second,counts,out=np.zeros(size),where=counts>0)-mean*mean)
    valid=counts>=8
    times=(start+(np.arange(size)+.5)*bin_us)/1e6
    return times,variance,valid,counts


def component(times,values,valid):
    valid=np.asarray(valid,bool);t=np.asarray(times)[valid];y=np.asarray(values)[valid]
    if len(t)<64 or np.mean(valid)<.6:return dict(qualified=False,reason='sparse_moment_bins',bins=int(valid.sum()),coverage=float(np.mean(valid)))
    t=t-t[0]
    # Irregularly missing bins stay missing; no zero-filled artificial bursts.
    y=y-np.column_stack((np.ones(len(t)),t))@np.linalg.lstsq(np.column_stack((np.ones(len(t)),t)),y,rcond=None)[0]
    if np.std(y)<=1e-7*max(1.,float(np.mean(abs(np.asarray(values)[valid])))):
        return dict(qualified=False,reason='constant_spatial_moment')
    grid=np.arange(60.,1200.001,3.125)
    power=lombscargle(t,y,2*np.pi*grid,normalize='normalize',floating_mean=True)
    index=int(np.nanargmax(power));peak=float(grid[index]);ratio=float(power[index]/max(float(np.median(power)),1e-12))
    return dict(qualified=bool(ratio>=8. and power[index]>=.2),reason='periodic_spatial_variance' if ratio>=8. and power[index]>=.2 else 'weak_moment_spectrum',
        frequency_hz=peak,normalized_power=float(power[index]),peak_to_median=ratio,bins=int(valid.sum()),coverage=float(np.mean(valid)))


def observe(events,t,center,axis,velocity):
    times,value,valid,counts=binned_moment(events,t,center,axis,velocity)
    full=component(times,value,valid);mid=len(times)//2
    halves=[component(times[a:b],value[a:b],valid[a:b]) for a,b in ((0,mid),(mid,len(times)))]
    stable=bool(full.get('qualified') and all(h.get('qualified') and abs(h['frequency_hz']-full['frequency_hz'])<=25. for h in halves))
    return dict(qualified=stable,reason='current_spatial_variance_and_half_window_agreement' if stable else 'moment_or_half_window_unqualified',
        full=full,halves=halves,bin_us=250,span_s=.16,mechanical_RPM_measured=False,thrust_measured=False,
        current_hub_required=False,missing_bins_zero_filled=False),dict(times=times.tolist(),variance=value.tolist(),valid=valid.tolist(),counts=counts.tolist())
