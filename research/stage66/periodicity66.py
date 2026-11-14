"""Local ON/OFF spectral candidates without ellipse centers or rotor identities."""
import numpy as np
from scipy.signal import periodogram,find_peaks
from event_drone_forecast.paper_config import PaperConfig
CFG=PaperConfig()
PARAMS=dict(window_us=CFG.frequency_window_us,bin_us=100,grid=[6,3],
           min_events=100,min_window_span_fraction=.6,snr_min=8.,
           min_hz=CFG.frequency_min_hz,max_hz=CFG.frequency_max_hz,
           max_buffer_events=CFG.max_buffer_events,
           interpretation='Local brightness-change periodicity candidates, not mechanical '
                          'RPM, rotor IDs or aggregate thrust. Half-window agreement '
                          'is an extra diagnostic; all harmonic ambiguity retained.')


def spectrum(signals,fs=10000.):
    f,p=periodogram(np.asarray(signals,float)-np.mean(signals,axis=-1,keepdims=True),
                    fs=fs,window='hann',detrend='linear',axis=-1)
    return f,p.sum(axis=-2)


def peak(f,p):
    allowed=(f>=PARAMS['min_hz'])&(f<=PARAMS['max_hz']);indices=np.flatnonzero(allowed)
    if not len(indices) or p[indices].max()<=0:return None
    noise=max(float(np.median(p[indices])),1e-12)
    peaks,_=find_peaks(p,distance=2);peaks=[k for k in peaks if allowed[k]]
    peaks=sorted(peaks,key=lambda k:p[k],reverse=True)[:5]
    if not peaks:peaks=[int(indices[np.argmax(p[indices])])]
    k=peaks[0]
    return dict(peak_hz=float(f[k]),peak_to_noise=float(p[k]/noise),
                top_peaks_hz=[float(f[j]) for j in peaks],
                frequency_resolution_hz=float(f[1]-f[0]))


def analyse(signals,event_counts,spans_us):
    signals=np.asarray(signals,float);f,p=spectrum(signals);middle=signals.shape[-1]//2
    first_f,first_p=spectrum(signals[...,:middle]);last_f,last_p=spectrum(signals[...,middle:])
    result=[]
    for i in range(len(signals)):
        if event_counts[i]<PARAMS['min_events']:
            result.append(dict(valid=False,stable=False,reason='too_few_events'));continue
        if spans_us[i]<PARAMS['window_us']*PARAMS['min_window_span_fraction']:
            result.append(dict(valid=False,stable=False,reason='window_warming_up'));continue
        q=peak(f,p[i])
        if q is None:
            result.append(dict(valid=False,stable=False,reason='empty_spectrum'));continue
        early=peak(first_f,first_p[i]);late=peak(last_f,last_p[i])
        valid=q['peak_to_noise']>=PARAMS['snr_min'];half_tolerance=float(first_f[1]-first_f[0])
        stable=bool(valid and early and late and early['peak_to_noise']>=PARAMS['snr_min'] and late['peak_to_noise']>=PARAMS['snr_min']
                    and abs(early['peak_hz']-q['peak_hz'])<=half_tolerance and abs(late['peak_hz']-q['peak_hz'])<=half_tolerance)
        q.update(valid=valid,stable=stable,reason='spectral_candidate' if valid else 'weak_periodicity',
                 first_half=early,last_half=late,harmonic_ambiguous=True,
                 harmonic_candidates_hz=[q['peak_hz']*v for v in [.5,1.,2.] if PARAMS['min_hz']<=q['peak_hz']*v<=PARAMS['max_hz']],
                 event_support=int(event_counts[i]),event_span_us=int(spans_us[i]))
        result.append(q)
    return result


def tile_measurement(events,bbox,t,velocity):
    start=round(t*1e6)-PARAMS['window_us'];n=PARAMS['window_us']//PARAMS['bin_us']
    gx,gy=PARAMS['grid'];b=np.asarray(bbox,float);size=b[2:]-b[:2]
    if np.any(size<12):return dict(valid=False,reason='small_roi',tiles=[])
    age=t-events['t']/1e6
    x=events['x'].astype(float)+age*velocity[0];y=events['y'].astype(float)+age*velocity[1]
    bins=((events['t']-start)//PARAMS['bin_us']).astype(int)
    keep=(bins>=0)&(bins<n)&(x>=b[0])&(x<b[2])&(y>=b[1])&(y<b[3])
    x=x[keep];y=y[keep];e=events[keep];bins=bins[keep]
    ix=np.minimum(gx-1,((x-b[0])/size[0]*gx).astype(int));iy=np.minimum(gy-1,((y-b[1])/size[1]*gy).astype(int));cell=iy*gx+ix
    polarity=(e['p']>0).astype(int)
    signals=np.bincount((cell*2+polarity)*n+bins,minlength=gx*gy*2*n).reshape(gx*gy,2,n)
    support=signals.sum(axis=(1,2));low=np.full(gx*gy,round(t*1e6),np.int64);high=np.full(gx*gy,start,np.int64)
    np.minimum.at(low,cell,e['t']);np.maximum.at(high,cell,e['t'])
    measurements=analyse(signals,support,np.maximum(high-low,0))
    for j,m in enumerate(measurements):m.update(tile=j,column=j%gx,row=j//gx)
    return dict(valid=True,reason='local_tile_spectra',bbox=b.tolist(),tiles=measurements,
                spectral_candidates=sum(q['valid'] for q in measurements),
                half_stable_candidates=sum(q['stable'] for q in measurements),
                ROI_events=len(e),mechanical_RPM_or_thrust_estimated=False)


def controlled():
    n=PARAMS['window_us']//PARAMS['bin_us'];t=(np.arange(n)+.5)/10000.;signals=[];cases=[]
    for kind in ['periodic','constant_poisson','single_burst','one_step','mixed_periodic']:
        for seed in range(32):
            rng=np.random.default_rng(6600+seed);f=[143.,180.,250.,397.][seed%4]
            if kind=='periodic':a=2.+1.8*np.cos(2*np.pi*f*t);b=2.-1.8*np.cos(2*np.pi*f*t)
            elif kind=='constant_poisson':a=np.full(n,2.);b=a
            elif kind=='single_burst':a=1.+25*np.exp(-.5*((t-.08)/.001)**2);b=a
            elif kind=='one_step':a=1.+8*(t>.08);b=a
            else:a=3.+1.3*np.cos(2*np.pi*143*t)+1.3*np.cos(2*np.pi*197*t);b=6-a
            signals.append(np.stack([rng.poisson(a),rng.poisson(b)]));cases.append(dict(kind=kind,seed=seed,true_rate_hz=f if kind=='periodic' else None))
    signals=np.asarray(signals);rr=analyse(signals,signals.sum(axis=(1,2)),np.full(len(signals),PARAMS['window_us']))
    for c,r in zip(cases,rr):
        c.update(r)
        if c['true_rate_hz'] is not None and 'peak_hz' in c:c['rate_error_hz']=abs(c['peak_hz']-c['true_rate_hz'])
    # Batched FFT must retain original per-polarity periodogram arithmetic.
    f,p=spectrum(signals[:1]);manual=[]
    for signal in signals[0]:
        signal=signal.astype(float);signal-=signal.mean();ff,power=periodogram(signal,fs=10000.,window='hann',detrend='linear');manual.append(power)
    np.testing.assert_allclose(p[0],sum(manual),rtol=1e-12,atol=1e-12)
    return dict(rows=cases,vectorized_spectrum_matches_original_polarity_loop=True,
                scope='Known count-rate process only, not blade geometry or real mechanical RPM truth. '
                      'Burst/step/Poisson controls estimate failure of periodicity gates, not field false-positive rates.')
