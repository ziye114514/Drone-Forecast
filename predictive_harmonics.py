"""Discover periodic templates in the past half-window, validate on the later half.

DDHF-inspired experimental candidate evidence, not an author reproduction.
Discovery never sees validation polarities. A one-sided bounded-sign score
tests the frozen template; its bound assumes independent stationary random
polarities conditional on timestamps, NOT calibrated event-camera noise.
Periodic illumination is expected to pass and is not called a rotor here.
"""
import numpy as np

BASE_FREQUENCIES=np.arange(120.,961.,15.)
HARMONICS=np.arange(1.,5.)
PARAMS=dict(window_us=33333,discovery_fraction=.5,frequency_step_hz=15.,
    frequency_min_hz=120.,frequency_max_hz=960.,harmonics=4,templates_per_pixel=2,
    min_events_per_half=6,min_span_fraction_per_half=.5,min_quarters_per_half=2,
    conditional_random_sign_bound=.01,world_false_alarm_rate_calibrated=False,
    periodic_flicker_is_not_distinguished_from_rotors=True)


def predictive_harmonics(events,cutoff_s,window_us=33333,width=1280,height=720,base_frequencies=None):
    frequencies=BASE_FREQUENCIES if base_frequencies is None else np.asarray(base_frequencies,float)
    if frequencies.ndim!=1 or not len(frequencies) or not np.isfinite(frequencies).all() or np.any(frequencies<=0):
        raise ValueError('Nonempty finite positive candidate frequencies required')
    end=round(cutoff_s*1e6);begin=end-window_us;mid=begin+.5*window_us
    e=events[(events['t']>=begin)&(events['t']<end)]
    pix=e['y'].astype(np.int64)*width+e['x'];half=e['t']>=mid
    count=np.bincount(pix+half.astype(np.int64)*width*height,minlength=2*width*height).reshape(2,-1)
    eligible=np.all(count>=PARAMS['min_events_per_half'],axis=0)
    index=np.flatnonzero(eligible[pix]);index=index[np.argsort(pix[index],kind='stable')]
    pixels,first,counts=np.unique(pix[index],return_index=True,return_counts=True)
    n=len(pixels);score=np.zeros(n);frequency=np.zeros(n);candidate_f=np.zeros((n,2));bound=np.ones((n,2));spatial_phase=np.zeros(n)
    distribution=np.zeros(n,bool);train_count=np.zeros(n,np.int32);validation_count=np.zeros(n,np.int32)
    # Original event timestamps, not binned counts or artificial uniform samples.
    for lo in range(0,n,128):
        hi=min(lo+128,n);a=first[lo];b=first[hi] if hi<n else len(index)
        selected=e[index[a:b]];local=np.repeat(np.arange(hi-lo),counts[lo:hi])
        train=selected['t']<mid;validation=~train
        train_t=(selected['t'][train]-begin)/1e6;train_p=2*selected['p'][train].astype(float)-1
        train_id=local[train];nt=np.bincount(train_id,minlength=hi-lo);train_count[lo:hi]=nt
        mean_p=np.bincount(train_id,weights=train_p,minlength=hi-lo)/nt
        phase=2*np.pi*train_t[:,None,None]*frequencies[None,:,None]*HARMONICS[None,None,:]
        z=np.add.reduceat((train_p-mean_p[train_id])[:,None,None]*np.exp(-1j*phase),
            np.cumsum(np.r_[0,nt[:-1]]),axis=0)
        z/=nt[:,None,None]
        power=np.sum(abs(z)**2,axis=2);first_choice=np.argmax(power,axis=1)
        choices=[first_choice]
        separated=abs(np.log(frequencies[None]/frequencies[first_choice,None]))>.2
        choices.append(np.argmax(np.where(separated,power,-1),axis=1))
        tv=(selected['t'][validation]-begin)/1e6;pv=2*selected['p'][validation].astype(float)-1;iv=local[validation]
        nv=np.bincount(iv,minlength=hi-lo);validation_count[lo:hi]=nv
        for choice_index,choice in enumerate(choices):
            f=frequencies[choice];templates=z[np.arange(hi-lo),choice]
            candidate_f[lo:hi,choice_index]=f
            phase=2*np.pi*tv[:,None]*f[iv,None]*HARMONICS[None]
            predicted=np.real(np.sum(templates[iv]*np.exp(1j*phase),axis=1))
            mean=np.bincount(iv,weights=predicted,minlength=hi-lo)/nv
            centered=predicted-mean[iv]
            projection=np.bincount(iv,weights=pv*centered,minlength=hi-lo)
            energy=np.bincount(iv,weights=centered**2,minlength=hi-lo)
            exponent=np.where(projection>0,projection**2/(2*np.maximum(energy,1e-12)),0.)
            adjusted=np.minimum(1.,2*np.exp(-exponent));bound[lo:hi,choice_index]=adjusted
            improvement=exponent>score[lo:hi]
            score[lo:hi][improvement]=exponent[improvement];frequency[lo:hi][improvement]=f[improvement]
            spatial_phase[lo:hi][improvement]=np.angle(templates[improvement,0])
        for j in range(hi-lo):
            times=selected['t'][local==j]
            parts=[times[times<mid],times[times>=mid]]
            distribution[lo+j]=all(np.ptp(part)>=.5*(window_us/2) and
                len(np.unique(np.minimum(3,((part-(begin if h==0 else mid))/(window_us/8)).astype(int))))>=2
                for h,part in enumerate(parts))
    accepted=distribution&(np.min(bound,axis=1)<=PARAMS['conditional_random_sign_bound'])
    def dense(values,dtype=float):
        plane=np.zeros(width*height,dtype);plane[pixels]=values;return plane.reshape(height,width)
    return dict(mask=dense(accepted,np.uint8)*255,evidence=dense(score),
        frequency_hz=dense(np.where(accepted,frequency,0.)),candidate_frequency_hz=candidate_f,
        candidate_bound=bound,pixels=pixels,spatial_phase=dense(spatial_phase),
        discovery_counts=dense(train_count,np.int32),validation_counts=dense(validation_count,np.int32),
        temporal_coverage=dense(distribution,np.uint8),eligible_pixels=n,
        accepted_pixels=int(accepted.sum()),parameters=dict(PARAMS,window_us=window_us,
            base_frequency_hypotheses_hz=frequencies.tolist(),candidate_band_from_past=base_frequencies is not None),
        physical_rim_measured=False,mechanical_RPM_measured=False,
        independent_frequency_selection_and_validation=True)
