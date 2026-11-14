"""CUDA batched equivalent of predictive_harmonics; caller owns the GPU process.

Native timestamps remain unbinned. The component includes CPU grouping and
host/device transfers; it is not the RAW/NN/geometry/forecast system latency.
"""
import numpy as np
from predictive_harmonics import BASE_FREQUENCIES,HARMONICS,PARAMS


def predictive_harmonics_cuda(events,cutoff_s,window_us=33333,width=1280,height=720,base_frequencies=None,pixel_mask=None):
    import torch
    end=round(cutoff_s*1e6);begin=end-window_us;mid=begin+.5*window_us
    frequencies=BASE_FREQUENCIES if base_frequencies is None else np.asarray(base_frequencies,float)
    e=events[(events['t']>=begin)&(events['t']<end)]
    pix=e['y'].astype(np.int64)*width+e['x'];half=e['t']>=mid
    count=np.bincount(pix+half.astype(np.int64)*width*height,minlength=2*width*height).reshape(2,-1)
    eligible=np.all(count>=PARAMS['min_events_per_half'],axis=0)
    if pixel_mask is not None:
        pixel_mask=np.asarray(pixel_mask,bool)
        if pixel_mask.shape!=(height,width):raise ValueError('Pixel selection must match the native sensor size')
        eligible&=pixel_mask.ravel()
    index=np.flatnonzero(eligible[pix]);index=index[np.argsort(pix[index],kind='stable')]
    pixels,first,counts=np.unique(pix[index],return_index=True,return_counts=True);n=len(pixels)
    score=np.zeros(n);frequency=np.zeros(n);spatial_phase=np.zeros(n);bound=np.ones((n,2));candidate_f=np.zeros((n,2))
    distribution=np.zeros(n,bool)
    selected=e[index];ids=np.repeat(np.arange(n),counts);train=selected['t']<mid
    it=ids[train];iv=ids[~train]
    nt=np.bincount(it,minlength=n);nv=np.bincount(iv,minlength=n)
    if n:
        assert np.all(nt>0) and np.all(nv>0) and nt.sum()==len(it) and nv.sum()==len(iv)
        ts=selected['t'];ts_train=ts[train];ts_validation=ts[~train]
        first_t=np.cumsum(np.r_[0,nt[:-1]]);first_v=np.cumsum(np.r_[0,nv[:-1]])
        span_t=ts_train[first_t+nt-1]-ts_train[first_t];span_v=ts_validation[first_v+nv-1]-ts_validation[first_v]
        quarter=np.minimum(7,((ts-begin)/(window_us/8)).astype(int))
        occupancy=np.bincount(ids*8+quarter,minlength=n*8).reshape(n,2,4)>0
        distribution=(span_t>=window_us/4)&(span_v>=window_us/4)&np.all(occupancy.sum(2)>=2,axis=1)
        dev=torch.device('cuda')
        lengths_t=torch.as_tensor(nt,device=dev);lengths_v=torch.as_tensor(nv,device=dev)
        train_p=torch.as_tensor(2*selected['p'][train].astype(np.float32)-1,device=dev)
        val_p=torch.as_tensor(2*selected['p'][~train].astype(np.float32)-1,device=dev)
        ti=torch.as_tensor(it,device=dev);vi=torch.as_tensor(iv,device=dev)
        tt=torch.as_tensor((ts_train.astype(np.float64)-begin).astype(np.float32)/1e6,device=dev)
        vt=torch.as_tensor((ts_validation.astype(np.float64)-begin).astype(np.float32)/1e6,device=dev)
        f=torch.as_tensor(frequencies,dtype=torch.float32,device=dev);h=torch.as_tensor(HARMONICS,dtype=torch.float32,device=dev)
        ntf=lengths_t.float();nvf=lengths_v.float()
        mean=torch.segment_reduce(train_p,'sum',lengths=lengths_t,unsafe=True)/ntf
        weights=train_p-mean[ti]
        # Frequency blocks bound temporary memory. Segment reductions use the
        # sorted original pixel groups, avoiding event-wise global atomics.
        zr=torch.empty((n,len(frequencies),4),device=dev);zi=torch.empty_like(zr)
        for lo in range(0,len(frequencies),64):
            hi=min(lo+64,len(frequencies));omega=2*torch.pi*f[lo:hi,None]*h[None]
            phase=tt[:,None,None]*omega[None]
            zr[:,lo:hi]=torch.segment_reduce(weights[:,None,None]*torch.cos(phase),'sum',lengths=lengths_t,unsafe=True)/ntf[:,None,None]
            zi[:,lo:hi]=torch.segment_reduce(-weights[:,None,None]*torch.sin(phase),'sum',lengths=lengths_t,unsafe=True)/ntf[:,None,None]
        power=(zr*zr+zi*zi).sum(2);choice0=power.argmax(1)
        separated=torch.abs(torch.log(f[None]/f[choice0,None]))>.2
        choice1=torch.where(separated,power,-torch.ones_like(power)).argmax(1)
        choices=torch.stack([choice0,choice1],1);ar=torch.arange(n,device=dev)[:,None]
        chosen_f=f[choices];tr=zr[ar,choices];im=zi[ar,choices]
        phase=2*torch.pi*vt[:,None,None]*chosen_f[vi,:,None]*h[None,None]
        predicted=(tr[vi]*torch.cos(phase)-im[vi]*torch.sin(phase)).sum(2)
        mean=torch.segment_reduce(predicted,'sum',lengths=lengths_v,unsafe=True)/nvf[:,None]
        centered=predicted-mean[vi]
        projection=torch.segment_reduce(val_p[:,None]*centered,'sum',lengths=lengths_v,unsafe=True)
        energy=torch.segment_reduce(centered*centered,'sum',lengths=lengths_v,unsafe=True)
        exponent=torch.where(projection>0,projection*projection/(2*energy.clamp_min(1e-12)),0.)
        adjusted=(2*torch.exp(-exponent)).clamp_max(1.)
        which=exponent.argmax(1);ar1=torch.arange(n,device=dev)
        score=exponent[ar1,which].cpu().numpy();frequency=chosen_f[ar1,which].cpu().numpy()
        spatial_phase=torch.atan2(im[ar1,which,0],tr[ar1,which,0]).cpu().numpy()
        bound=adjusted.cpu().numpy();candidate_f=chosen_f.cpu().numpy()
    accepted=distribution&(bound.min(1)<=PARAMS['conditional_random_sign_bound'])
    def dense(values,dtype=float):
        out=np.zeros(width*height,dtype);out[pixels]=values;return out.reshape(height,width)
    return dict(mask=dense(accepted,np.uint8)*255,evidence=dense(score),
        frequency_hz=dense(np.where(accepted,frequency,0.)),candidate_frequency_hz=candidate_f,
        candidate_bound=bound,pixels=pixels,spatial_phase=dense(spatial_phase),
        temporal_coverage=dense(distribution,np.uint8),eligible_pixels=n,accepted_pixels=int(accepted.sum()),
        physical_rim_measured=False,mechanical_RPM_measured=False,
        independent_frequency_selection_and_validation=True,parameters=dict(PARAMS,window_us=window_us,
            spatial_pixel_selection=pixel_mask is not None,all_timestamps_at_each_selected_pixel_retained=True))
