"""CPU reference of DDHF per-pixel signed-event NDFT and harmonic comb.

Bezick & Sahin, arXiv:2603.08386, equations6-11. No author code copied.
This is an explicitly incomplete reproduction: K=128 and the searched base
modes are pilot choices not specified by the paper's parameter table. No GPU,
box-merging, inferred physical rims, or calibrated motor RPM is claimed here.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d


def harmonic_pixels(events,cutoff_s,window_us=33333,width=1280,height=720,modes=128):
    cutoff=round(cutoff_s*1e6);start=cutoff-window_us
    e=events[(events['t']>=start)&(events['t']<=cutoff)]
    pixel=e['y'].astype(np.int64)*width+e['x']
    counts=np.bincount(pixel,minlength=width*height)
    eligible=counts>=6
    indices=np.flatnonzero(eligible[pixel]);order=np.argsort(pixel[indices],kind='stable');indices=indices[order]
    key=pixel[indices]
    pixels,first,n=np.unique(key,return_index=True,return_counts=True)
    power=np.empty((len(pixels),modes-1),np.float32)
    bins=np.arange(1,modes,dtype=np.float32)
    for lo in range(0,len(pixels),256):
        hi=min(lo+256,len(pixels));a=int(first[lo]);b=int(first[hi]) if hi<len(pixels) else len(indices)
        selected=e[indices[a:b]]
        phase=(selected['t'].astype(np.float64)-start)*(2*np.pi/window_us)
        signal=np.exp((1j*phase[:,None]*bins).astype(np.complex64))
        signal*=2*selected['p'].astype(np.float32)[:,None]-1
        total=np.add.reduceat(signal,first[lo:hi]-a,axis=0)
        power[lo:hi]=np.abs(total)**2
    mask=np.zeros(width*height,np.uint8);frequency=np.zeros(width*height,np.float32)
    comb_map=np.zeros(width*height,np.float32);flatness_map=np.ones(width*height,np.float32)
    if len(pixels):
        epsilon=np.maximum(power.mean(1,keepdims=True)*1e-12,1e-12)
        flatness=np.exp(np.mean(np.log(np.maximum(power,epsilon)),axis=1))/np.maximum(power.mean(1),1e-12)
        nearby=maximum_filter1d(power,7,axis=1,mode='nearest')
        best=np.zeros(len(pixels));fundamental=np.zeros(len(pixels),np.int32)
        # Four harmonic windows must lie wholly within the non-DC band.
        # Candidate4..31 =>120..930Hz for33.333ms, a declared pilot choice.
        for base in range(4,(modes-4)//4+1):
            harmonic=base*np.arange(1,5)
            if harmonic[-1]+3>=modes:continue
            union=np.any(abs(bins[None]-harmonic[:,None])<=3,axis=0)
            noise=np.maximum(np.median(power[:,~union],axis=1),1e-12)
            score=nearby[:,harmonic-1].mean(1)/noise
            better=score>best;best[better]=score[better];fundamental[better]=base
        accepted=(flatness<=.89)&(best>=1.5)
        mask[pixels[accepted]]=255;frequency[pixels[accepted]]=fundamental[accepted]*1e6/window_us
        comb_map[pixels]=best;flatness_map[pixels]=flatness
    return dict(mask=mask.reshape(height,width),frequency_hz=frequency.reshape(height,width),
        comb_score=comb_map.reshape(height,width),spectral_flatness=flatness_map.reshape(height,width),
        eligible_mask=eligible.reshape(height,width),input_events=len(e),eligible_pixels=len(pixels),
        retained_pixels=int(np.count_nonzero(mask)),parameters=dict(window_us=window_us,min_events=6,
            modes=modes,flatness_max=.89,comb_min=1.5,harmonics=4,half_bandwidth_bins=3,
            base_modes=[4,(modes-4)//4],frequency_resolution_hz=1e6/window_us,
            exact_author_implementation=False,physical_rims_measured=False,real_RPM_measured=False))
