"""Unsigned local event-timing periodicity with a frozen earlier-half template.

Spatial pooling avoids requiring repeated ON/OFF switches at each pixel.
Conditioning on 1ms-block counts rejects a slowly varying moving-edge envelope.
This is regional periodic evidence, NOT individually confirmed pixels or RPM.
Original implementation inspired by EEPPR spatial repetition and DDHF combs.
The conditional independent-uniform-event null is not a calibrated camera FAR.
"""
from functools import lru_cache
import numpy as np
from predictive_harmonics import BASE_FREQUENCIES

PARAMS=dict(cell_px=16,bins_per_half=256,nuisance_block_bins=16,
    min_events_per_half=6,min_spatial_pixels=12,templates=2,family_alpha=.01,harmonics=4,
    repeated_pixel_votes_capped_per_nuisance_block=True,
    vote_selection='minimum_fixed_pseudorandom_priority_independent_of_event_time',
    conditional_null='independent event votes with unknown nonnegative linear intensity within each short block',
    nuisance_degree=1,variance_bound='worst-case within-block second moment',
    camera_false_alarm_calibrated=False)

HISTOGRAM_ENGINE=None

def priorities(n):
    z=np.arange(n,dtype=np.uint64)+np.uint64(0x9e3779b97f4a7c15+194)
    z=(z^(z>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
    z=(z^(z>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    return z^(z>>np.uint64(31))

@lru_cache(maxsize=4)
def bank(window_us):
    n=PARAMS['bins_per_half'];dt=window_us/(2*n*1e6)
    times=(np.arange(n,dtype=np.float32)+.5)*dt
    freq=BASE_FREQUENCIES[:,None]*np.arange(1,5)[None]
    phase=2*np.pi*times[:,None]*freq.ravel()[None]
    validation=np.exp(2j*np.pi*(times[None,:,None]+window_us/2e6)*freq[:,None,:]).astype(np.complex64)
    return times,dt,np.exp(-1j*phase).astype(np.complex64),freq,validation

def measure(events,cutoff_s,width=1280,height=720,window_us=33333,candidate_roi=None):
    end=round(cutoff_s*1e6);begin=end-window_us;n=PARAMS['bins_per_half'];cell=PARAMS['cell_px']
    cols=(width+cell-1)//cell;rows=(height+cell-1)//cell;ng=cols*rows*2
    e=events[(events['t']>=begin)&(events['t']<end)]
    if HISTOGRAM_ENGINE is not None:
        hist,spatial=HISTOGRAM_ENGINE(e,begin,end,width,height,cell,n,PARAMS['nuisance_block_bins'])
    else:
        native_pixel=e['y'].astype(np.int64)*width+e['x']
        tick=np.minimum(2*n-1,((e['t']-begin)*2*n//window_us)).astype(np.int64)
        # Select an effectively uniform event, NOT the first in each block:
        # first-event capping makes dense random input appear960Hz periodic.
        nuisance=2*n//PARAMS['nuisance_block_bins']
        keys=(native_pixel*2+(e['p']>0))*nuisance+tick//PARAMS['nuisance_block_bins']
        order=np.lexsort((np.arange(len(e)),priorities(len(e)),keys))
        keep=order[np.r_[True,keys[order][1:]!=keys[order][:-1]]] if len(e) else order
        e=e[keep];native_pixel=native_pixel[keep];tick=tick[keep]
        ids=((e['y'].astype(np.int64)//cell)*cols+e['x']//cell)*2+(e['p']>0)
        hist=np.bincount(ids*(2*n)+tick,minlength=ng*2*n).reshape(ng,2*n).astype(np.float32)
        unique_pixel_pol=np.unique(native_pixel*2+(e['p']>0));up=unique_pixel_pol//2
        unique_ids=((up//width//cell)*cols+(up%width)//cell)*2+unique_pixel_pol%2
        spatial=np.bincount(unique_ids,minlength=ng)
    a,b=hist[:,:n],hist[:,n:];na=a.sum(1);nb=b.sum(1)
    eligible=(na>=6)&(nb>=6)&(spatial>=PARAMS['min_spatial_pixels'])
    if candidate_roi is not None:
        x,y,X,Y=candidate_roi;cell_id=np.arange(ng)//2
        xx=(cell_id%cols)*cell;yy=(cell_id//cols)*cell
        eligible&=(xx<X)&(xx+cell>x)&(yy<Y)&(yy+cell>y)
    selected=np.flatnonzero(eligible);mask=np.zeros((height,width),bool)
    diagnostic=dict(groups_tested=int(len(selected)),qualified_groups=0,conditional_bound_assumption=PARAMS['conditional_null'],
        individual_pixel_confirmation=False,mechanical_RPM_measured=False,parameters=PARAMS,
        activity_candidate_roi=candidate_roi,spatial_proposal_is_not_periodic_evidence=True)
    if not len(selected):return mask,diagnostic,[]
    a=a[selected];b=b[selected];block=PARAMS['nuisance_block_bins']
    # Discovery removes the short-block envelope before selecting the comb.
    x=np.arange(block,dtype=np.float32)-(block-1)/2;energy_x=float(x@x)
    original=a.reshape(-1,n//block,block)
    detrended=original-original.mean(2,keepdims=True)
    detrended-=((detrended*x).sum(2,keepdims=True)/energy_x)*x
    detrended=detrended.reshape(-1,n)
    times,dt,spectrum,freq,validation=bank(window_us)
    z=(detrended@spectrum).reshape(-1,len(BASE_FREQUENCIES),4)
    power=np.sum(abs(z)**2,axis=2);first=power.argmax(1)
    separated=abs(np.log(BASE_FREQUENCIES[None]/BASE_FREQUENCIES[first,None]))>.2
    second=np.where(separated,power,-1).argmax(1);choices=np.c_[first,second]
    coeff=z[np.arange(len(z))[:,None],choices]
    predicted=np.einsum('nqh,nqth->nqt',coeff,validation[choices],optimize=True).real
    # Both null mean and variance are conditioned only on nuisance block counts.
    g=predicted.reshape(-1,2,n//block,block);g=g-g.mean(3,keepdims=True)
    g-=((g*x).sum(3,keepdims=True)/energy_x)*x
    b=b.reshape(-1,n//block,block)
    projection=(g*b[:,None]).sum((2,3));variance=(g*g).max(3)*b.sum(2)[:,None]
    variance=variance.sum(2);bound_range=np.max(abs(g),axis=(2,3))
    score=np.where(projection>0,projection**2/(2*np.maximum(variance+bound_range*projection/3,1e-15)),0.)
    choice=score.argmax(1);best=score[np.arange(len(score)),choice]
    threshold=float(np.log(ng*2/PARAMS['family_alpha']))
    good=best>=threshold;groups=[]
    for j in np.flatnonzero(good):
        identifier=int(selected[j]);cell_id=identifier//2;y=(cell_id//cols)*cell;x=(cell_id%cols)*cell
        mask[y:min(y+cell,height),x:min(x+cell,width)]=True
        groups.append(dict(cell_xy=[x,y],polarity=identifier%2,frequency_hz=float(BASE_FREQUENCIES[choices[j,choice[j]]]),
            score=float(best[j]),spatial_pixels=int(spatial[identifier]),discovery_events=int(na[identifier]),validation_events=int(nb[identifier])))
    diagnostic.update(qualified_groups=len(groups),maximum_score=float(best.max()),threshold=threshold,
        frequencies_are_regional_candidates=True)
    return mask,diagnostic,groups
