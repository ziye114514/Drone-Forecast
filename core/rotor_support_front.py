"""Rotor-only support front end: density suppression, then past-window period validation.

No whole-body detector box, no grey-level reconstruction, no labels, no future
events. Two outputs are deliberately kept separate:

  density mask   only a fast candidate filter (event density and local contrast)
  period mask    pixels whose polarity/timing fits a template discovered in the
                 PAST half window and validated on the LATER half window

Only the period mask is called rotor support. An event-count threshold alone is
never called a rotor, and neither mask is a physical rim, a blade identity or a
calibrated RPM. Components are described (size, axis, frequency); a rotor disc
and a 4/3/2 count are NOT claimed here.
"""
import numpy as np
from scipy.ndimage import binary_closing,binary_dilation,label,uniform_filter
from predictive_harmonics import predictive_harmonics

PARAMS=dict(window_us=33333,min_events_per_pixel=6,local_size=31,local_multiplier=2.,
    min_component_pixels=40,close_radius=2,dilate_radius=6,min_periodic_fraction=.05)


def accumulate(events,cutoff_s,width=1280,height=720,window_us=PARAMS['window_us']):
    end=round(cutoff_s*1e6);begin=end-window_us
    selected=events[(events['t']>=begin)&(events['t']<end)]
    flat=selected['y'].astype(np.int64)*width+selected['x'].astype(np.int64)
    on=np.bincount(flat[selected['p']>0],minlength=width*height).reshape(height,width)
    off=np.bincount(flat[selected['p']<=0],minlength=width*height).reshape(height,width)
    return on,off,dict(events=int(len(selected)),window_us=window_us,begin_us=begin,end_us=end)


def density_mask(count,params=PARAMS,fast_spatial=False):
    """Absolute floor plus local contrast; a flat dense field is suppressed."""
    if fast_spatial:
        import cv2
        local=cv2.boxFilter(count.astype(np.float32),-1,(params['local_size'],params['local_size']),borderType=cv2.BORDER_REPLICATE)
    else:local=uniform_filter(count.astype(np.float32),size=params['local_size'],mode='nearest')
    keep=(count>=params['min_events_per_pixel'])&(count>=params['local_multiplier']*np.maximum(local,1e-3))
    return keep,local


def components(mask,params=PARAMS,fast_spatial=False):
    if fast_spatial:
        import cv2
        closed=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),
            iterations=params['close_radius'],borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        count,labels,stats,_=cv2.connectedComponentsWithStats(closed.astype(np.uint8),connectivity=4)
        count-=1
    else:
        closed=binary_closing(mask,structure=np.ones((3,3)),iterations=params['close_radius'])
        labels,count=label(closed)
    found=[]
    for index in range(1,count+1):
        if fast_spatial:
            x,y,w,h,n=stats[index]
            if n<params['min_component_pixels']:continue
            ys,xs=np.nonzero(labels[y:y+h,x:x+w]==index);ys+=y;xs+=x
        else:ys,xs=np.nonzero(labels==index)
        if len(xs)<params['min_component_pixels']:continue
        points=np.column_stack((xs,ys)).astype(float)
        centred=points-points.mean(axis=0)
        covariance=np.cov(centred.T) if len(points)>2 else np.eye(2)
        values,vectors=np.linalg.eigh(covariance)
        axis=vectors[:,-1]
        length=float(np.ptp(centred@axis));width_=float(np.ptp(centred@vectors[:,0]))
        found.append(dict(index=int(index),pixels=int(len(xs)),
            bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
            centroid=[float(points[:,0].mean()),float(points[:,1].mean())],
            axis=[float(axis[0]),float(axis[1])],length_px=length,width_px=width_))
    found.sort(key=lambda item:-item['pixels'])
    return closed,labels,found


class ConsensusFrequencySelector:
    """Causal frequency follower with hysteresis.

    The median-closest rule in select_frequency re-selects the past value while
    that value stays among the candidates, so it latches. This follower keeps the
    current value until the reported value deviates by more than `deviation` in
    the same direction for `k` consecutive frames. It adds lag on purpose and the
    lag must be published (see tools/probe_frequency_step.py).
    """
    def __init__(self,deviation=.10,k=3):
        self.current=None;self.streak=0;self.direction=0
        self.deviation=deviation;self.k=k

    def step(self,reported):
        if reported is None or reported<=0:return self.current
        if self.current is None:
            self.current=float(reported);return self.current
        deviation=np.log(reported/self.current)
        if abs(deviation)<=np.log(1+self.deviation):
            self.streak=0;self.direction=0;return self.current
        sign=1 if deviation>0 else -1
        self.streak=self.streak+1 if sign==self.direction else 1
        self.direction=sign
        if self.streak>=self.k:
            self.current=float(reported);self.streak=0;self.direction=0
        return self.current


def select_frequency(reported,candidates,past,max_factor=1.6):
    """Causal frequency choice: stay close to the running median of past choices.

    Only past assigned values are used. A real change larger than max_factor is
    rejected by construction, so the step response must be published.
    """
    if not reported or reported<=0:return None,False
    if len(past)<2:return float(reported),False
    median=float(np.median(past))
    options=[float(reported)]+[float(value) for value,count in (candidates or [])]
    valid=[value for value in options if value>0 and abs(np.log(value/median))<=np.log(max_factor)]
    if not valid:return float(reported),False
    chosen=min(valid,key=lambda value:abs(np.log(value/median)))
    return float(chosen),bool(abs(np.log(chosen/reported))>1e-9)


def component_candidates(labels,index,lookup,width,top=3,bbox=None):
    """Aggregate the two per-pixel frequency hypotheses inside one label."""
    if not lookup:return []
    if bbox is None:ys,xs=np.nonzero(labels==index)
    else:
        x,y,X,Y=bbox;ys,xs=np.nonzero(labels[y:Y+1,x:X+1]==index);ys+=y;xs+=x
    counts={}
    for y,x in zip(ys,xs):
        pair=lookup.get(int(y)*width+int(x))
        if pair is None:continue
        for value in pair:counts[value]=counts.get(value,0)+1
    return sorted(((value,count) for value,count in counts.items()),key=lambda item:-item[1])[:top]


def rotor_support(events,cutoff_s,width=1280,height=720,params=PARAMS,
        warp=None,base_frequencies=None,period_estimator=None,fast_spatial=False,native_only=False,retain_candidates=True):
    """warp: optional already-warped events from a qualified PAST velocity only."""
    source=events if warp is None else warp
    device_front=hasattr(period_estimator,'prepare_rotor_support')
    if device_front:
        if params != PARAMS or base_frequencies is not None:
            raise ValueError('Device front currently uses the fixed native candidate parameters')
        on,off,meta,dense,local,period=period_estimator.prepare_rotor_support(
            source,cutoff_s,width,height,params['window_us'])
    else:
        on,off,meta=accumulate(source,cutoff_s,width,height,params['window_us'])
        count=on+off
        dense,local=density_mask(count,params,fast_spatial=fast_spatial)
        kept=source[dense[source['y'].astype(np.int64),source['x'].astype(np.int64)]]
        estimator=predictive_harmonics if period_estimator is None else period_estimator
        period=estimator(kept,cutoff_s,window_us=params['window_us'],width=width,
            height=height,base_frequencies=base_frequencies)
    if native_only:
        # RotorOnlyFront reconstructs native components independently. Generic
        # strict-pixel labeling/dilation below is not an input to that pipeline.
        # Keep it for RegionTracker/diagnostic users, omit only unused work here.
        lookup={int(p):(float(f[0]),float(f[1])) for p,f in zip(period['pixels'],period['candidate_frequency_hz'])} if retain_candidates else {}
        return dict(on=on,off=off,density=dense,density_local=local,candidate_lookup=lookup,
            native_counts20=meta.get('native_counts20'),native_precomputed=meta.get('native_precomputed'),
            regional_graph_fallback=meta.get('regional_graph_fallback'),
            regional_only_motor_frequency_unobserved=meta.get('regional_only_motor_frequency_unobserved',False),
            current_spatial_track_qualified=meta.get('current_spatial_track_qualified',False),
            current_spatial_track=meta.get('current_spatial_track'),
            period_pixels_unique=period['pixels'],period_candidate_frequencies=period['candidate_frequency_hz'],
            first_harmonic_statistics=period.get('first_harmonic_statistics'),
            backend_timing={k:meta[k] for k in ('cuda_service_ms','cuda_roundtrip_ms','sparse_transfer_bytes') if k in meta},
            period_mask=period['mask'],frequency_hz=period['frequency_hz'],period_evidence=period['evidence'],
            components=[],components_all=[],component_count=None,components_not_computed=True,
            support=None,bbox=None,
            counts=dict(events=meta['events'],density_pixels=meta.get('density_pixels',int(dense.sum())),
                period_pixels=int(period['accepted_pixels']),eligible_pixels=int(period['eligible_pixels']),components=None),
            parameters=dict(params,window_us=params['window_us']),body_box_used=False,
            grey_reconstruction_used=False,labels_used=False,future_events_used=False,
            physical_rim_measured=False,mechanical_RPM_measured=False,rotor_count_claimed=False)
    closed,labels,found=components(period['mask']>0,params,fast_spatial=fast_spatial)
    # Two separated frequency choices per eligible pixel, kept for temporal
    # assignment. They are hypotheses from the past half-window, not measurements.
    candidates={}
    candidate_lookup={}
    for pixel,f1,f2 in zip(period['pixels'],period['candidate_frequency_hz'][:,0],
            period['candidate_frequency_hz'][:,1]):
        candidates[int(pixel)]=(float(f1),float(f2))
    candidate_lookup=candidates
    for component in found:
        if fast_spatial:
            x,y,X,Y=component['bbox']
            inside=labels[y:Y+1,x:X+1]==component['index']
            frequency=period['frequency_hz'][y:Y+1,x:X+1][inside]
        else:
            inside=labels==component['index']
            frequency=period['frequency_hz'][inside]
        frequency=frequency[frequency>0]
        component['median_frequency_hz']=float(np.median(frequency)) if len(frequency) else None
        component['periodic_fraction']=float(len(frequency)/max(component['pixels'],1))
        component['candidate_frequencies']=component_candidates(labels,component['index'],
            candidates,width,bbox=component['bbox'] if fast_spatial else None)
    accepted=[c for c in found if c['periodic_fraction']>=params['min_periodic_fraction']]
    if accepted:
        xs=[c['bbox'][0] for c in accepted]+[c['bbox'][2] for c in accepted]
        ys=[c['bbox'][1] for c in accepted]+[c['bbox'][3] for c in accepted]
        pad=params['dilate_radius']
        bbox=[max(0,min(xs)-pad),max(0,min(ys)-pad),
              min(width-1,max(xs)+pad),min(height-1,max(ys)+pad)]
    else:
        bbox=None
    if bbox and fast_spatial:
        import cv2
        support=cv2.dilate(closed.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=params['dilate_radius'],
            borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    else:support=binary_dilation(closed,iterations=params['dilate_radius']) if bbox else np.zeros_like(closed)
    return dict(on=on,off=off,density=dense,density_local=local,candidate_lookup=candidate_lookup,
        period_pixels_unique=period['pixels'],period_candidate_frequencies=period['candidate_frequency_hz'],
        first_harmonic_statistics=period.get('first_harmonic_statistics'),
        native_counts20=meta.get('native_counts20'),
        native_precomputed=meta.get('native_precomputed'),
        backend_timing={k:meta[k] for k in ('cuda_service_ms','cuda_roundtrip_ms','sparse_transfer_bytes') if k in meta},
        period_mask=period['mask'],
        frequency_hz=period['frequency_hz'],period_evidence=period['evidence'],
        components=accepted,components_all=found,component_count=len(accepted),
        support=support,bbox=bbox,
        counts=dict(events=meta['events'],density_pixels=int(dense.sum()),
            period_pixels=int(period['accepted_pixels']),eligible_pixels=int(period['eligible_pixels']),
            components=len(accepted)),
        parameters=dict(params,window_us=params['window_us'],
            base_frequencies='template_discovered' if base_frequencies is None else list(base_frequencies)),
        body_box_used=False,grey_reconstruction_used=False,labels_used=False,
        future_events_used=False,physical_rim_measured=False,mechanical_RPM_measured=False,
        rotor_count_claimed=False)


def accumulate_support(masks,times,tau_s=.15,min_level=.35,params=PARAMS):
    """Causal decayed accumulation; a single hit survives about ln(1/min_level)*tau.

    Without a qualified past velocity the accumulator lives in fixed pixels and a
    moving rotor smears. That cost is measured and reported, not hidden.
    """
    accumulator=None;previous_t=None
    for mask,t in zip(masks,times):
        if accumulator is None:accumulator=mask.astype(np.float32)
        else:
            dt=max(float(t-previous_t),1/240.)
            accumulator=accumulator*np.exp(-dt/tau_s)
            accumulator=np.maximum(accumulator,mask.astype(np.float32))
        previous_t=t
    stable=accumulator>=min_level
    closed,labels,found=components(stable,params)
    return dict(stable=stable,level=accumulator,closed=closed,labels=labels,components=found,
        tau_s=tau_s,min_level=min_level,single_hit_persistence_s=float(np.log(1/min_level)*tau_s),
        motion_compensation=False)


def shift_mask(mask,dx,dy):
    """Integer whole-pixel shift of a binary mask; no interpolation."""
    return np.roll(np.roll(mask,int(round(dy)),axis=0),int(round(dx)),axis=1)


def motion_image(on,off,factor=1):
    """Log ON/OFF energy, optionally block-summed; blade texture is suppressed by blur."""
    from scipy.ndimage import gaussian_filter
    image=np.log1p(on.astype(np.float32)+off.astype(np.float32))
    if factor>1:
        height,width=image.shape
        height=height//factor*factor;width=width//factor*factor
        image=image[:height,:width].reshape(height//factor,factor,width//factor,factor).sum(axis=(1,3))
    return gaussian_filter(image,2.0)


def estimate_region_translation(previous,previous_support,current,max_shift=40,pad=24,width=1280,height=720,current_support=None,integral_energy=True,native_matching=False,subpixel=False,predicted_shift=(0.,0.),coarse_matching=False,interior_border=False,centered_matching=False,peak_shape=False,joint_peak=False,peak_covariance=True,joint_peak_neighborhood=False,polarity_separated=False,capture_surface=False,local_contrast=False):
    """Matched-filter shift of the ROTOR REGION only, past frames only.

    Whole-frame phase correlation is dominated by the static background and
    returns zero. This crops to the previous support window, then searches the
    current frame inside +-max_shift; the peak offset is the forward displacement.
    """
    if not native_matching:
        from scipy.ndimage import gaussian_filter
        from scipy.signal import fftconvolve
    if native_matching:
        import cv2
        bx,by,bw,bh=cv2.boundingRect(previous_support.view(np.uint8))
        if not bw:return None
        x0=max(0,bx-pad);x1=min(width,bx+bw+pad)
        y0=max(0,by-pad);y1=min(height,by+bh+pad)
    else:
        ys,xs=np.nonzero(previous_support)
        if not len(xs):return None
        x0=max(0,int(xs.min())-pad);x1=min(width,int(xs.max())+pad+1)
        y0=max(0,int(ys.min())-pad);y1=min(height,int(ys.max())+pad+1)
    if x1-x0<24 or y1-y0<24:return None
    original_bounds=(x0,y0,x1,y1)
    if interior_border:
        # A template beginning at sensor y=0 cannot be matched at dy<0 by a
        # valid-only correlator. Use supported interior pixels, never pad in
        # fabricated rotor observations outside the sensor.
        x0+=min(max(0,max_shift-x0),max(0,(x1-x0-24)//2))
        x1-=min(max(0,max_shift-(width-x1)),max(0,(x1-x0-24)//2))
        y0+=min(max(0,max_shift-y0),max(0,(y1-y0-24)//2))
        y1-=min(max(0,max_shift-(height-y1)),max(0,(y1-y0-24)//2))
        if np.count_nonzero(previous_support[y0:y1,x0:x1])<12:x0,y0,x1,y1=original_bounds
    m=max_shift
    px,py=np.rint(predicted_shift).astype(int)
    X0=max(0,x0+px-m);X1=min(width,x1+px+m);Y0=max(0,y0+py-m);Y1=min(height,y1+py+m)
    if X1-X0<x1-x0 or Y1-Y0<y1-y0:return None
    ox,oy,oX,oY=original_bounds
    if polarity_separated:
        if not native_matching:raise ValueError('Separate polarity matching requires native matcher')
        patch=np.stack([p[oy:oY,ox:oX] for p in previous],axis=-1).astype(np.float32)
        target=np.stack([p[Y0:Y1,X0:X1] for p in current],axis=-1).astype(np.float32)
    else:
        patch=previous[0][oy:oY,ox:oX].astype(np.float32)+previous[1][oy:oY,ox:oX].astype(np.float32)
        target=current[0][Y0:Y1,X0:X1].astype(np.float32)+current[1][Y0:Y1,X0:X1].astype(np.float32)
    if current_support is not None:
        # Exactly the same filtered inputs, materialized only in the search ROI.
        pm=previous_support[oy:oY,ox:oX];tm=current_support[Y0:Y1,X0:X1]
        if polarity_separated:pm=pm[:,:,None];tm=tm[:,:,None]
        patch=np.where(pm,patch,0.)
        target=np.where(tm,target,0.)
    patch=np.log1p(patch);target=np.log1p(target)
    if native_matching:
        # Same sigma1.5, radius6/truncate4 and half-sample reflection as the
        # existing SciPy filter; separable float32 arithmetic can round slightly
        # differently. Spatial observations/events are not decimated.
        patch=cv2.GaussianBlur(patch,(13,13),1.5,borderType=cv2.BORDER_REFLECT)
        target=cv2.GaussianBlur(target,(13,13),1.5,borderType=cv2.BORDER_REFLECT)
    else:patch=gaussian_filter(patch,1.5);target=gaussian_filter(target,1.5)
    if local_contrast:
        from local_contrast_motion import normalize
        patch=normalize(patch);target=normalize(target)
    # Smooth before selecting an interior template. Reflecting at an artificial
    # crop boundary would invent an edge and bias the known negative shift.
    patch=patch[y0-oy:y1-oy,x0-ox:x1-ox]
    patch=patch-patch.mean(axis=(0,1),keepdims=True) if polarity_separated else patch-patch.mean()
    if not np.isfinite(patch).all() or np.abs(patch).sum()<=0:return None
    score_origin=(0,0);search_diag=dict(coarse=False)
    native_shape=(target.shape[0]-patch.shape[0]+1,target.shape[1]-patch.shape[1]+1)
    if native_matching:
        import cv2
        if coarse_matching:
            from native_correlation import normalized_search
            score,score_origin,search_diag=normalized_search(target,patch,centered=centered_matching)
        else:
            # Normalize both means/variances consistently. The legacy route
            # changes its statistic when an interior crop becomes necessary
            # near the FOV; otherwise its target denominator includes DC
            # event energy despite a zero-mean template numerator.
            method=cv2.TM_CCOEFF_NORMED if centered_matching or original_bounds!=(x0,y0,x1,y1) else cv2.TM_CCORR_NORMED
            score=cv2.matchTemplate(target,patch,method)*float(np.linalg.norm(patch))
        if float(score.max())<=0:return None
    else:
        numerator=fftconvolve(target,patch[::-1,::-1],mode='valid')
    if not native_matching and integral_energy:
        # The denominator is a rectangular box sum, not a general convolution.
        # An integral image removes a second FFT without changing the model.
        import cv2
        h,w=patch.shape;s=cv2.integral(target*target,sdepth=cv2.CV_64F)
        summed=s[h:,w:]-s[:-h,w:]-s[h:,:-w]+s[:-h,:-w]
        energy=np.sqrt(np.maximum(summed,1e-9))
    elif not native_matching:
        energy=np.sqrt(np.maximum(fftconvolve(target*target,np.ones_like(patch),mode='valid'),1e-9))
    if not native_matching:score=numerator/energy
    index=np.unravel_index(int(np.argmax(score)),score.shape)
    dy=int(index[0])+score_origin[0]+Y0-y0;dx=int(index[1])+score_origin[1]+X0-x0
    refinement=np.zeros(2)
    if subpixel:
        # Local quadratic interpolation of the CURRENT matching peak. This
        # removes the previous integer-per-frame quantization; no future
        # image, extra trajectory branch or scene-specific fit is introduced.
        row,col=index
        for axis,i in [(0,col),(1,row)]:
            line=score[row,:] if axis==0 else score[:,col]
            if 0<i<len(line)-1:
                left,middle,right=map(float,line[i-1:i+2]);curvature=left-2*middle+right
                if curvature<0:
                    refinement[axis]=np.clip(.5*(left-right)/curvature,-.5,.5)
        dx+=float(refinement[0]);dy+=float(refinement[1])
    result=dict(dx=dx,dy=dy,score=float(score[index]),method='native_normalized_rotor_match' if native_matching else 'region_matched_filter_log_energy',
        subpixel_refinement=subpixel,peak_offset_px=refinement.tolist(),
        search_center_px=[int(px),int(py)],
        multiscale_search=search_diag,
        normalization_profile='centered_ncc' if centered_matching else 'legacy_fov_dependent',
        supported_interior_template_used=bool(original_bounds!=(x0,y0,x1,y1)),
        feasible_displacement_xy=[[int(X0-x0),int(X1-x1)],[int(Y0-y0),int(Y1-y1)]],
        peak_at_search_boundary=bool(index[0]+score_origin[0] in (0,native_shape[0]-1) or index[1]+score_origin[1] in (0,native_shape[1]-1)),
        search_px=max_shift,region=[x0,y0,x1,y1],peak=float(score[index]))
    if peak_shape:
        from native_peak_shape import assess
        info=assess(score,index,max_shift,full_neighborhood=joint_peak_neighborhood);result['native_peak_shape']=info
        if peak_covariance:result['displacement_covariance_px2']=info['covariance_px2']
        if joint_peak and subpixel and info.get('joint_refinement_xy') is not None:
            value=np.asarray(info['joint_refinement_xy']);difference=value-refinement
            result['dx']+=float(difference[0]);result['dy']+=float(difference[1]);result['peak_offset_px']=value.tolist()
            result['joint_two_dimensional_subpixel_refinement']=True
    if local_contrast:result['native_local_contrast_normalization']=dict(window_px=31,geometry_and_frequency_counts_unchanged=True)
    if polarity_separated:result['matching_channels']='separate_ON_OFF_joint_normalized_correlation'
    if capture_surface:
        if not native_matching:raise ValueError('Joint native surface requires native matcher')
        result['_matching_surface']=dict(score=score/float(np.linalg.norm(patch)),
            minimum_xy=[int(X0-x0+score_origin[1]),int(Y0-y0+score_origin[0])])
    return result


def region_gap(a,b):
    """Edge-to-edge gap between two bboxes, zero when they overlap."""
    dx=max(0.,max(a[0],b[0])-min(a[2],b[2]))
    dy=max(0.,max(a[1],b[1])-min(a[3],b[3]))
    return float(np.hypot(dx,dy))


def merge_region_clusters(components,gap_px=24.):
    """Group support regions whose boxes are closer than gap_px.

    A blade sweep covers roughly the disc's minor axis, so fragments of one disc
    are separated by less than that while two rotors are far apart. gap_px is a
    declared choice, not a calibrated value; the caller must report the curve.
    Merged groups are 'possible disc regions', still not blade identities.
    """
    count=len(components)
    parent=list(range(count))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[b]=a
    for i in range(count):
        for j in range(i+1,count):
            if region_gap(components[i]['bbox'],components[j]['bbox'])<=gap_px:union(i,j)
    groups={}
    for index in range(count):groups.setdefault(find(index),[]).append(index)
    merged=[]
    for members in groups.values():
        boxes=[components[i]['bbox'] for i in members]
        frequencies=[components[i].get('median_frequency_hz') for i in members]
        frequencies=[f for f in frequencies if f is not None]
        merged.append(dict(members=[components[i]['id'] for i in members],size=len(members),
            bbox=[min(b[0] for b in boxes),min(b[1] for b in boxes),
                  max(b[2] for b in boxes),max(b[3] for b in boxes)],
            pixels=int(sum(components[i]['pixels'] for i in members)),
            median_frequency_hz=None if not frequencies else float(np.median(frequencies)),
            frequencies=frequencies))
    merged.sort(key=lambda item:-item['pixels'])
    return dict(gap_px=float(gap_px),regions_before=count,groups_after=len(merged),clusters=merged)


def circle_fit(points):
    """Kasa algebraic circle fit with residual and angular coverage.

    A short arc barely constrains the radius, so coverage is reported next to the
    residual: a low residual over a narrow arc is not evidence of one disc.
    """
    points=np.asarray(points,float)
    if len(points)<8:return None
    x=points[:,0];y=points[:,1]
    design=np.c_[2*x,2*y,np.ones(len(x))]
    values=x*x+y*y
    solution,*_=np.linalg.lstsq(design,values,rcond=None)
    cx,cy=solution[0],solution[1]
    radius=float(np.sqrt(max(solution[2]+cx*cx+cy*cy,1e-9)))
    distances=np.hypot(x-cx,y-cy)
    rms=float(np.sqrt(np.mean((distances-radius)**2)))
    angles=np.sort(np.arctan2(y-cy,x-cx))
    if len(angles)>1:
        gaps=np.diff(np.r_[angles,angles[0]+2*np.pi])
        coverage=float((2*np.pi-gaps.max())*180/np.pi)
    else:
        coverage=0.
    return dict(cx=float(cx),cy=float(cy),r=radius,rms=rms,coverage_deg=coverage,
        rms_over_r=rms/max(radius,1e-9),points=int(len(points)))


def fit_disc_candidates(points_by_fragment,max_rms_ratio=.12,min_coverage_deg=120.,
        radius_range=(12.,140.),min_points=30):
    """Agglomerate fragments while one circle still explains them.

    Only merges whose combined fit keeps rms/r low AND angular coverage high are
    accepted; coverage guards against the short-arc degeneracy where any two
    small blobs lie on some circle. Returns accepted groups with per-merge evidence.
    """
    usable=[(key,points) for key,points in points_by_fragment.items() if len(points)>=min_points]
    groups=[dict(members=[key],points=points,fit=circle_fit(points)) for key,points in usable]
    groups=[group for group in groups if group['fit'] is not None]
    accepted=[];rejected=[]
    def ok(fit):
        if fit is None:return False
        return (fit['rms_over_r']<=max_rms_ratio and fit['coverage_deg']>=min_coverage_deg
            and radius_range[0]<=fit['r']<=radius_range[1])
    while True:
        pairs=[]
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                points=np.vstack((groups[i]['points'],groups[j]['points']))
                fit=circle_fit(points)
                if fit is None:continue
                pairs.append((fit['rms_over_r'],i,j,fit,points))
        if not pairs:break
        pairs.sort(key=lambda item:item[0])
        chosen=None
        for score,i,j,fit,points in pairs:
            if ok(fit):chosen=(score,i,j,fit,points);break
        if chosen is None:
            best=pairs[0]
            rejected.append(dict(reason='no_acceptable_pair',best_rms_over_r=best[3]['rms_over_r'],
                best_coverage_deg=best[3]['coverage_deg'],best_radius=best[3]['r']))
            break
        score,i,j,fit,points=chosen
        merged=dict(members=groups[i]['members']+groups[j]['members'],points=points,fit=fit)
        accepted.append(dict(members=merged['members'],fit=fit))
        groups=[group for index,group in enumerate(groups) if index not in (i,j)]+[merged]
        groups.sort(key=lambda group:group['members'][0])
    candidates=[dict(members=group['members'],fit=group['fit'],
        accepted=ok(group['fit'])) for group in groups]
    candidates.sort(key=lambda item:-item['fit']['points'])
    return dict(candidates=candidates,merges=accepted,rejected=rejected,
        fragments=len(usable),disc_candidates=sum(1 for c in candidates if c['accepted']),
        max_rms_ratio=max_rms_ratio,min_coverage_deg=min_coverage_deg)


class RegionTracker:
    """Greedy association of support regions across cutoffs, with age and misses.

    Previous centres are moved by the tracker's own past shift before matching, so
    a moving drone does not break every association. Tracks are described only by
    geometry, age and a short frequency history; no rotor identity, count or RPM
    is claimed.
    """
    def __init__(self,gate_scale=.75,min_gate_px=12.,max_miss=3,ratio=(.4,2.5),history=5):
        self.tracks={};self.next_id=1
        self.gate_scale=gate_scale;self.min_gate_px=min_gate_px
        self.max_miss=max_miss;self.ratio=ratio;self.history=history

    @staticmethod
    def _radius(component):
        return float(np.sqrt(max(component.get('pixels',1),1)/np.pi))

    def update(self,components,t,shift_px=(0.,0.),frequency_key='median_frequency_hz'):
        alive=[(track_id,track) for track_id,track in self.tracks.items()
            if track['misses']<=self.max_miss]
        pairs=[]
        for component in components:
            cx,cy=component['centroid']
            for track_id,track in alive:
                px,py=track['centroid']
                predicted=(px+shift_px[0],py+shift_px[1])
                distance=float(np.hypot(cx-predicted[0],cy-predicted[1]))
                gate=max(self.min_gate_px,self.gate_scale*(self._radius(component)+
                    np.sqrt(max(track['pixels'],1)/np.pi)))
                ratio=self._radius(component)/max(np.sqrt(max(track['pixels'],1)/np.pi),1e-6)
                if distance<=gate and self.ratio[0]<=ratio<=self.ratio[1]:
                    pairs.append((distance,component['index'],track_id,component))
        pairs.sort(key=lambda item:item[0])
        used_components=set();used_tracks=set();assigned=[]
        for distance,index,track_id,component in pairs:
            if index in used_components or track_id in used_tracks:continue
            used_components.add(index);used_tracks.add(track_id)
            track=self.tracks[track_id]
            track.update(misses=0,hits=track['hits']+1,last_t=float(t),
                centroid=[float(v) for v in component['centroid']],bbox=component['bbox'],
                pixels=int(component['pixels']),axis=component['axis'],
                length_px=component['length_px'],width_px=component['width_px'],
                distance_to_prediction=distance)
            frequency=component.get(frequency_key)
            track['frequency_history']=(track['frequency_history']+[frequency])[-self.history:]
            track['median_frequency_hz']=frequency
            track['candidate_frequencies']=component.get('candidate_frequencies')
            assigned.append(component['index'])
        for track_id,track in alive:
            if track_id not in used_tracks:track['misses']+=1
        for component in components:
            if component['index'] in used_components:continue
            self.tracks[self.next_id]=dict(id=self.next_id,first_t=float(t),last_t=float(t),
                hits=1,misses=0,centroid=[float(v) for v in component['centroid']],
                bbox=component['bbox'],pixels=int(component['pixels']),axis=component['axis'],
                length_px=component['length_px'],width_px=component['width_px'],
                distance_to_prediction=None,median_frequency_hz=component.get(frequency_key),
                frequency_history=[component.get(frequency_key)])
            used_components.add(component['index']);self.next_id+=1
        for track_id in [k for k,v in self.tracks.items() if v['misses']>self.max_miss]:
            self.tracks.pop(track_id,None)
        summary=[]
        for track in self.tracks.values():
            values=[v for v in track['frequency_history'] if v]
            summary.append(dict(id=track['id'],hits=track['hits'],misses=track['misses'],
                age_s=track['last_t']-track['first_t'],pixels=track['pixels'],bbox=track['bbox'],
                median_frequency_hz=track['median_frequency_hz'],
                candidate_frequencies=track.get('candidate_frequencies'),
                frequency_std=None if len(values)<2 else float(np.std(values)),
                qualified=bool(track['hits']>=3 and track['misses']<=1)))
        summary.sort(key=lambda item:(-item['hits'],item['id']))
        return dict(t=float(t),tracks=summary,confirmed=sum(1 for item in summary if item['qualified']),
            total=len(summary),associations=len(assigned),
            unmatched_components=len(components)-len(assigned))


class RotorSupportTracker:
    """Streaming wrapper: per-cutoff support, accumulation, and optional self motion.

    shift_px  explicit displacement of the accumulator (past-only)
    velocity  explicit past velocity in px/s; converted with the real dt
    self_motion  estimate the displacement from the tracker's own ON/OFF images
    """
    def __init__(self,tau_s=.15,min_level=.35,params=PARAMS,self_motion=True):
        self.accumulator=None;self.last_t=None;self.tau_s=tau_s;self.min_level=min_level
        self.params=params;self.self_motion=self_motion;self.previous_image=None
        self.previous_support=None

    def step(self,events,t,shift_px=None,velocity=None):
        return self.observe(rotor_support(events,t,params=self.params),t,shift_px,velocity)

    def observe(self,frame,t,shift_px=None,velocity=None):
        mask=(frame['period_mask']>0).astype(np.float32)
        source='none';shift=None;estimate=None
        if self.accumulator is None:
            self.accumulator=mask
            source='first_cutoff_no_history'
        else:
            if shift_px is None and velocity is not None:
                dt=max(t-self.last_t,1/240.)
                shift_px=(float(velocity[0])*dt,float(velocity[1])*dt);source='given_velocity'
            if shift_px is None and self.self_motion and self.previous_image is not None:
                estimate=estimate_region_translation(self.previous_image,self.previous_support,
                    (frame['on'],frame['off']))
                if estimate is None:
                    shift_px=None;source='self_region_estimate_unavailable'
                else:
                    shift_px=(estimate['dx'],estimate['dy']);source='self_region_matched_filter'
            dt=max(t-self.last_t,1/240.)
            if shift_px is None:
                self.accumulator*=np.exp(-dt/self.tau_s);source='no_motion_model'
            else:
                shift=[float(shift_px[0]),float(shift_px[1])]
                self.accumulator=shift_mask(self.accumulator,shift[0],shift[1])
                self.accumulator*=np.exp(-dt/self.tau_s)
            self.accumulator=np.maximum(self.accumulator,mask)
        self.previous_image=(frame['on'],frame['off'])
        self.previous_support=(frame['period_mask']>0)
        self.last_t=t
        stable=self.accumulator>=self.min_level
        closed,labels,found=components(stable,self.params)
        for component in found:
            inside=labels==component['index']
            values=frame['frequency_hz'][inside]
            values=values[values>0]
            component['median_frequency_hz']=float(np.median(values)) if len(values) else None
            component['periodic_fraction']=float(len(values)/max(component['pixels'],1))
            component['candidate_frequencies']=component_candidates(labels,component['index'],
                frame.get('candidate_lookup'),self.params.get('width',1280))
        frame['stable_support']=closed;frame['accumulation_level']=self.accumulator
        frame['accumulated_components']=found
        frame['accumulation']=dict(source=source,shift_px=shift,estimate=estimate,
            tau_s=self.tau_s,min_level=self.min_level,self_motion=self.self_motion)
        return frame
