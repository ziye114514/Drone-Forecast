"""Independent implementation of repetitive-event rotor clustering and ellipse cues.

No imported detection weights, labels, video frames, or whole-clip statistics.
Ellipse normals below are weak-perspective image cues, not calibrated 3-D attitude.
"""
from collections import deque
from itertools import combinations
import cv2
import numpy as np
from scipy.signal import periodogram, find_peaks
from scipy.optimize import linear_sum_assignment
from .geometry import fit_rotor_geometry

class RotorPerception:
    def __init__(self,width,height,cfg):
        self.w,self.h,self.cfg=width,height,cfg
        self.history=deque()
        self.center=None
        self.velocity=np.zeros(2)
        self.last_t=None
        self.last_seen=None
        self.last_extent=120.
        self.normal=None
        self.freqs=[]
        self.frequency_quality=0.
        self.frequency_reference=None
        self.frame_id=0
        self.constellation=None
        self.constellation_t=None
        self.pending_center=None
        self.pending_streak=0
        self.previous_rotors=[]
        self.previous_rotors_t=None
        self.layout_measured_t=None
        self.rotor_frequency_quality=[]
        self.frequency_diagnostics=[]
        cv2.setNumThreads(cfg.opencv_threads)

    def detect(self,events,t):
        c=self.cfg; w,h=self.w,self.h; b=c.spatial_bin
        if self.last_seen is not None and t-self.last_seen>c.reset_gap_s:
            self.center=None;self.velocity=np.zeros(2);self.last_t=None;self.last_seen=None
            self.normal=None;self.constellation=None;self.constellation_t=None;self.pending_streak=0
            self.previous_rotors=[];self.previous_rotors_t=None;self.layout_measured_t=None
        pixel=events['y'].astype(np.int32)*w+events['x'] if len(events) else np.empty(0,int)
        total=np.bincount(pixel,minlength=w*h).reshape(h,w)
        positive=np.bincount(pixel[events['p']>0],minlength=w*h).reshape(h,w)
        negative=total-positive
        # Repeated bipolar activity suppresses one-pass body/background edges.
        repeated=np.minimum(positive,negative).astype(np.float32)
        repeated[total<c.min_pixel_count]=0
        hs,ws=h//b,w//b
        small=repeated[:hs*b,:ws*b].reshape(hs,b,ws,b).sum(axis=(1,3))
        blurred=cv2.GaussianBlur(small,(3,3),.8)
        positive_bins=blurred[blurred>0]
        threshold=max(2.,float(np.quantile(positive_bins,c.density_quantile))) if len(positive_bins) else 2.
        raw_mask=np.uint8(blurred>=threshold)*255
        mask=raw_mask.copy()
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        candidates=self._candidates(contours,small,t,b)
        if c.enable_line_geometry:
            # Opening erases narrow rotors. A separate pre-opening pass admits
            # only elongated repetitive components, never arbitrary body edges.
            thin,_=cv2.findContours(raw_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            extras=self._candidates(thin,small,t,b,thin_only=True)
            for candidate in extras:
                if not any(np.linalg.norm(candidate['center']-q['center'])<max(6.,q['major']*.2) for q in candidates):
                    candidates.append(candidate)
        candidates.sort(key=lambda x:x['activity']*x['shape_quality'],reverse=True)
        candidates=candidates[:c.max_candidates]
        selected=self._group(candidates,t)
        # Bounded causal frequency buffer: uniform event thinning, never future frames.
        stride=max(1,int(np.ceil(len(events)/max(1,c.max_frequency_events//6))))
        self.history.append((t,events[::stride].copy()))
        while self.history and self.history[0][0]<t-c.frequency_window_us/1e6-.03:
            self.history.popleft()
        self.frame_id+=1
        if selected is None:
            self.pending_streak=0
            return dict(t=t,detected=False,confidence=0.,rotors=[],n_events=len(events)),total
        rotors=selected
        if self.constellation is None and len(rotors)<3:
            pair_center=np.mean([r['center'] for r in rotors],axis=0)
            self.pending_streak=(self.pending_streak+1) if self.pending_center is not None and np.linalg.norm(pair_center-self.pending_center)<80 else 1
            self.pending_center=pair_center
            if self.pending_streak<3:
                return dict(t=t,detected=False,confidence=0.,rotors=[],n_events=len(events)),total
        geometry=rotors
        if self.constellation is not None and t-self.constellation_t<.3 and len(rotors)<4:
            previous=np.array([r['center'] for r in self.constellation])
            current=np.array([r['center'] for r in rotors])
            predicted=previous+self.velocity*(t-self.constellation_t)
            ids,cols=linear_sum_assignment(np.linalg.norm(current[:,None]-predicted[None,:],axis=2))
            source=previous[cols].astype(np.float32);target=current[ids].astype(np.float32)
            if len(source)==1:
                transform=np.column_stack((np.eye(2),target[0]-source[0]))
            else:
                transform,_=cv2.estimateAffinePartial2D(source,target,method=cv2.LMEDS)
            if transform is not None:
                scale=np.linalg.norm(transform[:,0])
                error=np.linalg.norm(source@transform[:,:2].T+transform[:,2]-target,axis=1).mean()
                if .8<scale<1.25 and error<max(12,self.last_extent*.08):
                    geometry=[]
                    for old in self.constellation:
                        center=transform[:,:2]@old['center']+transform[:,2]
                        # Conservative transformed axis-aligned disk bounding box.
                        half=(old['bbox'][2:]-old['bbox'][:2])*scale/2
                        geometry.append(dict(old,center=center,bbox=np.r_[center-half,center+half],major=old['major']*scale))
        if len(rotors)>=2:self.layout_measured_t=t
        if len(geometry)==4:
            self.constellation=[dict(r,center=r['center'].copy(),bbox=r['bbox'].copy()) for r in geometry]
            self.constellation_t=t
        boxes=np.array([r['bbox'] for r in geometry]); lo=boxes[:,:2].min(axis=0); hi=boxes[:,2:].max(axis=0)
        span=hi-lo
        # Target box includes the body/legs below the rotor plane; generic image prior.
        lo-=np.array([.015,.015])*span[0]
        hi+=np.array([.015,.12])*span[0]
        lo=np.maximum(lo,0); hi=np.minimum(hi,[w-1,h-1])
        center=(lo+hi)/2
        if self.center is not None and self.last_t is not None:
            dt=max(t-self.last_t,.001)
            self.velocity=.7*self.velocity+.3*np.clip((center-self.center)/dt,-2500,2500)
        self.center=center; self.last_t=t; self.last_seen=t;self.last_extent=float(max(span))
        self.previous_rotors=[dict(r) for r in rotors];self.previous_rotors_t=t
        best=max(rotors,key=lambda r:r['geometry_quality']*np.log1p(r['activity']))
        n=best['normal'].copy()
        if self.normal is not None and n@self.normal<0:n=-n
        elif self.normal is None and n[0]<0:n=-n
        self.normal=n
        # Axis consistency is sign invariant; conflicting rotor fits cannot all
        # confidently describe a common plane.
        directors=np.array([[r['normal'][0]**2-r['normal'][1]**2,
                              2*r['normal'][0]*r['normal'][1]] for r in rotors])
        lengths=np.linalg.norm(directors,axis=1)
        usable=lengths>.05
        agreement=float(np.linalg.norm(np.mean(directors[usable]/lengths[usable,None],axis=0))) if usable.any() else 0.
        geometry_quality=float(np.mean([r['geometry_quality'] for r in rotors]))*agreement
        if self.frame_id%c.frequency_update_steps==0 or len(self.freqs)!=len(rotors):
            self.freqs,self.frequency_quality=self._frequency(rotors,t)
        valid=[f for f in self.freqs if np.isfinite(f)]
        freq=float(np.median(valid)) if valid else None
        if freq and self.frequency_reference is None:self.frequency_reference=freq
        s=(freq/self.frequency_reference)**2 if freq and self.frequency_reference else 1.
        confidence=min(1.,len(rotors)/4)*float(np.mean([r['shape_quality'] for r in rotors]))
        # Tracking and geometric observability are separate qualities. A line
        # can locate activity well while not defining a full ellipse.
        confidence=float(np.clip(confidence,.25,.98))
        center_sigma=float(np.sqrt(np.mean([r['center_sigma_px']**2 for r in rotors])))
        serialized=[dict(cx=float(r['center'][0]),cy=float(r['center'][1]),major=float(r['major']),
                    minor=float(r['minor']),angle=r['angle'],ratio=float(r['ratio']),
                    **{key:r[key] for key in ('mode','visible_arc_fraction','support_fraction','fit_residual_px',
                         'center_sigma_px','axis_angle_sigma_deg','geometry_quality','inferred_shape','axis_only')}) for r in rotors]
        result=dict(t=t,detected=True,confidence=confidence,center=center.tolist(),
                    bbox=np.r_[lo,hi].tolist(),rotors=serialized,normal_xy=n.tolist(),
                    tilt_proxy_deg=None if best['axis_only'] else float(np.rad2deg(np.arccos(np.clip(best['ratio'],0,1)))),
                    geometry_quality=geometry_quality,axis_agreement=agreement,
                    normal_candidates_xy=[n.tolist(),(-n).tolist()],normal_sign_ambiguous=True,
                    center_sigma_px=center_sigma,center_covariance_px2=(np.eye(2)*center_sigma**2).tolist(),
                    layout_inferred=len(geometry)>len(rotors),observed_rotor_count=len(rotors),
                    observation_modes={mode:sum(r['mode']==mode for r in rotors) for mode in ('ellipse','arc','line')},
                    blade_pass_hz=freq,rotor_frequencies_hz=[float(f) if np.isfinite(f) else None for f in self.freqs],
                    frequency_quality=self.frequency_quality,frequency_squared_relative=float(np.clip(s,.1,10)),
                    frequency_diagnostics=self.frequency_diagnostics,
                    n_events=len(events))
        return result,total

    def _candidates(self,contours,activity_map,t,b,thin_only=False):
        candidates=[];c=self.cfg
        # Bound robust hypothesis work in heavily textured scenes.
        contours=sorted(contours,key=cv2.contourArea,reverse=True)[:c.max_candidates*3]
        for contour in contours:
            if len(contour)<8:continue
            area=cv2.contourArea(contour)
            if area>c.max_blob_fraction*activity_map.size:continue
            x,y,bw,bh=cv2.boundingRect(contour)
            if max(bw,bh)*b<10:continue
            if thin_only:
                (_, _),(d1,d2),_=cv2.minAreaRect(contour)
                if min(d1,d2)/max(d1,d2,1)>.14:continue
            elif area<c.min_blob_area:continue
            activity=float(activity_map[y:y+bh,x:x+bw].sum())
            if activity<40:continue
            prior=None
            if self.previous_rotors and t-self.previous_rotors_t<c.geometry_prior_max_age_s:
                guess=np.array([x+bw/2,y+bh/2])*b
                previous=min(self.previous_rotors,key=lambda r:np.linalg.norm(r['center']+self.velocity*(t-self.previous_rotors_t)-guess))
                predicted=previous['center']+self.velocity*(t-self.previous_rotors_t)
                if np.linalg.norm(predicted-guess)<max(20,previous['major']*.6):
                    prior=dict(previous,center=predicted)
            measurement=fit_rotor_geometry(contour.astype(float)*b,self.w*.65,prior,c.enable_partial_geometry)
            if measurement is None or (measurement['mode']=='line' and not c.enable_line_geometry):continue
            if thin_only and measurement['mode']!='line':continue
            if measurement['mode']=='arc':
                angle=np.deg2rad(measurement['angle']);a=measurement['major']/2;d=measurement['minor']/2
                half=np.sqrt([(a*np.cos(angle))**2+(d*np.sin(angle))**2,
                              (a*np.sin(angle))**2+(d*np.cos(angle))**2])
                box=np.r_[measurement['center']-half,measurement['center']+half]
            else:box=np.array([x*b,y*b,(x+bw)*b,(y+bh)*b],float)
            candidates.append(dict(measurement,bbox=box,activity=activity))
        return candidates

    def _group(self,candidates,t):
        if not candidates:return None
        centers=np.array([x['center'] for x in candidates])
        predicted=self.center+self.velocity*(t-self.last_t) if self.center is not None else None
        best=None;best_score=-np.inf;seen_groups=set()
        for seed in candidates:
            distance=np.linalg.norm(centers-seed['center'],axis=1)
            near=[i for i in np.argsort(distance) if distance[i]<max(80.,seed['major']*4.5)][:6]
            allow_single=self.layout_measured_t is not None and t-self.layout_measured_t<self.cfg.single_rotor_max_age_s
            for count in ((4,3,2,1) if allow_single else (4,3,2)):
                if len(near)<count:continue
                for ids in combinations(near,count):
                    key=tuple(sorted(ids))
                    if key in seen_groups:continue
                    seen_groups.add(key)
                    rs=[candidates[i] for i in ids];ds=np.array([r['major'] for r in rs])
                    if count==1:
                        if self.constellation is None:continue
                        expected=[q['center']+self.velocity*(t-self.constellation_t) for q in self.constellation]
                        gap=min(np.linalg.norm(rs[0]['center']-q) for q in expected)
                        if gap>max(12.,self.last_extent*.15):continue
                        score=np.log1p(rs[0]['activity'])*rs[0]['shape_quality']+5-gap
                        if score>best_score:best_score=score;best=rs
                        continue
                    if ds.max()/max(ds.min(),1)>3:continue
                    ps=centers[list(ids)]; distances=np.linalg.norm(ps[:,None]-ps[None,:],axis=2)
                    if np.min(distances+np.eye(count)*1e8)<np.median(ds)*.35:continue
                    extent=np.ptp(ps,axis=0);diameter=max(extent)
                    if diameter>np.median(ds)*5 or diameter<np.median(ds)*.5:continue
                    aspect=max(extent)/max(min(extent),10)
                    score=sum(np.log1p(r['activity'])*r['shape_quality'] for r in rs)+count*5
                    score-=abs(np.log(max(aspect,1)))*2
                    if predicted is not None and t-self.last_seen<self.cfg.reset_gap_s:
                        gap=np.linalg.norm(ps.mean(axis=0)-predicted)
                        if gap>max(80,self.last_extent*1.3):continue
                        score-=gap/max(20,self.last_extent)*15
                    if score>best_score:best_score=score;best=rs
        return best

    def _frequency(self,rotors,t):
        if not self.history:return [],0.
        e=np.concatenate([q for _,q in self.history])
        start=int(t*1e6)-self.cfg.frequency_window_us
        e=e[(e['t']>=start)&(e['t']<int(t*1e6))]
        if len(e)>self.cfg.max_frequency_events:
            e=e[::int(np.ceil(len(e)/self.cfg.max_frequency_events))]
        if len(e)<100:return [np.nan]*len(rotors),0.
        # Approximate translational dewarping uses only the past tracked velocity.
        age=(t-e['t']/1e6)
        xx=e['x']+age*self.velocity[0]; yy=e['y']+age*self.velocity[1]
        fs=10000.; n=int(self.cfg.frequency_window_us/100)
        ts=((e['t']-start)//100).astype(int)
        frequencies=[];qualities=[];diagnostics=[]
        for rotor in rotors:
            center=rotor['center'];r=rotor['major']/2
            angle=np.deg2rad(rotor.get('angle',0.))
            dx,dy=xx-center[0],yy-center[1]
            along=dx*np.cos(angle)+dy*np.sin(angle)
            across=-dx*np.sin(angle)+dy*np.cos(angle)
            half_width=max(2.,rotor.get('minor',r*2)/2*.65)
            selected=(along>r*.1)&(along<r*.85)&(abs(across)<half_width)
            bins=ts[selected];bins=bins[(bins>=0)&(bins<n)]
            if len(bins)<50:
                frequencies.append(np.nan);diagnostics.append(dict(valid=False,reason='insufficient_events'));continue
            signal=np.bincount(bins,minlength=n)
            freq,power=periodogram(signal,fs=fs,window='hann',detrend='linear')
            ok=(freq>=self.cfg.frequency_min_hz)&(freq<=self.cfg.frequency_max_hz)
            indexes=np.flatnonzero(ok)
            k=indexes[np.argmax(power[indexes])]
            quality=float(power[k]/max(np.median(power[indexes]),1e-8))
            if quality<6:
                frequencies.append(np.nan);diagnostics.append(dict(valid=False,reason='weak_peak'));continue
            # Two past half-windows must support a coherent phase at the peak.
            centered=signal-signal.mean();carrier=np.exp(-2j*np.pi*freq[k]*np.arange(n)/fs)
            z1=np.sum(centered[:n//2]*carrier[:n//2]);z2=np.sum(centered[n//2:]*carrier[n//2:])
            coherence=float(abs(z1+z2)/max(abs(z1)+abs(z2),1e-9))
            competitors=[]
            for factor in (.5,2.):
                f=freq[k]*factor
                if self.cfg.frequency_min_hz<=f<=self.cfg.frequency_max_hz:
                    j=int(np.argmin(abs(freq-f)))
                    competitors.append(float(np.max(power[max(0,j-1):j+2])/max(power[k],1e-9)))
            ambiguous=bool(max(competitors,default=0.)>.35)
            q=min(1.,np.log10(quality)/2)*coherence*(.5 if ambiguous else 1.)
            frequencies.append(float(freq[k]));qualities.append(q)
            diagnostics.append(dict(valid=True,peak_hz=float(freq[k]),peak_to_median=quality,
                                    phase_coherence=coherence,harmonic_ambiguous=ambiguous,quality=q))
        self.frequency_diagnostics=diagnostics
        return frequencies,float(np.mean(qualities)) if qualities else 0.
