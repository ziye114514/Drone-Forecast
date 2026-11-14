"""Stable per-rotor IDs and polarity/phase-aware periodicity sensing.

FFT quadrants follow the target paper; rotor-plane phase consistency is an
independent reduced implementation inspired by HelixTrack, not its GN tracker.
"""
from dataclasses import dataclass
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment,minimize_scalar
from scipy.signal import periodogram,find_peaks
from .config import Config
from .perception import RotorPerception
from .geometry import fit_rotor_geometry


class PaperPerception(RotorPerception):
    def __init__(self,width,height,paper_cfg):
        cfg=Config(step_us=paper_cfg.step_us,spatial_bin=paper_cfg.spatial_bin,
                   min_pixel_count=paper_cfg.min_repeated_count,density_quantile=paper_cfg.density_quantile,
                   min_blob_area=paper_cfg.min_area,max_candidates=paper_cfg.max_candidates,
                   frequency_window_us=paper_cfg.frequency_window_us,
                   max_frequency_events=paper_cfg.max_buffer_events,enable_partial_geometry=False,
                   enable_line_geometry=True,opencv_threads=paper_cfg.opencv_threads)
        super().__init__(width,height,cfg);self.paper_cfg=paper_cfg

    def detect(self,events,t):
        observation,total=super().detect(events,t)
        if observation['detected']:
            for serialized,measured in zip(observation['rotors'],self.previous_rotors):
                serialized['activity']=float(measured['activity'])
        return observation,total

    def _frequency(self,rotors,t):
        # Periodic measurements are owned by stable rotor IDs below, not list order.
        return [np.nan]*len(rotors),0.

    def _candidates(self,contours,activity_map,t,b,thin_only=False):
        found=[];partial=[]
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:self.cfg.max_candidates*3]:
            if len(contour)<6:continue
            area=cv2.contourArea(contour)
            if area>self.cfg.max_blob_fraction*activity_map.size:continue
            x,y,bw,bh=cv2.boundingRect(contour)
            if max(bw,bh)*b<10 or (not thin_only and area<self.cfg.min_blob_area):continue
            try:(cx,cy),(d1,d2),angle=cv2.fitEllipse(contour.astype(np.float32))
            except cv2.error:continue
            major,minor=max(d1,d2)*b,min(d1,d2)*b
            if major<10 or major>self.w*.5 or major>max(bw,bh)*b*1.5:continue
            angle=np.deg2rad(angle+(90 if d1<d2 else 0));ratio=minor/max(major,1)
            line=ratio<.10 or minor<3
            if thin_only and not line:continue
            fill=area/max(np.pi*d1*d2/4,1.)
            if not line and not .40<fill<1.6:
                partial.append(contour)
                continue
            activity=float(activity_map[y:y+bh,x:x+bw].sum())
            if activity<30:continue
            quality=float(np.exp(-abs(fill-1)))
            geometry_quality=quality*(.6 if line else 1.)
            if ratio>.9:geometry_quality*=max(.05,(1-ratio)/.1)
            center_sigma=float(np.clip(major*.015,1.,6.))
            found.append(dict(center=np.array([cx*b,cy*b]),major=float(major),minor=float(max(minor,1)),
                angle=float(np.rad2deg(angle)),normal=np.array([-np.sin(angle),np.cos(angle)])*np.sqrt(max(0,1-ratio*ratio)),
                ratio=float(np.clip(ratio,0,1)),bbox=np.array([x*b,y*b,(x+bw)*b,(y+bh)*b],float),
                activity=activity,shape_quality=max(.4,quality),geometry_quality=geometry_quality,
                mode='line' if line else 'ellipse',visible_arc_fraction=None if line else 1.,
                support_fraction=1.,fit_residual_px=None,center_sigma_px=center_sigma,
                axis_angle_sigma_deg=5. if line else 3.,inferred_shape=False,axis_only=line))
        # Keep the paper's fast full-rim branch. Residual arcs may supplement
        # a recent measured rotor; never invent a disk from an arbitrary edge.
        if (self.paper_cfg.enable_partial_arcs and not thin_only and len(found)<4
                and self.previous_rotors and t-self.previous_rotors_t<.12):
            tried=0
            for contour in partial:
                x,y,bw,bh=cv2.boundingRect(contour);guess=np.array([x+bw/2,y+bh/2])*b
                old=min(self.previous_rotors,key=lambda r:np.linalg.norm(r['center']-guess))
                predicted=old['center']+self.velocity*(t-self.previous_rotors_t)
                if np.linalg.norm(predicted-guess)>max(15.,old['major']*.5):continue
                tried+=1
                measured=fit_rotor_geometry(contour.astype(float)*b,self.w*.5,dict(old,center=predicted),True)
                if measured is not None and measured['mode']=='arc' and measured['geometry_quality']>.2:
                    half=np.array([measured['major'],measured['major']])/2
                    found.append(dict(measured,bbox=np.r_[measured['center']-half,measured['center']+half],
                                      activity=float(activity_map[y:y+bh,x:x+bw].sum())))
                if tried>=2:break
        return found


def periodic_measurement(events,rotor,velocity,t,cfg,previous_hz=None):
    if len(events)<100:return dict(valid=False,reason='too_few_events')
    # Cull in integer sensor coordinates before rotating/dewarping the window.
    # The bound contains both ends of the moving disk, so this does not remove
    # events that the subsequent ellipse mask could accept.
    radius=.57*rotor['major']+2;window=cfg.frequency_window_us/1e6
    shift=np.asarray(velocity)*window
    low=np.array([rotor['cx'],rotor['cy']])-radius-np.maximum(shift,0)
    high=np.array([rotor['cx'],rotor['cy']])+radius-np.minimum(shift,0)
    events=events[(events['x']>=low[0])&(events['x']<=high[0])&
                  (events['y']>=low[1])&(events['y']<=high[1])]
    if len(events)<100:return dict(valid=False,reason='too_few_rotor_events')
    age=t-events['t']/1e6
    dx=events['x'].astype(float)+age*velocity[0]-rotor['cx']
    dy=events['y'].astype(float)+age*velocity[1]-rotor['cy']
    angle=np.deg2rad(rotor['angle']);c,s=np.cos(angle),np.sin(angle)
    u=dx*c+dy*s;v=-dx*s+dy*c;a=rotor['major']/2;b=max(rotor['minor']/2,1.)
    normalized=(u/a)**2+(v/b)**2
    in_disk=(normalized>.15)&(normalized<1.25)
    if in_disk.sum()<100:return dict(valid=False,reason='too_few_rotor_events')
    e=events[in_disk];u=u[in_disk];v=v[in_disk];normalized=normalized[in_disk]
    start=int(round(t*1e6))-cfg.frequency_window_us;n=cfg.frequency_window_us//100
    bins=((e['t']-start)//100).astype(int);good=(bins>=0)&(bins<n)
    e=e[good];u=u[good];v=v[good];bins=bins[good];normalized=normalized[good]
    if len(e)<100 or np.ptp(e['t'])<cfg.frequency_window_us*.6:
        return dict(valid=False,reason='period_window_warming_up')
    # A fixed quadrant avoids several spatially repeated blade crossings.
    quadrant=(u>0)&(v<0)
    if quadrant.sum()<60:quadrant=u>0
    signals=[];powers=[]
    for polarity in (0,1):
        selected=quadrant&(e['p']>0 if polarity else e['p']<=0)
        signal=np.bincount(bins[selected],minlength=n).astype(float)
        signal-=signal.mean();signals.append(signal)
        f,p=periodogram(signal,fs=10000.,window='hann',detrend='linear');powers.append(p)
    power=powers[0]+powers[1]
    allowed=(f>=cfg.frequency_min_hz)&(f<=cfg.frequency_max_hz);indices=np.flatnonzero(allowed)
    if not len(indices) or power[indices].max()<=0:return dict(valid=False,reason='empty_spectrum')
    noise=max(float(np.median(power[indices])),1e-12)
    peaks,_=find_peaks(power,distance=2);peaks=[k for k in peaks if allowed[k]]
    peaks=sorted(peaks,key=lambda k:power[k],reverse=True)[:5]
    if not peaks:peaks=[int(indices[np.argmax(power[indices])])]
    if power[peaks[0]]/noise<8:return dict(valid=False,reason='weak_periodicity')
    candidates=set()
    for k in peaks:
        for factor in (.5,1.,2.):
            hz=float(f[k]*factor)
            if cfg.frequency_min_hz<=hz<=cfg.frequency_max_hz:candidates.add(hz)
    if previous_hz is not None:candidates.add(float(np.clip(previous_hz,cfg.frequency_min_hz,cfg.frequency_max_hz)))
    stride=max(1,int(np.ceil(len(e)/cfg.max_frequency_samples)))
    theta=np.arctan2(v[::stride]/b,u[::stride]/a)*cfg.blade_count
    tau=e['t'][::stride]/1e6-t;polarity=e['p'][::stride]>0
    phase_usable=not rotor.get('axis_only',False) and b>=2.5 and rotor['ratio']>.09
    def phase_coherence(hz):
        best=0.;sign_best=1
        for sign in (-1,1):
            values=np.exp(1j*(theta*sign-2*np.pi*hz*tau))
            coherences=[abs(values[polarity==p].mean()) for p in (False,True) if (polarity==p).sum()>20]
            value=float(np.mean(coherences)) if coherences else 0.
            if value>best:best=value;sign_best=sign
        return best,sign_best
    results=[]
    for hz in sorted(candidates):
        k=int(np.argmin(abs(f-hz)));spectral=float(power[k]/max(power[peaks[0]],1e-12))
        phase,sign=phase_coherence(hz) if phase_usable else (0.,0)
        continuity=np.exp(-abs(np.log(hz/previous_hz))/.25) if previous_hz else 0.
        score=4*phase+.4*np.log1p(10*spectral)+(.3 if phase_usable else .8)*continuity
        results.append((score,hz,phase,sign,spectral))
    _,hz,phase,sign,spectral=max(results)
    resolution=10000./n
    if phase_usable and phase>.12:
        refinement=minimize_scalar(lambda q:-phase_coherence(q)[0],bounds=(max(cfg.frequency_min_hz,hz-resolution),min(cfg.frequency_max_hz,hz+resolution)),method='bounded',options={'maxiter':10,'xatol':.1})
        hz=float(refinement.x);phase,sign=phase_coherence(hz)
    k=int(np.argmin(abs(f-hz)));snr=float(power[k]/noise)
    ambiguous=not phase_usable or phase<.15
    quality=float(np.clip(np.log10(max(snr,1))/2,0,1))*(min(1.,phase/.4) if phase_usable else .45)
    valid=bool(snr>=6 and quality>.12)
    return dict(valid=valid,blade_pass_hz=hz,mechanical_rpm_assuming_blade_count=60*hz/cfg.blade_count,
                blade_count_assumed=cfg.blade_count,phase_coherence=phase,rotation_sign=sign,
                harmonic_ambiguous=ambiguous,phase_observable=phase_usable,quality=quality,
                peak_to_noise=snr,frequency_resolution_hz=resolution,event_support=len(e),
                interpretation='Mechanical RPM is provisional unless blade count and harmonics are independently validated.')


class RotorTrack:
    def __init__(self,identifier,rotor,t):
        self.id=identifier;self.rotor=dict(rotor);self.t=t;self.seen=t;self.velocity=np.zeros(2)
        self.frequency=None;self.frequency_variance=100.;self.frequency_t=None;self.frequency_quality=0.
        self.reference_hz=None;self.diagnostics=dict(valid=False,reason='warming_up');self.observations=1
        self.image_side=None

    def position(self,t):
        return np.array([self.rotor['cx'],self.rotor['cy']])+self.velocity*(t-self.t)

    def update(self,rotor,t):
        dt=max(t-self.t,.001);old=np.array([self.rotor['cx'],self.rotor['cy']]);new=np.array([rotor['cx'],rotor['cy']])
        self.velocity=.8*self.velocity+.2*np.clip((new-old)/dt,-2000,2000)
        self.rotor=dict(rotor);self.t=t;self.seen=t;self.observations+=1

    def update_frequency(self,measurement,t):
        self.diagnostics=measurement
        if not measurement.get('valid'):return
        hz=measurement['blade_pass_hz'];quality=measurement['quality']
        if self.frequency is None:self.frequency=hz
        else:
            self.frequency_variance+=max(1.,(t-self.frequency_t)*150)
            R=(measurement['frequency_resolution_hz']/max(quality,.1))**2
            if abs(hz-self.frequency)>max(30,self.frequency*.25) and measurement.get('phase_coherence',0)<.25:return
            gain=self.frequency_variance/(self.frequency_variance+R)
            self.frequency+=gain*(hz-self.frequency);self.frequency_variance*=1-gain
        self.frequency_t=t;self.frequency_quality=quality
        if self.reference_hz is None and quality>.3:self.reference_hz=self.frequency


class RotorSensing:
    def __init__(self,cfg):
        self.cfg=cfg;self.tracks=[];self.next_id=0;self.steps=0
        self.reference_total=None;self.last_observed=None

    def step(self,observation,history,t):
        self.steps+=1;current=observation.get('rotors',[])
        if current:
            if self.last_observed is not None and t-self.last_observed>.5:self.reference_total=None
            self.last_observed=t
        self.tracks=[q for q in self.tracks if t-q.seen<=self.cfg.track_expiry_s]
        matched=set();used=set()
        if self.tracks and current:
            cost=np.array([[np.linalg.norm(q.position(t)-[r['cx'],r['cy']])+.1*abs(q.rotor['major']-r['major']) for r in current] for q in self.tracks])
            rows,columns=linear_sum_assignment(cost)
            for i,j in zip(rows,columns):
                if cost[i,j]<max(20.,self.tracks[i].rotor['major']*.75):
                    self.tracks[i].update(current[j],t);matched.add(i);used.add(j)
        for j,r in enumerate(current):
            if j not in used and len(self.tracks)<4:
                self.tracks.append(RotorTrack(self.next_id,r,t));self.next_id+=1
        visible=[q for q in self.tracks if abs(q.seen-t)<1e-6]
        if len(visible)==4 and any(q.image_side is None for q in visible):
            ordered=sorted(visible,key=lambda q:q.rotor['cx'])
            for i,q in enumerate(ordered):q.image_side=-1 if i<2 else 1
        if self.steps%self.cfg.frequency_update_steps==0 and history:
            events=np.concatenate([e for _,e in history]);start=int(t*1e6)-self.cfg.frequency_window_us
            events=events[(events['t']>=start)&(events['t']<int(t*1e6))]
            if len(events)>self.cfg.max_buffer_events:events=events[::int(np.ceil(len(events)/self.cfg.max_buffer_events))]
            for q in visible:
                m=periodic_measurement(events,q.rotor,q.velocity,t,self.cfg,q.frequency)
                q.update_frequency(m,t)
        records=[]
        for q in self.tracks:
            fresh=q.frequency is not None and t-q.frequency_t<.16
            r=dict(q.rotor,rotor_id=q.id,observed_now=abs(q.seen-t)<1e-6,
                   last_seen_s=q.seen,frequency_valid=fresh,blade_pass_hz=q.frequency if fresh else None,
                   frequency_quality=q.frequency_quality if fresh else 0.,reference_hz=q.reference_hz,
                   frequency_diagnostics=q.diagnostics,image_side=q.image_side,
                   rpm_provisional=60*q.frequency/self.cfg.blade_count if fresh else None)
            records.append(r)
        valid=[r for r in records if r['frequency_valid'] and r['reference_hz'] is not None]
        complete=len(records)==4 and len(valid)==4
        denominator=sum(r['reference_hz']**2 for r in valid)
        total=sum(r['blade_pass_hz']**2 for r in valid)
        if complete and self.reference_total is None:self.reference_total=total
        differential=sum((r['image_side'] or 0)*r['blade_pass_hz']**2 for r in valid)/max(total,1.)
        return dict(rotors=records,four_visible=len(visible)==4,
                    four_ellipses=len(visible)==4 and all(q.rotor.get('mode')=='ellipse' for q in visible),four_frequencies=complete,
                    thrust_ratio=total/self.reference_total if complete and self.reference_total else None,
                    sum_frequency_squared=total if complete else None,differential_proxy=differential,
                    frequency_quality=float(np.mean([r['frequency_quality'] for r in valid])) if valid else 0.,
                    identity_interpretation='Stable visual rotor IDs; image-left/right slots are not calibrated body motor labels.')
