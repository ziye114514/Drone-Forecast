"""Yaw-free image-plane disk/body-axis observer, conditional on past four-point lock.

Event support is not an RGB silhouette. Body major axis -> thrust axis only in
an approximate orthographic, near edge-on observation model. No metric pose,
mass, wind, rotor speed or true thrust magnitude is recovered by this module.
"""
import cv2,numpy as np

PARAMS=dict(density_thresholds=[1.4,3.,6.],close_kernel=7,min_support=70,
    min_extent_px=22,max_extent_px=950,side_ratio_max=.38,
    multiscale_max_deg=12.,line_pca_max_deg=15.,track_gate_fraction=.8,
    coast_s=.3,reacquire_max_axis_change_deg=65.,body_confidence_min=.45)

def angle_distance(a,b):return float(abs((a-b+np.pi/2)%np.pi-np.pi/2))

def principal(points,weights=None):
    points=np.asarray(points,float)
    if len(points)<15:return None
    center=points.mean(0) if weights is None else np.average(points,axis=0,weights=weights)
    d=points-center;cov=d.T@d/max(1,len(d)) if weights is None else (d*weights[:,None]).T@d/max(weights.sum(),1e-9)
    values,vectors=np.linalg.eigh(cov);u=vectors[:,-1]
    angle=float(np.arctan2(u[1],u[0])%np.pi)
    return dict(center=center,angle=angle,ratio=float(np.sqrt(max(values[0],0)/max(values[1],1e-9))))

def body_measurement(density,bbox,skip_unqualified_lines=False,body_workspace=None,center_uncertainty=False):
    h,w=density.shape;b=np.array(bbox,float)
    lo=np.maximum(np.floor(b[:2]).astype(int),0);hi=np.minimum(np.ceil(b[2:]).astype(int),[w-1,h-1])
    if np.any(hi-lo<12):return dict(valid=False,reason='small_roi')
    crop=density[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
    measurements=[];pointsets=[]
    for threshold in PARAMS['density_thresholds']:
        if body_workspace is not None and hasattr(body_workspace,'points'):
            points=body_workspace.points(crop>=threshold)
            if len(points)<PARAMS['min_support']:continue
        else:
            yy,xx=np.nonzero(crop>=threshold)
            if len(xx)<PARAMS['min_support']:continue
            points=np.column_stack((xx,yy)).astype(float)
            if len(points)>5000:points=points[::int(np.ceil(len(points)/5000))]
        m=principal(points);measurements.append(m);pointsets.append(points)
    if len(measurements)<2:return dict(valid=False,reason='insufficient_multiscale_support')
    base=measurements[0];angle=base['angle'];spread=max(np.degrees(angle_distance(angle,m['angle'])) for m in measurements)
    points=pointsets[0];p=points+lo
    low=np.quantile(p,.015,axis=0);high=np.quantile(p,.985,axis=0);extent=float(max(high-low))
    if extent<PARAMS['min_extent_px']:return dict(valid=False,reason='small_extent')
    if skip_unqualified_lines and (base['ratio']>=PARAMS['side_ratio_max'] or spread>PARAMS['multiscale_max_deg']):
        # LSD agreement can only add a rejection; it cannot rescue a failed
        # thickness/multiscale gate. Keep the original path as the control.
        # Report the omitted diagnostic as unknown rather than zero lines.
        axis=np.array([-np.sin(angle),np.cos(angle)])
        confidence=float(np.clip((1-base['ratio'])*np.exp(-spread/15)*min(1,len(points)/250),0,1))
        return dict(valid=False,reason='thick_or_inconsistent',center=((low+high)/2).tolist(),
            bbox=np.concatenate((low,high)).tolist(),angle_rad=angle,normal_axis_xy=axis.tolist(),
            ratio=base['ratio'],multiscale_spread_deg=spread,line_count=None,line_pca_disagreement_deg=None,
            confidence=confidence,event_support=len(points),extent_px=extent,
            camera_model='Weak/orthographic image-axis approximation; no calibrated 3D normal.',
            sign_observed=False,thrust_magnitude_observed=False,
            line_diagnostic_skipped_after_failed_necessary_shape_gate=True)
    # OpenCV LSD is an actual line detector. Lines are image support, not the
    # four virtual spokes of the disk model.
    gray=np.uint8(np.clip(np.log1p(crop)/4,0,1)*255) if body_workspace is None else body_workspace.render(crop)
    lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0] if body_workspace is None else body_workspace.detect(gray)
    usable=[]
    if lines is not None:
        for line in lines.reshape(-1,4):
            d=line[2:]-line[:2];length=float(np.linalg.norm(d))
            if length<max(8,extent*.08):continue
            a=float(np.arctan2(d[1],d[0])%np.pi)
            if np.degrees(angle_distance(a,angle))<25:usable.append((a,length))
    line_angle=None;agreement=None
    if len(usable)>=2:
        aa=np.array([v[0] for v in usable]);weights=np.array([v[1] for v in usable])**2
        z=np.sum(weights*np.exp(2j*aa));line_angle=float((np.angle(z)/2)%np.pi)
        agreement=float(abs(z)/weights.sum())
    disagreement=float(np.degrees(angle_distance(line_angle,angle))) if line_angle is not None else None
    thin=base['ratio']<PARAMS['side_ratio_max']
    valid=bool(thin and spread<=PARAMS['multiscale_max_deg'] and
               (line_angle is None or disagreement<=PARAMS['line_pca_max_deg']))
    if valid and line_angle is not None and agreement>.7:
        z=np.exp(2j*angle)+np.exp(2j*line_angle);angle=float((np.angle(z)/2)%np.pi)
    confidence=float(np.clip((1-base['ratio'])*np.exp(-spread/15)*min(1,len(points)/250),0,1))
    axis=np.array([-np.sin(angle),np.cos(angle)])
    result=dict(valid=valid,reason='side_body_axis' if valid else 'thick_or_inconsistent',
        center=((low+high)/2).tolist(),bbox=np.r_[low,high].tolist(),
        angle_rad=angle,normal_axis_xy=axis.tolist(),ratio=base['ratio'],
        multiscale_spread_deg=spread,line_count=len(usable),line_pca_disagreement_deg=disagreement,
        confidence=confidence,event_support=len(points),extent_px=extent,
        camera_model='Weak/orthographic image-axis approximation; no calibrated 3D normal.',
        sign_observed=False,thrust_magnitude_observed=False)
    if center_uncertainty:
        centers=np.array([(np.quantile(p,.015,axis=0)+np.quantile(p,.985,axis=0))/2 for p in pointsets])
        delta=centers-centers[0]
        result.update(center_multiscale_covariance_proxy_px2=(delta.T@delta/max(len(delta),1)).tolist(),
            center_uncertainty_calibrated=False,correlated_density_levels_not_independent_measurements=True)
    return result

def proposals(density):
    mask=np.uint8(cv2.GaussianBlur(density,(5,5),1.)>=1.4)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((PARAMS['close_kernel'],)*2,np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    result=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i];extent=max(w,h)
        if area<PARAMS['min_support'] or not PARAMS['min_extent_px']<extent<PARAMS['max_extent_px']:continue
        # The silhouette candidate may be connected by body/arms, rather than
        # each propeller being segmented independently.
        result.append(dict(bbox=[float(x),float(y),float(x+w),float(y+h)],area=int(area)))
    return sorted(result,key=lambda r:r['area'],reverse=True)[:12]

class BodyObserver:
    def __init__(self):
        self.memory=None;self.last_t=None;self.last_seen=None;self.center=None
        self.velocity=np.zeros(2);self.extent=None;self.normal=None;self.last_axis_t=None
        self.axis_updates=0;self.mode='uninitialized'

    def step(self,t,density,observation,four_model):
        gap=self.last_t is not None and t-self.last_t>.5
        dt=.02 if self.last_t is None or gap else t-self.last_t;self.last_t=t
        if gap:self.center=None;self.last_seen=None;self.velocity[:]=0
        four=bool(four_model.get('active') and four_model['mode'] in ['initialized','rotor_affine']
                  and len(observation['rotors'])==4)
        if four:
            rs=observation['rotors'];angles=np.deg2rad([r['angle'] for r in rs]);z=np.mean(np.exp(2j*angles))
            if abs(z)>.75 and np.median([r['ratio'] for r in rs])<.9:
                a=float(np.angle(z)/2);n=np.array([-np.sin(a),np.cos(a)])
                if self.normal is None:
                    if n[1]>0:n=-n
                elif n@self.normal<0:n=-n
                self.normal=n;self.last_axis_t=t
            if self.memory is None:
                b=np.array(observation['bbox']);self.memory=dict(initialized_t=t,reference_extent=float(max(b[2:]-b[:2])),
                    reference_normal=self.normal.tolist() if self.normal is not None else [0.,-1.],
                    model='Head-tail-free equivalent disk plane; normalized image template, no CAD or metric dimensions.')
        if self.memory is None:return dict(t=t,active=False,mode='no_four_point_model',axis_valid=False)
        candidates=proposals(density)
        if observation.get('detected'):
            b=np.array(observation['bbox']);c=(b[:2]+b[2:])/2;half=(b[2:]-b[:2])/2
            candidates.append(dict(bbox=np.r_[c-half*1.15,c+half*1.15].tolist(),area=0,source='current_rotor_roi'))
        expected=None
        if self.center is not None and self.last_seen is not None and t-self.last_seen<.3:
            expected=self.center+self.velocity*dt
        options=[]
        for candidate in candidates:
            m=body_measurement(density,candidate['bbox'])
            if 'center' not in m:continue
            extent=m['extent_px'];scale=extent/self.memory['reference_extent']
            if not .15<scale<8:continue
            center=np.array(m['center']);distance=float(np.linalg.norm(center-expected)) if expected is not None else 0.
            if expected is not None and distance>max(35,self.extent*.8):continue
            # Reacquisition uses retained undirected axis; this is a single-UAV
            # prior, not a semantic classifier or a guarantee of target identity.
            if expected is None and self.normal is not None:
                a=np.array(m['normal_axis_xy']);deviation=np.degrees(np.arccos(np.clip(abs(a@self.normal),0,1)))
                if deviation>PARAMS['reacquire_max_axis_change_deg']:continue
            score=np.log1p(m['event_support'])+m['confidence']*2-distance/max(20,extent)*8
            if candidate.get('source')=='current_rotor_roi':score+=.2
            options.append((score,m))
        measured=False;axis_measured=False;body=None;mode='coast'
        if options:
            _,body=max(options,key=lambda v:v[0]);new=np.array(body['center'])
            if self.center is not None:self.velocity=.7*self.velocity+.3*np.clip((new-self.center)/max(dt,.001),-2000,2000)
            self.center=new;self.extent=body['extent_px'];self.last_seen=t;measured=True
            mode='four_body_position' if four else 'whole_body_position'
            if body['valid'] and body['confidence']>=PARAMS['body_confidence_min']:
                n=np.array(body['normal_axis_xy'])
                if self.normal is None:
                    if n[1]>0:n=-n
                elif n@self.normal<0:n=-n
                if self.normal is None:self.normal=n
                else:
                    n=.7*self.normal+.3*n;self.normal=n/max(np.linalg.norm(n),1e-9)
                self.last_axis_t=t;self.axis_updates+=1;axis_measured=True;mode='silhouette_axis'
        elif self.center is not None:self.center=self.center+self.velocity*dt
        active=self.center is not None and self.last_seen is not None and t-self.last_seen<=.3
        axis_valid=self.normal is not None and self.last_axis_t is not None and t-self.last_axis_t<=.3
        return dict(t=t,active=bool(active),mode=mode,center=self.center.tolist() if self.center is not None else None,
            position_measured=measured,axis_measured=axis_measured,axis_valid=bool(axis_valid),
            normal_axis_xy=self.normal.tolist() if self.normal is not None else None,
            memory=self.memory,axis_updates=self.axis_updates,body=body,four_current_observations=four,
            sign_source='temporal continuity / initial image-up prior',arrow_sign_independently_observed=False,
            calibrated_3d_pose=False,thrust_magnitude_observed=False)

class AxisForecast:
    """Direction-constrained acceleration ablation; amplitude from past motion.

    Pixel gravity is an explicit assumption, not estimated metric gravity.
    This is not RPM/thrust sensing or physical parameter identification.
    """
    def __init__(self):self.history=[];self.last_t=None;self.weights=np.ones(3)/3;self.queue=[]
    def step(self,t,state,body,measured_center=None):
        if self.last_t is not None and t-self.last_t>.5:self.__init__()
        self.last_t=t;x=np.array(state)
        if measured_center is not None:self.history.append((t,np.array(measured_center)))
        self.history=[v for v in self.history if t-v[0]<=.35]
        acceleration=np.zeros(2)
        if len(self.history)>=8 and self.history[-1][0]-self.history[0][0]>.15:
            times=np.array([v[0]-t for v in self.history]);Z=np.array([v[1] for v in self.history])
            D=np.c_[np.ones(len(times)),times,.5*times**2]
            coef=np.linalg.solve(D.T@D+np.diag([1e-6,1e-5,1e-5]),D.T@Z)
            acceleration=np.clip(coef[2],-2000,2000)
        axis_ok=body.get('axis_valid',False);n=np.array(body.get('normal_axis_xy') or [0.,-1.])
        predictions={'cv':np.zeros(2),'ca_history':acceleration}
        for gravity in [0.,200.,500.]:
            g=np.array([0.,gravity]);beta=float(np.clip((acceleration-g)@n,0,2500))
            predictions['axis_g'+str(int(gravity))]=np.clip(beta*n+g,-2500,2500) if axis_ok else np.zeros(2)
        names=['cv','ca_history','axis_g200']
        self.queue=[v for v in self.queue if t-v[0]<=.26]
        if measured_center is not None:
            remaining=[]
            for old_t,old in self.queue:
                if t-old_t>=.2:
                    loss=[]
                    for name in names:
                        p,v,a=old[name];estimate=p+v*(t-old_t)+.5*a*(t-old_t)**2
                        loss.append(min(16,np.linalg.norm(estimate-measured_center)**2/35**2))
                    self.weights*=np.exp(-.08*np.array(loss));self.weights/=self.weights.sum();self.weights=.99*self.weights+.01/3
                else:remaining.append((old_t,old))
            self.queue=remaining
        self.queue.append((t,{n:(x[:2].copy(),x[2:].copy(),predictions[n].copy()) for n in names}))
        predictions['causal_fused']=sum(w*predictions[n] for w,n in zip(self.weights,names))
        return dict(accelerations={n:v.tolist() for n,v in predictions.items()},weights=self.weights.tolist(),
                    axis_applied=axis_ok,thrust_amplitude_source='Past image-position quadratic fit projected onto an assumed pixel-force axis; not RPM truth.')
