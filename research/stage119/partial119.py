"""Causal partial-conic observation pilot; no missing rotor is a measurement.

Standard projective fitting/association, specialized to a learned rotor layout.
Uncertainty gates are diagnostics, not calibrated probabilities or novelty proof.
"""
from pathlib import Path
import copy,itertools,sys,time
import cv2,numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage117')]
from fixed117 import FixedModel,geom,fixed_jac,pose_uncertainty
TEMPLATE=np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
PARAMS=dict(max_nfev=80,maximum_rms_px=3.,maximum_sigma_px=6.,maximum_shape_center_spread_px=6.,
    association_radius_major_fraction=.75,association_scale_major_fraction=.25,association_minimum_cost_margin=1.,
    modes=['body_hold','affine3','conic3','guarded3'],
    admission='Only after original five-full-frame lock; exactly three current full ellipses, current body, non-line geometry, fresh same identity. No initial lock from three points.',
    association='24 injective assignments of three current ellipse centers to four previous predicted ellipse centers, shifted by current body displacement. Best mean normalized cost; guarded requires second-best minus best>=1. No retained fourth measurement.',
    fit='Current three fitted-conic perimeters only, 64 samples each. Retained radius/aspect; H8. Initialization is affine update from the same three center correspondences. Same80 evaluations/3px RMS as original.',
    uncertainty='Guarded checks rank8/local conditional center sigma and center spread over five retained shape samples, each refit from current three conics. Neither bounds nor statistical coverage guarantees; shape ensemble can miss systematic bias.',
    control='Affine update from same three correspondences, and original body hold/expiry. All use the same original full-four model, initial lock, anchor and center filter.')

def project(H,p):
    q=np.c_[p,np.ones(len(p))]@H.T
    return q[:,:2]/q[:,2,None]

def native_H(f):
    s=f['scale_px'];x,y=f['origin_px']
    return np.array([[s,0,x],[0,s,y],[0,0,1.]])@np.asarray(f['homography_normalized'])

def predicted_centers(H,shape):
    phase=np.linspace(0,2*np.pi,64,endpoint=False)
    return np.array([cv2.fitEllipseDirect(np.float32(project(H,xy* [1.,shape[1]]+shape[0]*np.c_[np.cos(phase),np.sin(phase)])))[0] for xy in TEMPLATE])

def associate(rotors,H,shape,body_delta):
    old=predicted_centers(H,shape);expected=old+body_delta
    current=np.array([[r['cx'],r['cy']] for r in rotors]);major=np.array([r['major'] for r in rotors])
    scale=np.maximum(3.,major*PARAMS['association_scale_major_fraction']);candidates=[]
    for ids in itertools.permutations(range(4),3):
        distance=np.linalg.norm(current-expected[list(ids)],axis=1)
        candidates.append((float(np.mean((distance/scale)**2)),ids,distance))
    candidates.sort(key=lambda z:(z[0],z[1]));cost,ids,distance=candidates[0]
    diag=dict(ids=list(ids),best_cost=cost,cost_margin=candidates[1][0]-cost,
        max_distance_px=float(max(distance)),gate_pass=bool(np.all(distance<=np.maximum(3.,major*PARAMS['association_radius_major_fraction']))))
    A=cv2.getAffineTransform(old[list(ids)].astype(np.float32),current.astype(np.float32))
    affine=np.vstack([A,[0,0,1.]])@H
    return ids,affine,diag

def fit_three(rotors,ids,Hinitial,shape):
    start=time.perf_counter();centers=np.array([[r['cx'],r['cy']] for r in rotors]);origin=centers.mean(0)
    scale=max(1.,float(np.sqrt(np.mean(np.sum((centers-origin)**2,axis=1)))))
    T=np.array([[scale,0,origin[0]],[0,scale,origin[1]],[0,0,1.]])
    H=np.linalg.solve(T,Hinitial);H/=H[2,2]
    points=[np.c_[(geom.geometry58.perimeter(geom.geometry58.from_rotor(r))-origin)/scale,np.ones(64)] for r in rotors]
    template=TEMPLATE[list(ids)]
    def residual(p):
        H=np.r_[p,1.].reshape(3,3)
        try:inverse=np.linalg.inv(H)
        except np.linalg.LinAlgError:return np.full(192,1e6)
        result=[]
        for xy,edge in zip(template,points):
            x,y=xy*[1.,shape[1]];C=np.array([[1.,0,-x],[0,1.,-y],[-x,-y,x*x+y*y-shape[0]**2]])
            cp=edge@(inverse.T@C@inverse)
            result.extend(scale*np.sum(cp*edge,axis=1)/np.maximum(2*np.linalg.norm(cp[:,:2],axis=1),1e-12))
        return np.array(result)
    try:
        f=least_squares(residual,H.reshape(-1)[:8],jac=lambda p:fixed_jac(p,points,template,scale,shape),max_nfev=80,loss='linear',x_scale='jac')
        H=np.r_[f.x,1.].reshape(3,3);pixel=T@H
        q=np.c_[TEMPLATE*[1.,shape[1]],np.ones(4)]@H.T
        return dict(valid=bool(np.isfinite(f.fun).all()),optimizer_success=bool(f.success),rms_px=float(np.sqrt(np.mean(f.fun**2))),
            nfev=int(f.nfev),geometric_center=project(pixel,np.zeros((1,2)))[0].tolist(),projected_hubs=project(pixel,TEMPLATE*[1.,shape[1]]).tolist(),
            pixel_homography=pixel.tolist(),radius_in_template_units=float(shape[0]),frame_aspect=float(shape[1]),
            same_projective_halfspace=bool(np.all(q[:,2]>0)),uncertainty=pose_uncertainty(f,scale),current_conic_count=3,
            fitted_perimeters_are_completed_observed_ellipses=True,missing_rotor_is_observation=False,elapsed_ms=(time.perf_counter()-start)*1000)
    except (ValueError,np.linalg.LinAlgError,FloatingPointError):return dict(valid=False,optimizer_success=False,reason='partial_fit_failure')

class PartialModel:
    def __init__(self,mode):
        self.mode=mode;self.model=FixedModel();self.H=None;self.shape=None;self.body=None;self.last_pose=None
    def step(self,t,body,rotors,axis,baseline):
        if self.model.last_body is not None and t-self.model.last_body>.5:self.__init__(self.mode)
        if len(rotors)!=3 or self.mode=='body_hold':
            out,d=self.model.step(t,body,rotors,axis,baseline)
            f=d['measurement'].get('fit',{})
            if d['current'] and self.model.shape is not None and 'homography_normalized' in f:
                self.H=native_H(f);self.shape=np.array([f['radius_in_template_units'],f['frame_aspect']]);self.body=np.array(body['center']);self.last_pose=t
            d['partial119']=dict(attempted=False,mode=self.mode)
            return out,d
        if body.get('detected'):self.model.last_body=t
        m=dict(valid=False,reason='three_requires_previous_full_model',current_count=3);diagnostic=dict(attempted=False,mode=self.mode)
        if self.model.center.ready and self.H is not None and body.get('detected') and t-self.last_pose<=.5:
            m,diagnostic=self.measure(body,rotors)
        self.model.center.input_measurement=m;out,d=self.model.center.step(t,body,rotors,axis)
        if d['current']:
            self.H=np.asarray(m['fit']['pixel_homography']);self.body=np.array(body['center']);self.last_pose=t;d['mode']='three_conic_joint' if self.mode!='affine3' else 'three_center_affine'
        d.update(structure_ready=self.model.shape is not None,structure=None if self.model.shape is None else self.model.shape.tolist(),
            structure_learned_at=self.model.learned_at,partial119=diagnostic)
        return out,d
    def measure(self,body,rotors):
        diag=dict(attempted=True,mode=self.mode);m=dict(valid=False,reason='invalid_three_geometry',current_count=3)
        p=np.array([[r['cx'],r['cy']] for r in rotors]);b=np.asarray(body['bbox']);margin=(b[2:]-b[:2])*.25
        if any(r.get('inferred_shape',False) or not r.get('observed_now',True) or r.get('mode','ellipse')!='ellipse' or r.get('minor',0)<3 or r['minor']/max(r['major'],1)<.12 or r.get('geometry_quality',1)<.45 for r in rotors):return m,diag
        if np.any(p<b[:2]-margin) or np.any(p>b[2:]+margin):return dict(m,reason='outside_current_body'),diag
        s=np.linalg.svd(p-p.mean(0),compute_uv=False);flat=float(s[1]/max(s[0],1e-9))
        if flat<.08:return dict(m,reason='three_near_collinear'),diag
        ids,initial,association=associate(rotors,self.H,self.shape,np.array(body['center'])-self.body);diag['association']=association
        if not association['gate_pass']:return dict(m,reason='association_distance'),diag
        if self.mode=='guarded3' and association['cost_margin']<1:return dict(m,reason='association_ambiguous'),diag
        if self.mode=='affine3':
            f=dict(valid=True,optimizer_success=True,rms_px=0.,geometric_center=project(initial,np.zeros((1,2)))[0].tolist(),
                projected_hubs=project(initial,TEMPLATE*[1.,self.shape[1]]).tolist(),pixel_homography=initial.tolist(),same_projective_halfspace=True,
                radius_in_template_units=float(self.shape[0]),frame_aspect=float(self.shape[1]),current_conic_count=0,missing_rotor_is_observation=False)
            sigma=1.
        else:
            f=fit_three(rotors,ids,initial,self.shape);diag['fit']=f
            if not f.get('valid') or not f['optimizer_success']:return dict(m,reason='partial_optimizer_failed'),diag
            sigma=f['uncertainty']['center_sigma_px']
            if f['rms_px']>3 or not f['same_projective_halfspace']:return dict(m,reason='partial_conics_inconsistent'),diag
            if self.mode=='guarded3':
                if sigma is None or sigma>6:return dict(m,reason='partial_center_conditioning'),diag
                ensemble=[];swapped=self.shape[1]<1
                for radius,aspect in self.model.shape_samples:
                    shape=(radius/aspect,1/aspect) if swapped else (radius,aspect)
                    q=fit_three(rotors,ids,initial,shape)
                    if not q.get('valid') or not q['optimizer_success'] or not q['same_projective_halfspace'] or q['rms_px']>3:
                        return dict(m,reason='learned_shape_ensemble_inconsistent'),diag
                    ensemble.append(q['geometric_center'])
                spread=float(max(np.linalg.norm(np.asarray(ensemble)-f['geometric_center'],axis=1)));diag['shape_center_spread_px']=spread
                if spread>6:return dict(m,reason='learned_shape_center_unstable'),diag
                sigma=max(sigma,spread)
        center=np.array(f['geometric_center']);extent=float(np.linalg.norm(b[2:]-b[:2]))
        if np.linalg.norm(center-body['center'])>.35*extent:return dict(m,reason='center_body_disagreement'),diag
        # Four projected hubs below are model outputs, not four observations.
        m.update(valid=True,reason='current_three_model_center',center=center.tolist(),hubs=f['projected_hubs'],quality=min(r.get('geometry_quality',1) for r in rotors),
            view_weight=geom.rigid104.smooth(flat,.08,.25),center_noise_proxy_px=float(sigma or 1),frame_aspect=f['frame_aspect'],radius=f['radius_in_template_units'],
            fit=f,observed_template_ids=list(ids),projected_hubs_are_model_predictions=True)
        return m,diag
