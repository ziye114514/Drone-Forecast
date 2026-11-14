"""Causal planar-center observer; reuse58 conic fit and49 projective center.

The learned layout is image/projective geometry, not a metric CAD/COM model.
Body-box and conic observations share events; their weights are heuristics,
never an independent-measurement covariance or a probability guarantee.
"""
from pathlib import Path
from collections import deque
import sys,time,copy
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage58'),str(ROOT/'research/stage49')]
from geometry58 import joint_fit,from_rotor
from rigid49 import intersection

PARAMS=dict(init_frames=5,minimum_flatness=.08,minimum_diagonal_sine=.12,
    maximum_fit_rms_px=3.,maximum_center_noise_px=6.,minimum_ratio=.12,
    maximum_dimension_change_factor=1.35,minimum_geometry_quality=.45,
    side_flatness=.08,full_flatness=.25,body_gap_reset_s=.5,
    correction_hold_s=.16,correction_expiry_s=.3,layout_expiry_s=.5,
    maximum_center_body_fraction=.35,maximum_body_gate_margin=.25,
    max_current_fit_gap_s=.04000001,initial_anchor='First qualified five-frame lock aligns model center to current body box once; constant image offset, not COM calibration.',
    uncertainty='First-order finite-difference propagation of supplied ellipse-center sigma through diagonal intersection; uncalibrated and excludes model/lens error.',
    fixed_model='Median frame aspect and common-radius ratio from initialization; reject later inconsistent dimensions, do not force arbitrary four blobs into a drone.',
    partial='No hidden rotor is inserted as an observation. Missing/edge-on layout uses body center with a short-lived last correction, then original body center.',
    limits='Unknown lens distortion, ellipse bias, true rotor identity and COM remain unverified. No stationary/periodic-motion prior. No 3-D pose or true thrust.')


def smooth(value,lo,hi):
    x=float(np.clip((value-lo)/(hi-lo),0.,1.));return x*x*(3.-2.*x)


def diagonal_noise(points,sigmas):
    p=np.asarray(points,float);base,_=intersection(p);J=np.zeros((2,8));eps=1e-3
    for i in range(8):
        plus=p.copy();minus=p.copy();plus.flat[i]+=eps;minus.flat[i]-=eps
        a,_=intersection(plus);b,_=intersection(minus);J[:,i]=(a-b)/(2*eps)
    cov=(J*np.repeat(np.asarray(sigmas,float)**2,2)[None,:])@J.T
    return float(np.sqrt(max(np.linalg.eigvalsh(cov)[-1],0.)))


def measure_four(rotors,body):
    current=[r for r in rotors if r.get('observed_now',True)]
    diag=dict(valid=False,reason='needs_four_current',current_count=len(current))
    if len(current)!=4:return diag
    if not body.get('detected'):return dict(diag,reason='no_current_body_identity')
    if any(r.get('mode','ellipse')!='ellipse' or r.get('inferred_shape',False) or
           r.get('minor',0)<3 or r['minor']/max(r['major'],1)<PARAMS['minimum_ratio'] for r in current):
        return dict(diag,reason='incomplete_or_line_geometry')
    p=np.array([[r['cx'],r['cy']] for r in current],float)
    b=np.asarray(body['bbox'],float);margin=(b[2:]-b[:2])*PARAMS['maximum_body_gate_margin']
    if np.any(p<b[:2]-margin) or np.any(p>b[2:]+margin):return dict(diag,reason='outside_current_body')
    hull=cv2.convexHull(p.astype(np.float32),returnPoints=False).reshape(-1)
    if len(hull)!=4:return dict(diag,reason='not_convex_four')
    pp=p[hull];singular=np.linalg.svd(pp-pp.mean(0),compute_uv=False)
    flat=float(singular[1]/max(singular[0],1e-9));c,sine=intersection(pp)
    diag.update(flatness=flat,diagonal_sine=float(sine))
    if flat<PARAMS['minimum_flatness'] or sine<PARAMS['minimum_diagonal_sine']:
        return dict(diag,reason='edge_on_ill_conditioned')
    sigma=diagonal_noise(pp,[current[i].get('center_sigma_px',1.) for i in hull])
    diag['center_noise_proxy_px']=sigma
    if sigma>PARAMS['maximum_center_noise_px']:return dict(diag,reason='center_noise_amplified')
    fit=joint_fit([from_rotor(r) for r in current]);diag['fit']=fit
    if not fit['valid'] or not fit.get('optimizer_success'):return dict(diag,reason='conic_optimizer_failed')
    if fit['rms_px']>PARAMS['maximum_fit_rms_px'] or not fit['same_projective_halfspace']:
        return dict(diag,reason='inconsistent_coplanar_circles')
    hubs=np.asarray(fit['projected_hubs']);center=np.asarray(fit['geometric_center'])
    extent=float(np.linalg.norm(b[2:]-b[:2]))
    if not np.isfinite(center).all() or np.linalg.norm(center-body['center'])>PARAMS['maximum_center_body_fraction']*extent:
        return dict(diag,reason='center_body_disagreement')
    quality=min(float(r.get('geometry_quality',1.)) for r in current)
    if quality<PARAMS['minimum_geometry_quality']:return dict(diag,reason='low_geometry_quality')
    return dict(diag,valid=True,reason='current_joint_conic_center',center=center.tolist(),hubs=hubs.tolist(),
        quality=quality,view_weight=smooth(flat,PARAMS['side_flatness'],PARAMS['full_flatness']),
        frame_aspect=fit['frame_aspect'],radius=fit['radius_in_template_units'])


class PlanarCenter:
    def __init__(self):self.reset()
    def reset(self):
        self.pending=deque(maxlen=PARAMS['init_frames']);self.ready=False;self.last_seen=None
        self.anchor=None;self.dimensions=None;self.last_geometry=None;self.correction=np.zeros(2)
        self.hubs=None;self.layout_center=None;self.last_current=None;self.initializations=0
    def step(self,t,body,rotors,body_axis):
        start=time.perf_counter()
        if self.last_seen is not None and t-self.last_seen>PARAMS['body_gap_reset_s']:self.reset()
        detected=bool(body.get('detected'))
        if detected:self.last_seen=t
        m=measure_four(rotors,body);initialized=False;current=False
        if m['valid']:
            if self.last_current is not None and t-self.last_current>PARAMS['max_current_fit_gap_s']:self.pending.clear()
            self.last_current=t
            if not self.ready:
                self.pending.append((t,m))
                if len(self.pending)==PARAMS['init_frames']:
                    dim=np.array([[v['frame_aspect'],v['radius']] for _,v in self.pending])
                    # Orientation is unordered: a 90deg relabel swaps aspect.
                    # Compare scalar radius / sqrt(aspect), invariant to swap.
                    stable=dim[:,1]/np.sqrt(dim[:,0])
                    if stable.max()/max(stable.min(),1e-9)<=PARAMS['maximum_dimension_change_factor']:
                        self.dimensions=float(np.median(stable));self.ready=True;initialized=True;self.initializations+=1
                        self.anchor=np.asarray(body['center'])-m['center']
                    else:self.pending.popleft()
            if self.ready:
                shape=m['radius']/np.sqrt(m['frame_aspect'])
                ratio=max(shape/self.dimensions,self.dimensions/shape)
                if ratio>PARAMS['maximum_dimension_change_factor']:
                    m=dict(m,valid=False,reason='frozen_layout_dimensions_disagree')
                else:
                    c=np.asarray(m['center']);self.correction=c+self.anchor-np.asarray(body['center'])
                    self.last_geometry=t;self.hubs=np.asarray(m['hubs']);self.layout_center=c;current=True
        elif not self.ready:self.pending.clear()
        age=None if self.last_geometry is None else t-self.last_geometry
        if self.ready and age is not None and age>PARAMS['layout_expiry_s']:
            self.ready=False;self.pending.clear();self.anchor=None;self.dimensions=None
        if current:
            weight=m['view_weight'];correction=self.correction*weight;mode='four_rotor_joint'
        elif self.ready and age is not None:
            decay=1.-smooth(age,PARAMS['correction_hold_s'],PARAMS['correction_expiry_s'])
            weight=decay;correction=self.correction*decay;mode='side_body_hold' if body_axis.get('valid') else 'partial_body_hold'
        else:weight=0.;correction=np.zeros(2);mode='body_fallback'
        out=copy.deepcopy(body)
        if detected:out['center']=(np.asarray(body['center'])+correction).tolist();out['source']='planar_center104'
        predicted_hubs=None
        if self.ready and self.hubs is not None and detected:
            predicted_hubs=(self.hubs-self.layout_center+np.asarray(out['center'])-self.anchor).tolist()
        diag=dict(t=t,ready=self.ready,current=current,initialized_now=initialized,initializations=self.initializations,
            mode=mode,weight=float(weight),correction_px=correction.tolist(),geometry_age_s=age,
            center=out.get('center'),model_geometric_center=m.get('center') if current else None,
            center_is_COM_truth=False,measurement=m,pending_frames=len(self.pending),
            virtual_hubs=predicted_hubs,virtual_hubs_are_observations=False,
            side_axis=body_axis.get('normal_axis_xy') if body_axis.get('valid') else None,
            processing_ms=(time.perf_counter()-start)*1000)
        return out,diag
