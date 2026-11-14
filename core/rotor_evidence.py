"""Headless current rotor evidence, independently of full-template lock.

Consumes already fitted native ON/OFF candidate arcs and current period tests.
Does not turn inferred hubs, missing rates, or a side line into measurements.
"""
import numpy as np
import cv2
from itertools import permutations
from scipy.optimize import linear_sum_assignment
from five_circle_model import diagonal_center,FiveCircleModel,ellipse_tensors


class RotorEvidence:
    def __init__(self):
        self.previous={};self.body=None;self.t=None;self.serial=0;self.previous_full_sum=None

    def step(self,t,body,structure,rotors,rates):
        if self.t is not None and t<=self.t:raise ValueError('Increasing current cutoffs required')
        center=np.asarray(body['center'],float) if body.get('detected') else None
        current=[]
        for index,r in enumerate(rotors):
            supported=bool(r.get('current_observation',True) and not r.get('fov_clipped')
                and not r.get('association_ambiguous') and r.get('association_state')!='relock_pending'
                and (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated'))
                and r.get('ratio',0)>.16)
            if not supported:continue
            # Duplicate fragments must not become additional motors.
            if any(np.hypot(r['cx']-a['ellipse']['cx'],r['cy']-a['ellipse']['cy'])<max(4.,.2*min(r['major'],a['ellipse']['major'])) for a in current):continue
            rate=rates[index] if index<len(rates) else {};f=rate.get('frequency_hz')
            measured=bool(rate.get('valid') and f is not None and np.isfinite(f) and 60<=f<=1200)
            current.append(dict(region_index=index,ellipse=r,center=[r['cx'],r['cy']],
                frequency_hz=float(f) if measured else None,period_measured=measured,
                center_is_prior_conditioned=bool(r.get('inferred_shape')),
                current_arc_supported=True,motor_identity_resolved=False))
        recent=self.t is not None and t-self.t<=.12+1e-8 and center is not None and self.body is not None
        ids=sorted(self.previous) if recent else [];assigned=set()
        if ids and current:
            delta=center-self.body
            distance=np.array([[np.linalg.norm(np.asarray(a['center'])-np.asarray(self.previous[k]['center'])-delta) for a in current] for k in ids])
            ii,jj=linear_sum_assignment(distance)
            for i,j in zip(ii,jj):
                gate=max(4.,.15*current[j]['ellipse']['major'])
                if distance[i,j]<=gate:current[j]['local_source_id']=ids[i];assigned.add(j)
        for j,a in enumerate(current):
            if j not in assigned:self.serial+=1;a['local_source_id']=self.serial
        n=len(current);axis=structure.get('axis_xy');normal=np.asarray(axis,float) if axis is not None else np.zeros(0)
        axis_current=bool(normal.shape==(2,) and np.isfinite(normal).all() and np.linalg.norm(normal)>1e-8 and
            (structure.get('pose_source')=='current_qualified_visible_conics' or (structure.get('axis_support') or {}).get('valid')
             or structure.get('mode') in ('four','three','partial_two')))
        if axis_current and normal[1]>0:normal=-normal
        components={};used=[]
        for a in current:
            old=self.previous.get(a['local_source_id']);f=a['frequency_hz']
            coherent=bool(f is not None and old is not None and old.get('frequency_hz') is not None and
                abs(np.log(f/old['frequency_hz']))<np.log(4/3)-.08)
            a['period_temporally_supported']=coherent
            if coherent and axis_current:
                components['region:'+str(a['local_source_id'])]=(f*f/360000.*normal).tolist();used.append(a)
        # A full sum is invariant to motor labels. Incomplete sums still retain
        # source keys so visibility changes can be transferred to unknown d.
        full=bool(n==4 and all(a['period_measured'] for a in current))
        full_temporal=False;sum_squared=None
        if full:
            sum_squared=float(sum(a['frequency_hz']**2 for a in current))
            old=self.previous_full_sum
            full_temporal=bool(old is not None and t-old[0]<=.12+1e-8 and abs(.5*np.log(sum_squared/old[1]))<np.log(4/3)-.08)
            self.previous_full_sum=(float(t),sum_squared)
        if n>4:components={};used=[];full=False
        if full_temporal and axis_current:
            components={'unordered_current_four':(sum_squared/360000.*normal).tolist()};used=list(current)
        joint=None
        if n==4 and len(cv2.convexHull(np.asarray([a['center'] for a in current],np.float32)))==4:
            joint=diagonal_center([a['center'] for a in current])
            if joint is not None and not np.isfinite(joint).all():joint=None
        measured_ids=set(structure.get('observed_model_ids',[]));inferred_ids=set(structure.get('inferred_model_ids',[]))-measured_ids
        self.previous={a['local_source_id']:a for a in current};self.body=center;self.t=float(t)
        return dict(t=float(t),schema='rotor-evidence.v1',current_regions=current,
            visible_rotor_hypothesis_count=n,period_region_count=sum(a['period_measured'] for a in current),
            current_contribution_count=len(used),current_components=components,
            complete_current_frequency_sum=full,full_template_required_for_visible_sum=False,
            current_four_frequency_squared_sum=sum_squared,full_sum_temporally_supported=full_temporal,
            template_mode=structure.get('mode'),template_confirmed=structure.get('mode')=='four',
            current_quad_center_candidate=None if joint is None else joint.tolist(),
            quad_center_is_COM_truth=False,quad_center_is_weak_perspective_proxy=True,
            inferred_hubs=[dict(model_id=i,measured=False,frequency_hz=None) for i in sorted(inferred_ids)],
            branch='four_regions' if n==4 else 'three_regions' if n==3 else 'two_regions' if n==2 else
                'side_axis' if axis_current and n<=1 else 'unresolved',
            axis_xy=normal.tolist() if axis_current else None,axis_unsigned=True,
            side_axis_does_not_determine_rotor_count=True,mechanical_RPM_measured=False,metric_thrust_measured=False,
            unknown_owns=['unmeasured_rotor_thrust','gravity','wind','drag','projection_model_error'])


class VisibilityLayout:
    """Small reusable geometry layer; no body NN position is accepted.

    A full current quad supplies a geometric proxy. Three current regions
    update a recent layout, using translation when supported and the existing
    affine shape test only when deformation requires it. Two points/side line
    do not silently become a fully measured plane or center.
    """
    def __init__(self):
        self.reference=None;self.reference_center=None;self.shapes=None;self.last_full=None
        self.last_t=None;self.last_center=None;self.velocity=np.zeros(2)

    def step(self,packet):
        t=packet['t']
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing cutoffs required')
        regions=packet['current_regions'];n=len(regions)
        prior=None if self.last_center is None else self.last_center+self.velocity*(t-self.last_t)
        result=dict(t=t,current_center=None,prior_center=None if prior is None else prior.tolist(),
            template_age_s=None if self.last_full is None else t-self.last_full,
            prior_usable=self.last_full is not None and t-self.last_full<=.4+1e-8,
            prior_is_current_measurement=False,
            current_center_is_COM_truth=False,current_center_is_prior_conditioned=False,
            measured_centers=[r['center'] for r in regions],inferred_hubs=[],
            axis_xy=packet['axis_xy'],axis_unsigned=True,branch=packet['branch'],
            model='side_axis_without_full_center' if packet['branch']=='side_axis' else
                'two_regions_without_full_center' if n==2 else 'unresolved',
            measured_rank=0,full_3d_pose=False,NN_body_position_used=False)
        center=None;hubs=None;ids=[]
        if n==4 and packet['current_quad_center_candidate'] is not None:
            hubs=np.asarray([r['center'] for r in regions],float);center=np.asarray(packet['current_quad_center_candidate'])
            self.reference=hubs.copy();self.reference_center=center.copy();self.last_full=t
            self.shapes=ellipse_tensors([r['ellipse'] for r in regions]);ids=list(range(4))
            result.update(model='current_four_quad',measured_rank=8)
        elif n==3 and self.reference is not None and t-self.last_full<=.4+1e-8:
            expected=self.reference+(prior-self.reference_center)
            p=np.asarray([r['center'] for r in regions],float)
            choices=sorted((float(np.mean(np.sum((p-expected[list(a)])**2,axis=1))),a) for a in permutations(range(4),3))
            gate=max(8.,.6*np.median([r['ellipse']['major'] for r in regions]))
            if np.sqrt(choices[0][0])<=gate and choices[1][0]-choices[0][0]>4.:
                ids=list(choices[0][1]);delta=np.mean(p-expected[ids],axis=0)
                residual=float(np.sqrt(np.mean(np.sum((p-expected[ids]-delta)**2,axis=1))))
                result['translation_residual_px']=residual
                if residual<=6.:
                    center=prior+delta;hubs=self.reference+(center-self.reference_center)
                    result.update(model='recent_layout_shared_translation',measured_rank=2,current_center_is_prior_conditioned=True)
                else:
                    adapter=FiveCircleModel(3);adapter.shapes=self.shapes
                    solved=adapter._three(p,[r['ellipse'] for r in regions],expected)
                    if solved is not None:
                        hubs,shapes,ids=solved;center=diagonal_center(hubs)
                        result.update(model='recent_layout_shape_checked_affine',measured_rank=6,current_center_is_prior_conditioned=True)
                if center is not None:
                    result['inferred_hubs']=[dict(model_id=i,center=hubs[i].tolist(),measured=False,frequency_hz=None) for i in sorted(set(range(4))-set(ids))]
        if center is not None:
            if self.last_center is not None and 0<t-self.last_t<=.12:
                self.velocity=(center-self.last_center)/(t-self.last_t)
            self.last_center=center.copy();result['current_center']=center.tolist()
            result['model_hubs']=hubs.tolist();result['observed_model_ids']=ids
        else:
            # The prior is explicitly not a current image measurement.
            if prior is not None:self.last_center=prior
            result['reason']='two_or_line_do_not_fully_observe_center' if n==2 or packet['branch']=='side_axis' else 'no_supported_recent_layout'
        self.last_t=t
        return result
