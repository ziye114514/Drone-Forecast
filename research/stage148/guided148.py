"""A recent measured quad guides reassignment of CURRENT original contour arcs.

No prior ellipse is emitted without two separately supported current arc fits.
The independent147 concavity fit is tried first; template guidance is fallback.
"""
from pathlib import Path
from itertools import combinations
import copy,sys
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage147')]
import curvature147 as curve
from event_drone_forecast.geometry import _distance,_coverage

PARAMS=dict(max_prior_age_s=.12,min_prior_observations=3,arc_distance_px=2.5,
            assignment_distance_margin_px=.7,prior_center_gate_major=.15,
            minimum_current_arc_coverage=.48,no_prior_only_output=True,
            maximum_prior_major_ratio=1.6,minimum_fragment_union_iou=.72,
            fragment_source='110/124 actual pre-closing open components; same provenance as candidates')

def parameters(r):return np.array([r['cx'],r['cy'],r['major']/2,r['minor']/2,np.deg2rad(r['angle'])])

class QuadGuide:
    def __init__(self):self.quad=None;self.t=None;self.body=None
    def update(self,t,body,geometry):
        rs=geometry['observed_rotors']
        if geometry['state']!='four' or not body.get('detected') or len(rs)!=4:return
        if any(r.get('temporal_observations',0)<PARAMS['min_prior_observations'] or r.get('fov_clipped') for r in rs):return
        if max(r['major'] for r in rs)/min(r['major'] for r in rs)>PARAMS['maximum_prior_major_ratio']:return
        self.quad=copy.deepcopy(rs);self.t=t;self.body=np.asarray(body['center'],float)
    def split(self,contour,t,body):
        diag=dict(attempted=False,accepted=False,reason='no_recent_measured_quad')
        if self.quad is None or not body.get('detected') or t-self.t>PARAMS['max_prior_age_s']+1e-9:return None,diag
        p=np.asarray(contour,float).reshape(-1,2);extent=max(np.ptp(p,axis=0));lo=p.min(0);hi=p.max(0)
        shift=np.asarray(body['center'])-self.body
        seeds=[parameters(r)+np.r_[shift,0.,0.,0.] for r in self.quad]
        eligible=[i for i,s in enumerate(seeds) if np.all(s[:2]>=lo-3) and np.all(s[:2]<=hi+3)]
        diag.update(attempted=len(eligible)>=2,prior_age_s=t-self.t,eligible_seed_count=len(eligible))
        if len(eligible)<2:return None,diag
        for i,j in combinations(eligible,2):
            distance=np.array([abs(_distance(p,seeds[k])) for k in [i,j]])
            assignments=[(distance[k]<=PARAMS['arc_distance_px'])&(distance[1-k]-distance[k]>=PARAMS['assignment_distance_margin_px']) for k in (0,1)]
            if any(np.sum(m)<22 for m in assignments):continue
            if any(_coverage(p[m],seeds[k])<PARAMS['minimum_current_arc_coverage'] for m,k in zip(assignments,[i,j])):continue
            arcs=[curve.fit_arc(p[m],extent) for m in assignments]
            if any(a is None for a in arcs):continue
            compatible=True
            for arc,k in zip(arcs,[i,j]):
                fitted=arc['parameters'];seed=seeds[k]
                if np.linalg.norm(fitted[:2]-seed[:2])>max(4.,2*seed[2]*PARAMS['prior_center_gate_major']):compatible=False
                if not (.75<fitted[2]/seed[2]<1.3 and .6<fitted[3]/seed[3]<1.5):compatible=False
            if not compatible:continue
            evidence=curve.pair_evidence(contour,arcs)
            if evidence is None:continue
            diag.update(accepted=True,reason='recent_quad_guides_two_current_supported_arcs',evidence=evidence,
                        prior_visual_ids=[self.quad[k]['visual_track_id'] for k in [i,j]],
                        center_sigma_px=[a['sigma'] for a in arcs],coverages=[a['coverage'] for a in arcs],
                        rms=[a['rms'] for a in arcs],no_prior_only_output=True,no_cut_chord=True,
                        centers=[a['parameters'][:2].tolist() for a in arcs])
            return arcs,diag
        diag['reason']='current_arcs_do_not_support_prior_pair';return None,diag

class GuidedFrontend(curve.CurvatureFrontend):
    def __init__(self,w,h,threshold=4.,adaptive=False):
        super().__init__(w,h,threshold,adaptive);self.guide=QuadGuide();self.guide_body=dict(detected=False)
        self.fragment_audit=[]
    def set_hint(self,events,body,t):
        self.guide_body=body;return super().set_hint(events,body,t)
    def _candidates(self,contours,activity,t,b,thin_only=False):
        original=curve.split_contour
        def split(contour):
            arcs,diag=original(contour)
            if arcs is not None:
                diag['method148']='independent_concavity';return arcs,diag
            recovered,info=self.guide.split(contour,t,self.guide_body)
            diag['guidance148']=info
            if recovered is not None:
                diag.update(info,method148='prior_guided_current_arcs',cuts=[])
            return recovered,diag
        curve.split_contour=split
        try:
            component_contours=self.contours_before_merge(contours,activity,b) if not thin_only else contours
            candidates=super()._candidates(contours,activity,t,b,thin_only)
            return candidates if thin_only else self.merge_fragments(candidates,component_contours,activity,t,b)
        finally:curve.split_contour=original

    def merge_fragments(self,candidates,contours,activity,t,b):
        guide=self.guide
        if guide.quad is None or not self.guide_body.get('detected') or t-guide.t>.12+1e-9:return candidates
        shift=np.asarray(self.guide_body['center'])-guide.body
        # A contour is a connected mask fragment, not necessarily a rotor.
        # Only real external contour points are collected; no cut chord added.
        components=[np.asarray(c,float).reshape(-1,2)*b for c in contours if cv2.contourArea(c)*b*b>=12 and len(c)>=10]
        work=list(candidates)
        for seed_r in guide.quad:
            seed=parameters(seed_r)+np.r_[shift,0.,0.,0.]
            members=[]
            for i,r in enumerate(work):
                if r.get('curvature_validated') or r.get('fragment_merged148'):continue
                d=np.array(r['center'])-seed[:2];c,s=np.cos(seed[4]),np.sin(seed[4]);u=d[0]*c+d[1]*s;v=-d[0]*s+d[1]*c
                if (u/max(seed[2],1))**2+(v/max(seed[3],1))**2<.85 and r['major']<.85*2*seed[2]:members.append(i)
            if len(members)<2:continue
            audit=dict(seed=seed.tolist(),members=[work[i]['center'].tolist() for i in members],stage='members');self.fragment_audit.append(audit)
            sets=[]
            for points in components:
                keep=abs(_distance(points,seed))<=2.5
                if keep.sum()>=8:sets.append(points[keep])
            audit['current_components']=len(sets)
            if len(sets)<2:continue
            points=np.concatenate(sets)
            audit['prior_coverage']=_coverage(points,seed)
            if _coverage(points,seed)<.60:continue
            arc=curve.fit_arc(points,seed[2]*2)
            if arc is None:continue
            audit['stage']='arc_fitted';audit['fit']=arc['parameters'].tolist()
            p=arc['parameters']
            if np.linalg.norm(p[:2]-seed[:2])>max(4.,.15*2*seed[2]):continue
            if not(.75<p[2]/seed[2]<1.3 and .6<p[3]/seed[3]<1.5):continue
            # Sum of fragment supports should explain a filled projected disk;
            # this rejects unrelated small blobs arranged around a prior rim.
            lower=np.floor(p[:2]-p[2]-4).astype(int);upper=np.ceil(p[:2]+p[2]+5).astype(int)
            shape=tuple((upper-lower)[::-1]);actual=np.zeros(shape,np.uint8);ellipse=np.zeros_like(actual)
            member_points=[]
            for i in members:
                r=work[i];center=np.asarray(r['center']);nearest=min(components,key=lambda c:np.linalg.norm(c.mean(0)-center))
                member_points.append(nearest);cv2.fillPoly(actual,[np.rint(nearest-lower).astype(np.int32)],1)
            cv2.ellipse(ellipse,tuple(np.rint(p[:2]-lower).astype(int)),tuple(np.rint(p[2:4]).astype(int)),float(np.rad2deg(p[4])),0,360,1,-1)
            iou=float(np.sum((actual>0)&(ellipse>0))/max(1,np.sum((actual>0)|(ellipse>0))))
            audit['union_iou']=iou
            if iou<PARAMS['minimum_fragment_union_iou']:continue
            ratio=p[3]/p[2];quality=float(np.exp(-arc['rms']/2)*min(1.,arc['coverage']/.65));lo=points.min(0);hi=points.max(0)
            merged=dict(center=p[:2],major=2*p[2],minor=2*p[3],ratio=ratio,angle=float(np.rad2deg(p[4])),
                normal=np.array([-np.sin(p[4]),np.cos(p[4])])*np.sqrt(1-ratio**2),bbox=np.r_[lo,hi],
                activity=sum(work[i]['activity'] for i in members),shape_quality=quality,geometry_quality=quality*.8,
                mode='arc',visible_arc_fraction=arc['coverage'],support_fraction=arc['support_fraction'],
                fit_residual_px=arc['rms'],center_sigma_px=arc['sigma'],axis_angle_sigma_deg=arc['angle_sigma'],
                inferred_shape=True,axis_only=False,curvature_validated=True,curvature_parent=-1,no_cut_chord=True,
                original_arc_points=arc['points'][::max(1,len(arc['points'])//40)].tolist(),fragment_merged148=True,
                fragment_source_count=len(members),fragment_union_iou=iou)
            work=[r for i,r in enumerate(work) if i not in members]+[merged]
            audit['stage']='accepted'
        return work

    def _group(self,candidates,t):
        merged=[r for r in candidates if r.get('fragment_merged148')]
        for r in merged:r['curvature_validated']=False
        try:return super()._group(candidates,t)
        finally:
            for r in merged:r['curvature_validated']=True
    def detect_prepared(self,*args,**kwargs):
        self.fragment_audit=[]
        observation,total=super().detect_prepared(*args,**kwargs)
        for rotor,measured in zip(observation.get('rotors',[]),self.previous_rotors if observation.get('detected') else []):
            if rotor.get('curvature_validated'):
                if measured.get('fragment_merged148'):
                    rotor.update(split_method148='prior_guided_fragment_merge',fragment_source_count=measured['fragment_source_count'],fragment_union_iou=measured['fragment_union_iou'])
                else:
                    d=self.accepted147[rotor['curvature_parent']]
                    rotor['split_method148']=d.get('method148','independent_concavity')
                rotor['current_arc_support_required']=True
        return observation,total
