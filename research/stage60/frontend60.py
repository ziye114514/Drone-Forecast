"""Experimental paired-line observations in the original candidate/grouping path."""
from pathlib import Path
import sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage57'))
from gating57 import CandidateFrontend
import pairs60 as pairs
import profile60 as profile
PARAMS=dict(line_modes=['off','paired'],ellipse_ratio_max=.30,new_candidate_max=8,
            weight_smoothing_alpha=.3,max_fusion_weight=.7,
            source='59 illustrative ratio gate plus standard quality-weighted convex observation combination, no independent-information double counting. Geometry candidates do not certify real motor identity.')

class LineFrontend(CandidateFrontend):
    def __init__(self,w,h,method,line_mode='paired'):
        super().__init__(w,h,method);assert line_mode in PARAMS['line_modes'];self.line_mode=line_mode
        self.confirm=pairs.Confirm();self.weights={};self.line_audit={}
        self.lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    def extract(self,t):
        audit=dict(mode=self.line_mode,edge_lines=0,base_pairs=0,stable_pairs=0,confirmed_pairs=0,
            profile_usable=0,profile_rejections={},inserted=0,fused=0,periodic_identity_independently_verified=False)
        if self.line_mode=='off':return [],audit
        if self.candidate_roi is None:
            self.confirm.step(t,[],None);audit['reason']='no_past_body_roi';return [],audit
        prepared=self.prepared;feature=self.feature(prepared['on'],prepared['off']);feature[prepared['on']+prepared['off']<self.cfg.min_pixel_count]=0
        b=self.cfg.spatial_bin;assert b==1
        density=cv2.GaussianBlur(feature,(3,3),.8);positive=density[density>0]
        threshold=max(2.,float(np.quantile(positive,self.cfg.density_quantile))) if len(positive) else 2.
        variants=[]
        for multiplier in [1.,1.5]:
            gray=np.uint8(np.clip(density/(2*threshold),0,1)*255);gray[density<threshold*multiplier]=0
            lines=self.lsd.detect(gray)[0];found=[];roi=self.candidate_roi
            for p in ([] if lines is None else lines.reshape(-1,4)):
                length=np.linalg.norm(p[2:]-p[:2]);c=(p[:2]+p[2:])/2
                if 12<=length<=self.w*.5 and roi[0]<=c[0]<roi[2] and roi[1]<=c[1]<roi[3]:found.append(p.reshape(2,2))
            variants.append(pairs.pair_segments(found))
            if multiplier==1.:audit['edge_lines']=len(found)
        stable=pairs.cross_threshold(*variants);tracked=self.confirm.step(t,stable,self.candidate_roi)
        audit.update(base_pairs=len(variants[0]),stable_pairs=sum(p['cross_threshold_stable'] for p in stable),confirmed_pairs=sum(p['confirmed_candidate'] for p in tracked))
        proposals=[]
        for p in tracked:
            if not p['confirmed_candidate']:continue
            checked=profile.check(density,threshold,p)
            if not checked['usable']:
                reason=checked['reason'];audit['profile_rejections'][reason]=audit['profile_rejections'].get(reason,0)+1;continue
            audit['profile_usable']+=1;c=np.array(checked['center']);major=checked['major'];minor=p['minor'];angle=np.deg2rad(p['angle']);u=np.array([np.cos(angle),np.sin(angle)]);n=np.array([-u[1],u[0]])
            corners=c[None,:]+np.array([[-major/2,-minor/2],[-major/2,minor/2],[major/2,-minor/2],[major/2,minor/2]])@np.array([u,n])
            lo=np.maximum(np.floor(corners.min(0)).astype(int),0);hi=np.minimum(np.ceil(corners.max(0)).astype(int)+1,[self.w,self.h])
            activity=float(feature[lo[1]:hi[1],lo[0]:hi[0]].sum());q=float(p['shape_quality']*(1-checked['mirror_difference'])*np.exp(-p['threshold_center_shift_px']/max(3,major*.1)))
            sigma=max(3.,major*.1)
            proposals.append(dict(center=c,major=major,minor=minor,angle=p['angle'],normal=n,
                ratio=minor/max(major,1),bbox=np.r_[lo,hi].astype(float),activity=activity,
                shape_quality=max(.2,q),geometry_quality=min(.6,q),mode='line',visible_arc_fraction=None,
                support_fraction=checked['profile_support'],fit_residual_px=None,center_sigma_px=sigma,
                axis_angle_sigma_deg=5.,inferred_shape=False,axis_only=True,
                center_covariance_px2=(np.outer(u,u)*sigma**2+np.outer(n,n)*max(2,minor/2)**2).tolist(),
                line_candidate_track_id=p['candidate_track_id'],line_quality=q,physical_endpoints_verified=False,
                verified_rotor_identity=False,paired_profile=checked))
        return sorted(proposals,key=lambda p:p['line_quality'],reverse=True)[:PARAMS['new_candidate_max']],audit

    def _candidates(self,*args,**kwargs):
        found=super()._candidates(*args,**kwargs)
        thin=kwargs.get('thin_only',False)
        if thin:return found
        t=args[2];proposals,audit=self.extract(t);result=list(found)
        for p in proposals:
            distances=[np.linalg.norm(p['center']-q['center']) for q in result]
            j=int(np.argmin(distances)) if distances else None
            if j is not None and distances[j]<max(8.,result[j]['major']*.3):
                old=result[j]
                if old['ratio']>=PARAMS['ellipse_ratio_max']:continue
                side=float(np.clip((.30-old['ratio'])/(.30-.08),0,1));a=side*p['line_quality'];b=(1-side)*max(.05,old['geometry_quality'])
                target=min(PARAMS['max_fusion_weight'],a/(a+b)) if a+b>0 else 0.
                key=p['line_candidate_track_id'];last=self.weights.get(key,target);weight=.7*last+.3*target;self.weights[key]=weight
                new=dict(old);new['center']=(1-weight)*old['center']+weight*p['center']
                z=(1-weight)*np.exp(2j*np.deg2rad(old['angle']))+weight*np.exp(2j*np.deg2rad(p['angle']))
                new['angle']=float(np.degrees(np.angle(z)/2)%180);angle=np.deg2rad(new['angle'])
                new['normal']=np.array([-np.sin(angle),np.cos(angle)])*np.sqrt(max(0.,1-old['ratio']**2))
                # Convex same-window combination, with conservative disagreement inflation.
                new['center_sigma_px']=max(old['center_sigma_px'],p['center_sigma_px'],float(distances[j]))
                new['line_fusion']=dict(weight=weight,candidate_track_id=key,endpoint_semantics='bounded activity support, physical tip identity unverified')
                result[j]=new;audit['fused']+=1
            else:result.append(p);audit['inserted']+=1
        self.line_audit=audit
        return result
