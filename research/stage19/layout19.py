"""Soft periodic augmentation and bounded measured four-point constellation matching."""
from pathlib import Path
import copy,importlib.util,itertools,sys
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

inc=module('stage18_frozen',ROOT/'research/stage18/frontend18.py')
identity=module('stage11_reused',ROOT/'research/stage11/identity.py')
RecoveryFrontend=inc.old.RecoveryFrontend
PaperPerception=inc.old.PaperPerception
counts=inc.old.counts
PARAMS=dict(step_us=20000,augmentation_weight=.5,periodic_activity_bonus=.25,full_memory_s=.30,
    full_confirm_frames=3,min_full_anisotropy=.015,max_diameter_ratio=2.5,match_gate_fraction=.22,
    match_gate_floor_px=16.,max_scale_step=1.20,max_rotation_step_deg=25.,min_pair_span_fraction=.35,
    residual_floor_px=6.,residual_fraction=.06)


class SoftGeometry(RecoveryFrontend):
    def __init__(self,w,h,augmentation=False,layout=False):
        super().__init__(w,h,'neighbor10');self.augmentation=augmentation;self.use_layout=layout
        self.support=np.zeros((h,w),np.float32);self.spectrum=None;self.raw_candidates=[]
        self.memory=None;self.seed=None;self.match_info={};self.pending_fit=None;self.last_proposal=None

    def feature(self,on,off):
        base=super().feature(on,off)
        return np.maximum(base,PARAMS['augmentation_weight']*(on+off)*self.support) if self.augmentation else base

    def _candidates(self,*args,**kwargs):
        found=PaperPerception._candidates(self,*args,**kwargs)
        for r in found:
            b=inc.PARAMS['block'];x0,y0,x1,y1=r['bbox'];h,w=self.spectrum['mask'].shape
            xx0,yy0=max(0,int(x0//b)),max(0,int(y0//b));xx1,yy1=min(w,int(np.ceil(x1/b))),min(h,int(np.ceil(y1/b)))
            mask=self.spectrum['mask'][yy0:yy1,xx0:xx1];coverage=float(mask.mean()) if mask.size else 0.
            if self.augmentation:r['activity']*=1+PARAMS['periodic_activity_bonus']*coverage
            row=dict(cx=float(r['center'][0]),cy=float(r['center'][1]),major=float(r['major']),minor=float(r['minor']),
                angle=float(r['angle']),mode=r['mode'],periodic_coverage=coverage,activity=float(r['activity']))
            self.candidates.append(row);self.raw_candidates.append(row)
        return found

    def _valid_four(self,rotors):
        if len(rotors)!=4:return False
        p=np.array([r['center'] for r in rotors]);d=np.array([r['major'] for r in rotors])
        if d.max()/max(1,d.min())>PARAMS['max_diameter_ratio']:return False
        if identity.anisotropy(p)<PARAMS['min_full_anisotropy']:return False
        # A nondegenerate convex4-point template only; no physical-square claim.
        hull=cv2.convexHull(p.astype(np.float32));return len(hull)==4

    def _fit_memory(self,candidates,t):
        mem=self.memory
        if mem is None or t-mem['full_t']>PARAMS['full_memory_s'] or not candidates:return None
        points=mem['points'];predicted=points+mem['velocity']*(t-mem['t'])
        positions=np.array([r['center'] for r in candidates]);distance=np.linalg.norm(predicted[:,None,:]-positions[None,:,:],axis=2)
        extent=max(np.ptp(points,axis=0));gate=max(PARAMS['match_gate_floor_px'],extent*PARAMS['match_gate_fraction'])
        rows,cols=linear_sum_assignment(distance)
        pairs=[(int(i),int(j)) for i,j in zip(rows,cols) if distance[i,j]<=gate and
            1/PARAMS['max_diameter_ratio']<=candidates[j]['major']/max(1.,mem['rotors'][i]['major'])<=PARAMS['max_diameter_ratio']]
        if len(pairs)<2:return None
        source=np.array([points[i] for i,j in pairs]);target=np.array([positions[j] for i,j in pairs])
        if max(np.ptp(source,axis=0))<PARAMS['min_pair_span_fraction']*extent:return None
        transform=identity.RigidTracker._fit(None,source,target)
        scale=float(np.linalg.norm(transform[:2,0]));rotation=float(np.rad2deg(np.arctan2(transform[1,0],transform[0,0])))
        residual=float(np.sqrt(np.mean(np.sum((source@transform[:2,:2].T+transform[:2,2]-target)**2,axis=1))))
        spacing=float(np.median([np.linalg.norm(a-b) for a,b in itertools.combinations(points,2)]))
        if not 1/PARAMS['max_scale_step']<=scale<=PARAMS['max_scale_step']:return None
        if abs(rotation)>PARAMS['max_rotation_step_deg']:return None
        if residual>max(PARAMS['residual_floor_px'],spacing*PARAMS['residual_fraction']):return None
        newpoints=points@transform[:2,:2].T+transform[:2,2]
        selected=[candidates[j] for i,j in pairs]
        return dict(selected=selected,points=newpoints,transform=transform,scale=scale,residual=residual,pairs=pairs,
                    rotation_deg=rotation,matched=len(pairs),full_age_s=t-mem['full_t'])

    def _group(self,candidates,t):
        selected=super()._group(candidates,t)
        if not self.use_layout:return selected
        if self.group_info.get('transferred'):self.memory=None;self.seed=None
        eligible=candidates
        if self.reference is not None:
            diam=self.reference['diameter'];eligible=[r for r in candidates if diam/2.5<=r['major']<=diam*2.5]
        fit=self._fit_memory(eligible,t)
        self.match_info.update(memory_active=self.memory is not None and t-self.memory['full_t']<=PARAMS['full_memory_s'],
            memory_age_s=None if self.memory is None else t-self.memory['full_t'],matched=False)
        if fit is None:return selected
        self.pending_fit=fit;self.challenge=None
        self.match_info.update(matched=True,matched_count=fit['matched'],residual_px=fit['residual'],
            scale=fit['scale'],rotation_deg=fit['rotation_deg'],full_age_s=fit['full_age_s'])
        self.group_info.update(selected_count=len(fit['selected']),layout_matched=True)
        # The existing bounded partial-layout reconstruction executes the transform.
        self.constellation=copy.deepcopy(self.memory['rotors']);self.constellation_t=self.memory['t']
        self.velocity=self.memory['velocity'].copy()
        return fit['selected']

    def _remember(self,rotors,t):
        points=np.array([r['center'] for r in rotors]);before=self.memory
        velocity=np.zeros(2)
        if before is not None and 0<t-before['t']<=PARAMS['full_memory_s']:
            velocity=.8*before['velocity']+.2*np.clip((points.mean(0)-before['points'].mean(0))/(t-before['t']),-2500,2500)
        self.memory=dict(points=points,rotors=copy.deepcopy(rotors),t=t,full_t=t,velocity=velocity)

    def detect_prepared(self,events,t,prepared):
        self.pending_fit=None;self.match_info={};self.raw_candidates=[]
        observation,total=super().detect_prepared(events,t,prepared);self.last_proposal=observation
        if not self.use_layout:return observation,total
        measured=self.previous_rotors if observation['detected'] else []
        if self._valid_four(measured):
            center=np.mean([r['center'] for r in measured],axis=0)
            measured=sorted(measured,key=lambda r:np.arctan2(*(r['center']-center)[::-1]))
            points=np.array([r['center'] for r in measured]);count=1
            if self.seed is not None and 0<t-self.seed['t']<=.025:
                oldpoints=self.seed['points'];best=identity.best_assignment(oldpoints,points,orientation_only=True)
                if best and best[0]<=max(6.,np.ptp(points,axis=0).max()*.06):count=self.seed['count']+1
            self.seed=dict(points=points,t=t,count=count)
            if count>=PARAMS['full_confirm_frames']:
                self._remember(measured,t);self.match_info['full_template_refreshed']=True
        else:
            self.seed=None
            if observation['detected'] and self.pending_fit is not None and self.memory is not None:
                fit=self.pending_fit;previous=self.memory;transform=fit['transform'];scale=fit['scale'];rotors=[]
                for r,p in zip(previous['rotors'],fit['points']):
                    half=(r['bbox'][2:]-r['bbox'][:2])*scale/2
                    rotors.append(dict(r,center=p,bbox=np.r_[p-half,p+half],major=r['major']*scale))
                dt=max(t-previous['t'],.001)
                velocity=.8*previous['velocity']+.2*np.clip((fit['points'].mean(0)-previous['points'].mean(0))/dt,-2500,2500)
                self.memory=dict(points=fit['points'],rotors=rotors,t=t,full_t=previous['full_t'],velocity=velocity)
        if self.memory is not None and t-self.memory['full_t']>PARAMS['full_memory_s']:self.memory=None
        self.match_info['template_available_after']=self.memory is not None
        return observation,total


class Observer:
    def __init__(self,w,h,augmentation=False,layout=False):
        self.geometry=SoftGeometry(w,h,augmentation,layout);self.w=w;self.h=h

    def step(self,events,t,prepared,spectrum):
        g=self.geometry;mask=cv2.dilate(np.uint8(spectrum['mask']),np.ones((3,3),np.uint8))
        g.support=np.repeat(np.repeat(mask,8,0),8,1)[:self.h,:self.w].astype(np.float32);g.spectrum=spectrum
        observation,_=g.detect_prepared(events,t,prepared)
        return dict(observation=observation,candidates=g.candidates,audit=g.audit,layout=g.match_info)
