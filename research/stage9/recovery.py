"""Protect whole-UAV state and confirm new hypotheses; no image reconstruction.

Reuses the frozen stage8 frontend, old grouping and partial constellation fit.
This is a single-target geometry adapter, not a ByteTrack/DeepSORT execution.
"""
import copy
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage8'))
from polarity import PolarityFrontend

PARAMS=dict(diameter_factor=2.5,extent_factor=2.,reference_expiry_s=.5,
            partial_age_s=.3,reference_frames=3,reacquire_after_s=.08,
            confirm_frames=3,pending_gap_s=.025,center_fraction=.25,
            reference_alpha=.1)
GEOMETRY_STATE=('center','velocity','last_t','last_seen','last_extent','normal',
    'constellation','constellation_t','pending_center','pending_streak',
    'previous_rotors','previous_rotors_t','layout_measured_t','freqs',
    'frequency_quality','frequency_reference','rotor_frequency_quality','frequency_diagnostics')


def shape(rotors):
    boxes=np.array([r['bbox'] for r in rotors]);lo=boxes[:,:2].min(0);hi=boxes[:,2:].max(0)
    sizes=np.array([r['major'] for r in rotors])
    return dict(center=(lo+hi)/2,extent=float(max(hi-lo)),diameter=float(np.median(sizes)),
                diameter_spread=float(sizes.max()/max(sizes.min(),1.)))


class RecoveryFrontend(PolarityFrontend):
    def __init__(self,w,h,method,allow_recovery=True):
        super().__init__(w,h,method)
        self.allow_recovery=allow_recovery
        self.reference=None;self.reference_pending=None;self.challenge=None
        self.audit={};self.group_info={}

    def _advance(self,previous,measurement,t):
        if previous is not None:
            gap=t-previous['t']
            close=np.linalg.norm(measurement['center']-previous['center'])<max(20.,measurement['extent']*PARAMS['center_fraction'])
            scale=max(measurement['extent'],previous['extent'])/max(1.,min(measurement['extent'],previous['extent']))
            count=previous['count']+1 if 0<gap<=PARAMS['pending_gap_s'] and close and scale<PARAMS['extent_factor'] else 1
        else:count=1
        return dict(measurement,t=t,count=count)

    def _reset_motion(self):
        # Unlike clearing every lost frame, this runs only on confirmed transfer.
        self.center=None;self.last_t=None;self.last_seen=None;self.velocity=np.zeros(2)
        self.last_extent=120.
        self.constellation=None;self.constellation_t=None
        self.previous_rotors=[];self.previous_rotors_t=None;self.layout_measured_t=None
        self.normal=None;self.pending_center=None;self.pending_streak=0
        self.freqs=[];self.frequency_quality=0.;self.frequency_reference=None
        self.rotor_frequency_quality=[];self.frequency_diagnostics=[];self.history.clear()

    def _group(self,candidates,t):
        ref=self.reference
        kept=candidates
        if ref is not None:
            factor=PARAMS['diameter_factor'];diam=ref['diameter']
            kept=[r for r in candidates if diam/factor<=r['major']<=diam*factor]
        self.group_info=dict(original_candidates=len(candidates),scale_eligible=len(kept),
            reference_extent=None if ref is None else ref['extent'],
            reference_diameter=None if ref is None else ref['diameter'],
            transferred=False,selected_count=0)
        selected=super()._group(kept,t)
        if selected is not None:
            self.challenge=None;self.group_info['selected_count']=len(selected)
            return selected
        gap=float('inf') if self.last_seen is None else t-self.last_seen
        if not self.allow_recovery or self.center is None or gap<PARAMS['reacquire_after_s']:
            self.challenge=None
            return None
        saved=self.center
        try:
            self.center=None
            proposed=super()._group(kept,t)
        finally:self.center=saved
        # Two-point/single-point evidence alone cannot start a new hypothesis.
        if proposed is None or len(proposed)<3:
            self.challenge=None
            return None
        measurement=shape(proposed)
        if measurement['diameter_spread']>PARAMS['diameter_factor']:
            self.challenge=None
            return None
        if ref is not None and not ref['extent']/PARAMS['extent_factor']<=measurement['extent']<=ref['extent']*PARAMS['extent_factor']:
            self.challenge=None
            return None
        self.challenge=self._advance(self.challenge,measurement,t)
        self.group_info['challenge_count']=self.challenge['count']
        if self.challenge['count']<PARAMS['confirm_frames']:return None
        self._reset_motion();self.reference=None;self.reference_pending=None;self.challenge=None
        self.group_info.update(transferred=True,selected_count=len(proposed))
        return proposed

    def detect_prepared(self,events,t,prepared):
        if self.reference is not None and t-self.reference['t']>PARAMS['reference_expiry_s']:
            self.reference=None;self.reference_pending=None
        if self.last_seen is not None and t-self.last_seen>self.cfg.reset_gap_s:
            self.challenge=None;self.reference_pending=None
        before={k:copy.deepcopy(getattr(self,k)) for k in GEOMETRY_STATE}
        self.group_info={}
        result,total=super().detect_prepared(events,t,prepared)
        reject=None
        ref=self.reference
        if result['detected'] and ref is not None:
            # The old detector may infer a complete box from partial rotors.
            # Even then a collapsed box cannot rewrite the protected whole state.
            if not ref['extent']/PARAMS['extent_factor']<=self.last_extent<=ref['extent']*PARAMS['extent_factor']:
                reject='whole_extent_inconsistent'
        if reject:
            for k,value in before.items():setattr(self,k,value)
            self.reference_pending=None
            result=dict(t=t,detected=False,confidence=0.,rotors=[],n_events=len(events))
        elif result['detected']:
            measured=self.previous_rotors
            if len(measured)<4 and not self.group_info.get('transferred'):
                # Inferred hidden motors must not refresh measured layout age.
                self.constellation=before['constellation']
                self.constellation_t=before['constellation_t']
                self.layout_measured_t=before['layout_measured_t']
            if len(measured)==4:
                measurement=shape(measured)
                if measurement['diameter_spread']<=PARAMS['diameter_factor']:
                    self.reference_pending=self._advance(self.reference_pending,measurement,t)
                    if self.reference_pending['count']>=PARAMS['reference_frames']:
                        if self.reference is None:self.reference=dict(measurement,t=t)
                        else:
                            a=PARAMS['reference_alpha']
                            for k in ['extent','diameter']:self.reference[k]=(1-a)*self.reference[k]+a*measurement[k]
                            self.reference['center']=measurement['center'];self.reference['t']=t
                else:self.reference_pending=None
            else:self.reference_pending=None
        else:self.reference_pending=None
        self.audit=dict(self.group_info,state_rejection=reject,
            reference_active=self.reference is not None,
            reference_extent_after=None if self.reference is None else self.reference['extent'],
            detected=result['detected'],whole_extent_after=self.last_extent,
            status='measured' if result['detected'] else 'unconfirmed' if self.challenge else 'lost',
            recovery_enabled=self.allow_recovery)
        return result,total


def tests():
    def rotor(x,y,d=30):
        return dict(center=np.array([x,y],float),major=d,minor=8.,bbox=np.array([x-d/2,y-4,x+d/2,y+4]),
                    mode='ellipse',activity=100.,shape_quality=.8)
    f=RecoveryFrontend(640,480,'neighbor10')
    f.reference=dict(diameter=30.,extent=130.,center=np.array([150.,150.]),t=1.)
    f.center=np.array([150.,150.]);f.velocity[:]=0;f.last_t=f.last_seen=1.;f.last_extent=130.
    fragments=[rotor(150,150,8),rotor(160,150,8),rotor(155,160,8)]
    assert f._group(fragments,1.01) is None
    assert f.group_info['scale_eligible']==0
    f.last_extent=30.;f.reference['extent']=130.
    far=[rotor(400,200),rotor(500,200),rotor(500,250),rotor(400,250)]
    assert f._group(far,1.10) is None and f.challenge['count']==1
    assert f._group(far,1.11) is None and f.challenge['count']==2
    assert f._group(far,1.12) is not None and f.group_info['transferred']
    assert f.center is None and not f.previous_rotors
    f=RecoveryFrontend(640,480,'strict10');p=None
    m=dict(center=np.array([100.,100.]),extent=100.,diameter=30.)
    p=f._advance(p,m,1.);p=f._advance(p,m,1.01);assert p['count']==2
    p=f._advance(p,m,1.1);assert p['count']==1
    # Exercise measured-four versus inferred-four ages through the real detector.
    import cv2
    from polarity import counts
    def events_for(centers,t):
        mask=np.zeros((240,320),np.uint8)
        for center in centers:cv2.ellipse(mask,center,(22,7),12,0,360,1,-1)
        yy,xx=np.nonzero(mask)
        events=np.zeros(len(xx)*6,dtype=[('x','u2'),('y','u2'),('p','i2'),('t','i8')])
        events['x']=np.repeat(xx,6);events['y']=np.repeat(yy,6)
        events['p']=np.tile([0,1,0,1,0,1],len(xx));events['t']=round(t*1e6)-100
        return events
    f=RecoveryFrontend(320,240,'strict10');centers=[(90,70),(230,70),(90,170),(230,170)]
    for k in range(1,6):
        t=k*.01;e=events_for(centers,t);o,_=f.detect_prepared(e,t,counts(e,320,240));assert o['detected']
    assert f.reference is not None
    measured_age=f.constellation_t;ref_age=f.reference['t']
    e=events_for(centers[:3],.06);o,_=f.detect_prepared(e,.06,counts(e,320,240))
    assert o['detected'] and o['observed_rotor_count']==3
    assert f.constellation_t==measured_age and f.reference['t']==ref_age
    return dict(fragments_filtered=True,recovery_requires_three_consecutive_frames=True,
        missing_confirmation_resets=True,transfer_resets_motion_once=True,
        inferred_layout_does_not_refresh_age=True,no_extrapolated_measurements=True)
