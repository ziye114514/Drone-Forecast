"""Track-before-birth identity evidence; no new pixel frequency or thrust.

A tentative current bipolar component is not yet a drone track. It can confirm
identity only after a two-way current image match and12 distinct strict votes
over two windows. Current and past event streams remain entirely unwarped.
"""
import cv2
import numpy as np
from native_history_buffer import NativeHistoryBuffer
from rotor_only_front import native_reconstruction
from rotor_support_front import estimate_region_translation
from native_cycle_consistency import qualify
from observation_roles import RoleSeparatedContinuation


def distinct_votes(old_xy,current_xy,shift,current_support):
    old=np.rint(np.asarray(old_xy,float)+np.asarray(shift,float)).astype(int);h,w=current_support.shape
    inside=(old[:,0]>=0)&(old[:,0]<w)&(old[:,1]>=0)&(old[:,1]<h);old=old[inside]
    if len(old):old=old[current_support[old[:,1],old[:,0]]]
    all_xy=np.vstack((old,np.asarray(current_xy,int)))
    return int(len(np.unique(all_xy[:,1]*w+all_xy[:,0])))


class TentativeRotorBirth:
    def __init__(self):self.history=NativeHistoryBuffer();self.previous=None;self.mask=None;self.xy=None;self.t=None
    def consider(self,frame,front,t):
        native,_=native_reconstruction(frame);strict=(frame['period_mask']>0)&native
        yy,xx=np.nonzero(strict);xy=np.column_stack((xx,yy));count=len(xy)
        info=dict(used=False,current_strict_pixels_on_component=count,frequency_created=False,known_force_created=False)
        if front.center is not None:
            self.t=None;self.previous=None;self.mask=None;return dict(info,reason='existing_track_uses_existing_continuation')
        if count and native.sum()>=12:
            if count>=4 and self.previous is not None and 0<t-self.t<=.1+1e-8:
                bx,by,bw,bh=cv2.boundingRect(native.view(np.uint8));ox,oy,ow,oh=cv2.boundingRect(self.mask.view(np.uint8))
                if max(bw,bh)>=.5*max(ow,oh):
                    proposal=np.array([bx+bw/2-ox-ow/2,by+bh/2-oy-oh/2])
                    if np.max(abs(proposal))<=160:
                        opts=dict(native_matching=front.fast_geometry,subpixel=front.subpixel_motion,max_shift=40,
                            coarse_matching=front.coarse_motion,interior_border=front.border_motion,centered_matching=front.centered_motion)
                        current=(frame['on'],frame['off'])
                        f=estimate_region_translation(self.previous,self.mask,current,current_support=native,predicted_shift=proposal,**opts)
                        b=estimate_region_translation(current,native,self.previous,current_support=self.mask,predicted_shift=-proposal,**opts)
                        checked=qualify(f,b) if f is not None else None
                        if checked is not None and checked.get('current_displacement_qualified'):
                            votes=distinct_votes(self.xy,xy,[checked['dx'],checked['dy']],native)
                            info.update(distinct_registered_strict_pixels=votes,source_windows=2,age_s=t-self.t,
                                match_cycle=checked.get('cycle_consistency'),tentative_displacement_not_a_state_velocity=True)
                            info['current_displacement_xy']=[float(checked['dx']),float(checked['dy'])]
                            if votes>=12:info.update(used=True,reason='two_window_distinct_strict_evidence_and_current_bipolar_match')
            box=cv2.boundingRect(native.view(np.uint8));self.previous,self.mask=self.history.update((frame['on'],frame['off']),native,box);self.xy=xy;self.t=t
        if not info['used']:info.setdefault('reason','insufficient_distinct_qualified_birth_evidence')
        return info


class TemporalBirthContinuation(RoleSeparatedContinuation):
    def __init__(self,*,seed_motion_search=False):
        super().__init__(prior_consistent=True,allow_period_gap=True);self.birth=TentativeRotorBirth()
        self.seed_motion_search=bool(seed_motion_search)
    def prepare(self,frame,front,t):
        info=super().prepare(frame,front,t)
        if not info.get('raw_cold_entry_qualified') and not info.get('used'):
            proof=self.birth.consider(frame,front,t);info['tentative_birth']=proof
            if proof.get('used'):
                if self.seed_motion_search:
                    # The matched past/current pair may seed the next image
                    # search. It is NOT an observation of the physical state's
                    # velocity, a motor speed, or an inherited position fix.
                    front.last_velocity=np.asarray(proof['current_displacement_xy'])/proof['age_s']
                    info['birth_displacement_only_seeds_image_search']=True
                frame.update(current_spatial_track_qualified=True,current_support_without_force_update=True,
                    regional_only_motor_frequency_unobserved=True,continued_observation_preserves_gap_prior=True,
                    current_spatial_track=dict(reason='current_native_birth_after_temporal_period_confirmation',
                        current_motor_frequency_measured=False,tentative_velocity_not_imported=True))
                info.update(used=True,current_strict_period_refresh=True,qualification_roles_separated=True,
                    motor_bank_not_updated=True,current_known_force_created=False,current_support_is_real_image_measurement=True,
                    reason=proof['reason'],birth_from_current_and_past_evidence=True)
        else:
            self.birth.t=None;self.birth.previous=None;self.birth.mask=None
        self.last_info=info;return info
