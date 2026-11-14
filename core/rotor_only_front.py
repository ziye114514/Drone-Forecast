"""Current rotor evidence -> native contours -> observations, without a body NN.

Strict period pixels are seeds, never artificial ellipse rims. Reconstruction
selects original ON/OFF components touched by those seeds (SciPy morphological
reconstruction). Extra rim pixels are native activity, not confirmed periodic
pixels. All display planes zero everything outside their stated mask.
"""
import cv2
import numpy as np
import time
from scipy.ndimage import binary_dilation,binary_propagation
from rotor_support_front import rotor_support,estimate_region_translation
from fast_structure_front import FastStructure


def filtered_rgb(on,off,mask):
    out=np.zeros(on.shape+(3,),np.uint8)
    energy=np.maximum(on,off)
    scale=max(float(np.percentile(energy[mask],99)) if np.any(mask) else 1.,1.)
    out[:,:,0]=np.uint8(np.clip(np.log1p(on)/np.log1p(scale)*255,0,255))
    out[:,:,2]=np.uint8(np.clip(np.log1p(off)/np.log1p(scale)*255,0,255))
    out[~mask]=0
    return out


def native_reconstruction(frame):
    """Propagate only through native bipolar components; no distance-cut rims."""
    if frame.get('native_precomputed') is not None:
        return frame['native_precomputed']
    strict=frame['period_mask']>0
    if not np.any(strict):
        return np.zeros_like(strict),dict(strict_pixels=0,native_pixels=0,mask_support_pixels=0,
            seed_pixels=0,added_native_pixels_not_period_confirmed=True,mask_boundary_not_used_as_physical_rim=True)
    on=frame['on'].astype(np.float32);off=frame['off'].astype(np.float32)
    feature=np.minimum(cv2.GaussianBlur(on,(3,3),.8),cv2.GaussianBlur(off,(3,3),.8))
    feature[(on+off)<2]=0
    mask=np.uint8(cv2.GaussianBlur(feature,(3,3),.8)>=4.)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
    seeds=cv2.dilate(strict.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),
        iterations=2,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)&mask
    # Exact propagation inside an8-connected mask, using the existing OpenCV
    # connected-component primitive rather than repeated whole-image dilation.
    number,labels=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
    touched=np.unique(labels[seeds]);touched=touched[touched>0]
    lookup=np.zeros(number,bool);lookup[touched]=True;reconstructed=lookup[labels]
    # Only original event-bearing pixels are kept in image/motion/event outputs.
    native=reconstructed&((on+off)>0)
    return native,dict(strict_pixels=int(strict.sum()),native_pixels=int(native.sum()),
        mask_support_pixels=int(reconstructed.sum()),seed_pixels=int(seeds.sum()),
        added_native_pixels_not_period_confirmed=True,mask_boundary_not_used_as_physical_rim=True)


def select_native(events,t,mask,window_us=33333):
    end=round(t*1e6)
    keep=(events['t']>=end-window_us)&(events['t']<end)
    # Native integer coordinates are already valid NumPy indices. Select once,
    # avoiding a full event copy and two int64 coordinate arrays every window.
    return events[keep&mask[events['y'],events['x']]]


class RotorOnlyFront:
    def __init__(self,period_estimator=None,fast_geometry=False,subpixel_motion=False,motion_prior=False,plane_motion=False,coarse_motion=False,relock_support_guard=False,border_motion=False,legacy_visibility=False,guard_invalid_motion=None,rebase_geometry_reference=False,fixed_output_reference=False,centered_motion=False):
        self.geometry=FastStructure(shared_arc_recovery=True,partial_arc_information=True,
            optimized_compute=fast_geometry,native_histogram=fast_geometry,
            reuse_uncertainty=fast_geometry,side_axis_budget=fast_geometry,legacy_visibility=legacy_visibility)
        self.legacy_visibility=bool(legacy_visibility);self.geometry_reference_confirmed=False
        self.rebase_geometry_reference=bool(rebase_geometry_reference)
        self.fixed_output_reference=bool(fixed_output_reference)
        self.guard_invalid_motion=not legacy_visibility if guard_invalid_motion is None else bool(guard_invalid_motion)
        if fast_geometry:
            from functools import partial
            from native_axis_workspace import NativeAxisWorkspace
            from research.stage6.silhouette import body_measurement
            self.geometry.body_workspace=NativeAxisWorkspace()
            self.geometry.measure_body=partial(body_measurement,skip_unqualified_lines=True,body_workspace=self.geometry.body_workspace)
        self.period_estimator=period_estimator
        self.fast_geometry=fast_geometry;self.provide_phase_events=True
        self.subpixel_motion=bool(subpixel_motion)
        self.motion_prior=bool(motion_prior);self.last_velocity=None
        self.plane_motion=bool(plane_motion)
        self.coarse_motion=bool(coarse_motion)
        self.relock_support_guard=bool(relock_support_guard);self.support_extents=[];self.anchor_extent=None
        self.border_motion=bool(border_motion)
        self.centered_motion=bool(centered_motion)
        self.cycle_motion=False
        self.previous=None;self.previous_mask=None;self.center=None;self.last_t=None
        self.motion_counts20=False
        self.occupancy_motion=False
        self.native_shape_motion=False
        self.polarity_motion_quality=False
        self.current_region_search=False
        self.history_buffer=None
        self.periodic_continuation=None
        self.side_line_reference=False;self.side_line_reference_active=False
        if self.period_estimator is not None:
            from native_history_buffer import NativeHistoryBuffer
            self.history_buffer=NativeHistoryBuffer()
        self.center_confirmed=False
        if fast_geometry:
            # Load small-matrix LAPACK paths before live observations arrive.
            np.linalg.pinv(np.eye(5));np.linalg.svd(np.eye(5),compute_uv=False);np.linalg.eigh(np.eye(2))

    def step(self,events,t):
        started=time.perf_counter()
        if self.center is None:self.side_line_reference_active=False
        self.support_extents=[(s,e) for s,e in self.support_extents if 0<=t-s<=.5]
        frame=rotor_support(events,t,period_estimator=self.period_estimator,fast_spatial=True,
            native_only=True,retain_candidates=self.provide_phase_events)
        if self.periodic_continuation is not None:
            frame['periodic_track_continuation']=self.periodic_continuation.prepare(frame,self,t)
            if frame['periodic_track_continuation'].get('used'):self.side_line_reference_active=True
        native,diag=native_reconstruction(frame)
        strict=frame['period_mask']>0
        # Period discovery still uses every event in33.333ms. Motion can use
        # the already available20ms native counts, matching the geometry time
        # support and avoiding an unnecessary extra13.333ms of blade smear.
        short_counts=frame.get('native_counts20')
        use_short=bool(self.motion_counts20 and short_counts is not None)
        current=short_counts if use_short else (frame['on'],frame['off'])
        if self.occupancy_motion:
            # Motion of a rotor envelope, not motion of its rapidly changing
            # log-count texture. Geometry and period inputs remain ORIGINAL.
            occupied=native.astype(np.float32)
            current=(occupied,occupied)
        supported=time.perf_counter();plane_info=dict(qualified=False,reason='disabled')
        warped=None;warp_info=None
        match_previous=self.previous;match_old_mask=self.previous_mask
        match_current=current;match_mask=native
        if getattr(self,'timestamped_motion',False):
            from timestamped_rotor_motion import render
            a,b,warp_info=render(events,t,native,self.last_velocity)
            warped=(a,b,float(t))
        shift=None
        motion_dt=None
        if self.previous is not None and self.last_t is not None and 0<t-self.last_t<=.12:
            motion_dt=t-self.last_t
            verified=frame.get('verified_native_displacement')
            if verified is not None:
                if (abs(verified['source_t']-self.last_t)>1e-9 or abs(verified['cutoff_t']-t)>1e-9 or abs(verified['dt_s']-motion_dt)>1e-9):
                    raise ValueError('Verified displacement belongs to another native observation interval')
                shift=dict(verified['estimate'])
            prediction=self.last_velocity*motion_dt if self.motion_prior and self.last_velocity is not None else (0.,0.)
            radius=int(np.ceil(40*motion_dt*30)) if self.motion_prior else 40
            if self.current_region_search and np.any(native) and np.any(self.previous_mask):
                cx,cy,cw,ch=cv2.boundingRect(native.view(np.uint8));ox,oy,ow,oh=cv2.boundingRect(self.previous_mask.view(np.uint8))
                region_delta=np.array([cx+cw/2-ox-ow/2,cy+ch/2-oy-oh/2])
                # A measured coherent region proposes a search centre, never a
                # displacement. Preserve the original peak test and160px budget.
                if np.max(abs(region_delta-np.asarray(prediction)))>radius and np.max(abs(region_delta))<=160:
                    prediction=region_delta
            if shift is None and self.native_shape_motion and self.center is not None:
                from native_shape_registration import estimate
                shape,plane_info=estimate(self.previous_mask,native)
                if shape is not None:
                    matrix=np.asarray(shape['shared_affine'],float)
                    delta=matrix[:,:2]@self.center+matrix[:,2]-self.center
                    if np.isfinite(delta).all() and np.max(abs(delta))<=160:
                        shift=dict(dx=float(delta[0]),dy=float(delta[1]),score=min(plane_info['ecc'],plane_info['reverse_ecc']),
                            method='native_rotor_occupancy_shape_ECC',search_px=160,peak_at_search_boundary=False,
                            current_displacement_qualified=True,shared_affine=shape['shared_affine'],
                            displacement_covariance_px2=shape['displacement_covariance_px2'],
                            phase_texture_not_used=True,bidirectional_shape=plane_info)
            if shift is None and self.plane_motion and self.center is not None:
                from native_plane_motion import estimate_plane_motion
                shift,plane_info=estimate_plane_motion(self.previous,self.previous_mask,current,native,self.center,
                    require_center_support=getattr(self,'plane_center_hull_required',False),
                    local_contrast=getattr(self,'plane_local_contrast',False))
            if shift is None:
                cached=getattr(self,'timestamped_previous',None)
                if warped is not None and cached is not None and abs(cached[2]-self.last_t)<1e-9:
                    match_previous,match_old_mask=cached[:2];match_current,match_mask=warped[:2]
                surface_enabled=getattr(self,'symmetric_surface_motion',False)
                if surface_enabled and getattr(self,'symmetric_surface_complete_only',False):
                    # Reuse the existing6px-padded complete-support criterion.
                    # Censored fields of view cannot require reciprocal matches.
                    for mm in (match_old_mask,match_mask):
                        xx,yy,ww,hh=cv2.boundingRect(mm.view(np.uint8))
                        if not (ww and xx>6 and yy>6 and xx+ww<1274 and yy+hh<714):surface_enabled=False
                shift=estimate_region_translation(match_previous,match_old_mask,match_current,current_support=match_mask,
                    native_matching=self.fast_geometry,subpixel=self.subpixel_motion,max_shift=radius,predicted_shift=prediction,coarse_matching=self.coarse_motion,interior_border=self.border_motion,centered_matching=self.centered_motion,
                    peak_shape=bool(getattr(self,'peak_shape_uncertainty',False) or getattr(self,'joint_peak_refinement',False)),
                    joint_peak=getattr(self,'joint_peak_refinement',False),peak_covariance=getattr(self,'peak_shape_uncertainty',False),
                    joint_peak_neighborhood=getattr(self,'joint_peak_full_neighborhood',False),
                    polarity_separated=getattr(self,'polarity_separated_motion',False),
                    capture_surface=surface_enabled,local_contrast=getattr(self,'local_contrast_motion',False))
                if shift is not None and warped is not None and cached is not None and abs(cached[2]-self.last_t)<1e-9:
                    shift['timestamped_motion_rectification']=warp_info
                if self.cycle_motion and shift is not None:
                    from native_cycle_consistency import qualify
                    reverse=estimate_region_translation(match_current,match_mask,match_previous,current_support=match_old_mask,
                        native_matching=self.fast_geometry,subpixel=self.subpixel_motion,max_shift=radius,
                        predicted_shift=-np.asarray(prediction),coarse_matching=self.coarse_motion,
                        interior_border=self.border_motion,centered_matching=self.centered_motion)
                    shift=qualify(shift,reverse)
        if (getattr(self,'symmetric_surface_motion',False) and shift is not None and '_matching_surface' in shift):
            from symmetric_match_surface import combine
            reverse=estimate_region_translation(match_current,match_mask,match_previous,current_support=match_old_mask,
                native_matching=self.fast_geometry,subpixel=self.subpixel_motion,max_shift=radius,
                predicted_shift=-np.asarray(prediction),coarse_matching=self.coarse_motion,
                interior_border=self.border_motion,centered_matching=self.centered_motion,capture_surface=True)
            shift=combine(shift,reverse)
        if (getattr(self,'symmetric_native_motion',False) and shift is not None and self.previous is not None
                and shift.get('method')=='native_normalized_rotor_match'):
            from symmetric_native_motion import combine
            reverse=estimate_region_translation(current,native,self.previous,current_support=self.previous_mask,
                native_matching=self.fast_geometry,subpixel=self.subpixel_motion,max_shift=radius,
                predicted_shift=-np.asarray(prediction),coarse_matching=self.coarse_motion,
                interior_border=self.border_motion,centered_matching=self.centered_motion,
                peak_shape=getattr(self,'joint_peak_refinement',False),joint_peak=getattr(self,'joint_peak_refinement',False),
                peak_covariance=False,local_contrast=getattr(self,'local_contrast_motion',False))
            shift=combine(shift,reverse)
        if self.polarity_motion_quality and shift is not None and self.previous is not None:
            from polarity_motion_quality import qualify
            shift=qualify(match_previous,match_old_mask,match_current,match_mask,shift,
                native_matching=self.fast_geometry,subpixel=self.subpixel_motion,max_shift=radius,
                predicted_shift=prediction,coarse_matching=self.coarse_motion,interior_border=self.border_motion,
                centered_matching=self.centered_motion,local_contrast=getattr(self,'local_contrast_motion',False))
        matched=time.perf_counter()
        selected=native if np.any(native) else strict
        count=int(np.count_nonzero(selected));bx,by,bw,bh=cv2.boundingRect(selected.view(np.uint8))
        collective=bool(diag.get('regional_seed_groups',0)>0 and diag.get('regional_seed_pixels',0)>=3)
        spatial_current=bool(frame.get('current_spatial_track_qualified',False))
        detected=bool(count>=12 and (frame['counts']['period_pixels']>=12 or collective or spatial_current))
        center_source='unavailable';body=dict(detected=False)
        anchor_fragment=False
        if detected:
            rejected_motion=bool(self.guard_invalid_motion and shift is not None and
                (shift.get('peak_at_search_boundary') or not shift.get('current_displacement_qualified',True)))
            if (self.motion_prior or self.guard_invalid_motion) and shift is not None and not rejected_motion and not shift.get('peak_at_search_boundary'):
                self.last_velocity=np.array([shift['dx'],shift['dy']])/motion_dt
            box=[max(0,bx-6),max(0,by-6),min(1279,bx+bw-1+6),min(719,by+bh-1+6)]
            if rejected_motion and self.center is not None:
                delta=np.zeros(2) if self.last_velocity is None else self.last_velocity*motion_dt
                center=self.center+delta;center_source='predicted_rotor_reference_after_rejected_match'
            elif shift is not None and self.center is not None:
                center=self.center+np.array([shift['dx'],shift['dy']]);center_source='filtered_native_translation'
            else:
                center=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2]);center_source='rotor_region_initialization'
                self.center_confirmed=False
            complete=bool(box[0]>0 and box[1]>0 and box[2]<1279 and box[3]<719)
            extent=max(bw,bh);reference_extent=max((e for _,e in self.support_extents),default=extent)
            # Complete inside the FOV does not mean complete rotor support.
            # A newly acquired small fragment is not a new mass-center fix.
            # The ratio is a declared conservative visibility prior, not a
            # claim that viewpoint/scale can never contract this quickly.
            anchor_fragment=bool(self.relock_support_guard and not self.center_confirmed and extent<.5*reference_extent)
            if not self.center_confirmed and complete and not anchor_fragment:
                # Before a complete view, the tracked point is the centroid of
                # a CLIPPED region, not the vehicle center. Do not carry that
                # fixed initialization bias through the entire later track.
                center=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
                center_source='first_complete_rotor_region_anchor';self.center_confirmed=True
                self.anchor_extent=extent
            body=dict(detected=True,bbox=box,center=center.tolist(),confidence=.7,
                collective_periodic_support_used=collective,
                measurement_source=center_source,whole_body_NN_used=False,
                center_provisional=not self.center_confirmed,low_confidence=not self.center_confirmed,
                region_center_is_not_proven_mass_center=True)
            if frame.get('continued_observation_preserves_gap_prior'):
                body['continued_observation_preserves_gap_prior']=True
                body['observation_grade']='current_rotor_support_without_new_force_input'
            body.update(relock_anchor_fragment_rejected=anchor_fragment,
                observed_support_extent_px=extent,recent_reference_extent_px=reference_extent,
                support_guard_extent_ratio=.5 if self.relock_support_guard else None)
            if spatial_current:
                body.update(current_entry_uses_recent_rotor_identity_and_current_image_match=True,
                    current_motor_period_confirmed=False,current_spatial_track=frame.get('current_spatial_track'))
            if rejected_motion:
                body.update(center_provisional=True,low_confidence=True,motion_reference_advanced_without_measurement=True,
                            invalid_displacement_not_an_absolute_fix=True,reference_center_is_prediction_only=True)
        counts20=frame.get('native_counts20')
        geometry_support=native
        if counts20 is not None:
            if not self.legacy_visibility:
                from curvature_visibility import seeded_outline
                from fast_structure_front import roi_bounds
                geometry_support=native.copy()
                if body.get('detected'):
                    gx,gy,gX,gY=roi_bounds(body['bbox'])
                    geometry_support[gy:gY,gx:gX]|=seeded_outline(counts20[0][gy:gY,gx:gX],counts20[1][gy:gY,gx:gX],strict[gy:gY,gx:gX])
            def current_counts(_events,bounds,_t,_selector):
                x,y,X,Y=bounds;keep=geometry_support[y:Y,x:X]
                return [np.where(keep,a[y:Y,x:X],0.).astype(np.float32,copy=False) for a in counts20]
            self.geometry.count_current=current_counts
            filtered=events[:0]
        else:filtered=select_native(events,t,native)
        # A rejected search-boundary peak cannot become a geometry-motion
        # prior either. Retain inertia for search/transport only.
        if self.guard_invalid_motion and shift is not None and (shift.get('peak_at_search_boundary') or not shift.get('current_displacement_qualified',True)):
            velocity=np.zeros(2) if self.last_velocity is None else self.last_velocity.copy()
        else:
            velocity=np.zeros(2) if shift is None else np.array([shift['dx'],shift['dy']])/max(t-self.last_t,1e-6)
        structure,_,geom,_=self.geometry.step(filtered,t,body,None,velocity,defer_rates=True)
        fitted=time.perf_counter()
        phase_subset=False
        if counts20 is not None and self.provide_phase_events and structure.get('mode') in ('four','three','partial_two'):
            yy,xx=np.indices(native.shape)
            filtered=select_native(events,t,native & ((xx+3*yy)%4==0));phase_subset=True
        if detected:
            geometry_input_body_center=np.asarray(body['center'],float).copy()
            c=structure.get('current_geometric_center')
            if c is not None and not self.fixed_output_reference:
                if not self.legacy_visibility and not self.geometry_reference_confirmed:
                    body['measurement_reference_shift_xy']=(np.asarray(c)-np.asarray(body['center'])).tolist()
                    body['first_current_geometry_changes_reference_not_velocity']=True
                    self.geometry_reference_confirmed=True
                body['center']=list(c);body['measurement_source']='current_rotor_geometry'
                self.center_confirmed=True;body['center_provisional']=False;body['low_confidence']=False
                body.pop('motion_reference_advanced_without_measurement',None)
                body.pop('reference_center_is_prediction_only',None)
            elif (self.relock_support_guard and center_source=='filtered_native_translation' and
                    complete and self.anchor_extent is not None and max(bw,bh)>=2*self.anchor_extent):
                # A tiny initial periodic patch may later reveal the rest of
                # the rotor assembly. Change the tracked reference explicitly;
                # do not turn that support expansion into a force impulse.
                new_center=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
                body['measurement_reference_shift_xy']=(new_center-np.asarray(body['center'])).tolist()
                body['center']=new_center.tolist();body['measurement_source']='expanded_rotor_support_reference'
                body['support_expansion_is_not_measured_translation']=True
                self.anchor_extent=max(bw,bh)
            if self.fixed_output_reference:
                body.update(output_reference='initial_native_rotor_support_anchor',
                    layout_center_is_absolute_position_fix=False,
                    current_layout_center=None if c is None else list(c))
            if self.rebase_geometry_reference:
                reference_delta=np.asarray(body['center'],float)-geometry_input_body_center
                if np.any(reference_delta):
                    body['geometry_reference_transport']=self.geometry.rebase_body_reference(reference_delta)
            self.center=np.asarray(body['center']);self.last_t=t
            if not anchor_fragment:self.support_extents.append((t,max(bw,bh)))
            # GPU feature maps are reusable shared slots. A missed detection
            # or a bounded-latency skip may outlive that ring. Keep ownership
            # of the last observed image rather than silently matching against
            # a future map that has overwritten the slot.
            if self.history_buffer is not None:
                self.previous,self.previous_mask=self.history_buffer.update(current,native,(bx,by,bw,bh))
            else:self.previous=current;self.previous_mask=native.copy()
        elif self.last_t is not None and t-self.last_t>.12:
            self.center=None;self.previous=None;self.previous_mask=None
            self.center_confirmed=False
            self.geometry_reference_confirmed=False
            self.last_velocity=None
        frame.update(native_mask=native,geometry_native_mask=geometry_support,reconstruction=diag,body=body,structure=structure,geometry=geom,
            motion=dict(estimate=shift,uses_only_filtered_ONOFF=True,
                native_motion_accumulation_us=20000 if use_short else 33333,
                frequency_accumulation_us=33333,
                plane_motion=plane_info,
                dt_s=motion_dt,past_displacement_search_prior=self.motion_prior,
                current_observation=bool(detected and not body.get('center_provisional',False)),
                current_rotor_region_detected=detected,
                current_observation_is_position_evidence_not_region_presence=True,
                raw_rectangle_texture_used=False),
            filtered_events=filtered,filtered_events_are_phase_pixel_subset=phase_subset)
        if self.fixed_output_reference:
            from layout_reference_observer import layout_packet
            frame['motion']['layout_observation']=layout_packet(structure,geom,allow_line=self.side_line_reference,line_update_allowed=self.side_line_reference_active)
        if self.occupancy_motion:frame['motion']['representation']='native_filtered_rotor_occupancy'
        if self.periodic_continuation is not None:
            self.periodic_continuation.observe(t,body)
        frame['front_stage_ms']=dict(support=1000*(supported-started),motion=1000*(matched-supported),geometry_and_observation=1000*(fitted-matched),post=1000*(time.perf_counter()-fitted))
        if warped is not None and self.last_t is not None and abs(self.last_t-t)<1e-9:
            self.timestamped_previous=warped
        return frame
