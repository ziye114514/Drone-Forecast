"""ROI-only current ON/OFF geometry; reuse conic, curvature and five-circle code.

No historical multi-frontend controls, whole-frame masks, or virtual rotor pixels.
Counts retain a20ms physical window independent of the requested update rate.
"""
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'src'),str(ROOT/'research/stage146'),str(ROOT/'research/stage147')]
from event_drone_forecast.geometry import fit_rotor_geometry,_fit,_valid,_distance,_coverage
from research.stage146.branches146 import BranchGeometry,rotor_rates,plane_axis
from research.stage147.curvature147 import split_contour
from research.stage6.silhouette import body_measurement
from five_circle_model import FiveCircleModel
from continuous_five_circle import ContinuousFiveCircle
from shared_arc_recovery import recover as recover_shared_arcs
from partial_arc_information import update as update_partial_arcs


def roi_bounds(box,w=1280,h=720,padding=.15):
    b=np.asarray(box,float);pad=np.maximum((b[2:]-b[:2])*padding,12.)
    return np.concatenate((np.maximum(0,np.floor(b[:2]-pad)),np.minimum([w,h],np.ceil(b[2:]+pad)))).astype(int)


def roi_events(events,bounds):
    x,y,X,Y=bounds
    return events[(events['x']>=x)&(events['x']<X)&(events['y']>=y)&(events['y']<Y)]


class NativeROI:
    """Installed SDK C++ rectangle filter; same original coordinates/order."""
    def __init__(self):
        from metavision_sdk_core import RoiFilterAlgorithm
        self.algorithm=RoiFilterAlgorithm
        self.output=RoiFilterAlgorithm(0,0,1279,719,False).get_empty_output_buffer()
    def select(self,events,bounds):
        x,y,X,Y=map(int,bounds)
        a=self.algorithm(x,y,X-1,Y-1,False);a.process_events(events,self.output)
        return self.output.numpy()


def count_pair(events,bounds,t,selector=None,cpp_histogram=False):
    x,y,X,Y=bounds;w,h=X-x,Y-y
    # Timestamp-ordered suffix is a view, not a structured-array boolean copy.
    begin=int(np.searchsorted(events['t'],round(t*1e6)-20000))
    e=events[begin:]
    e=roi_events(e,bounds) if selector is None else selector.select(e,bounds)
    index=(e['y'].astype(np.int32)-y)*w+(e['x'].astype(np.int32)-x)
    # One joint polarity/pixel histogram avoids two boolean gathers of index.
    # Both returned planes remain contiguous and retain exact integer counts.
    ids=index+e['p'].astype(np.int32)*(w*h);bins=2*w*h
    if cpp_histogram and len(e)<2**24 and bins<=2**24:
        # Every pixel index and count is exactly representable as float32.
        # OpenCV writes the final float32 histogram directly, avoiding a large
        # int64 histogram plus its float32 copy. No saturation or sampling.
        if len(ids):
            if ids.min()<0 or ids.max()>=bins:raise ValueError('ROI event index outside count histogram')
            counts=cv2.calcHist([ids.astype(np.float32)],[0],None,[bins],[0.,float(bins)]).reshape(2,h,w)
        else:counts=np.zeros((2,h,w),np.float32)
    else:
        joint=np.bincount(ids,minlength=bins)
        counts=joint.reshape(2,h,w).astype(np.float32)
    return [counts[1],counts[0]]


def record(g,offset):
    g=dict(g);p=np.asarray(g.pop('center'))+offset
    return dict(g,cx=float(p[0]),cy=float(p[1]),current_observation=True,
        fov_clipped=bool(p[0]-g['major']/2<0 or p[0]+g['major']/2>=1280 or
                         p[1]-g['minor']/2<0 or p[1]+g['minor']/2>=720))


class FastStructure:
    def rebase_body_reference(self,shift):
        """Change the body-coordinate reference without moving image hubs.

        Guidance, pending confirmations and the shared affine state all store
        a body anchor. A late front-end reference correction must update those
        anchors together, otherwise its next prediction adds the correction a
        second time as if it were camera-plane motion.
        """
        shift=np.asarray(shift,float)
        if shift.shape!=(2,) or not np.isfinite(shift).all():raise ValueError('Finite2D body-reference change required')
        core=getattr(self.model,'core',self.model);updated=[]
        for obj,key,label in [(self,'previous_body','arc_guidance'),(core,'anchor','rigid_template'),
                              (self.model,'body','shared_affine')]:
            value=getattr(obj,key,None)
            if value is not None:
                setattr(obj,key,np.asarray(value,float)+shift);updated.append(label)
        pending=getattr(core,'pending',None)
        if pending is not None and pending[1] is not None:
            core.pending=(pending[0],np.asarray(pending[1],float)+shift);updated.append('pending_confirmation')
        return dict(shift_xy=shift.tolist(),updated_anchors=updated,image_hubs_moved=False,
            image_conic_shapes_moved=False,reference_change_is_measured_motion=False)

    def __init__(self,reuse_uncertainty=False,continuous=True,shared_arc_recovery=True,guide_full_only=False,bipolar_guide_area=False,contrast_guide_area=True,current_shape_feedback=True,partial_arc_information=True,optimized_compute=False,side_axis_budget=False,native_histogram=False,jit_counts=False,jit_conics=False,body_workspace=False,legacy_visibility=False):
        self.geometry=BranchGeometry();self.model=ContinuousFiveCircle(current_shape_feedback) if continuous else FiveCircleModel(3)
        self.tick=0;self.selector=NativeROI()
        self.legacy_visibility=bool(legacy_visibility)
        self.reconsider_small_cores=False
        self.projected_fov_guard=False
        self.isolated_rim_refit=False
        self.previous=[];self.previous_body=None;self.previous_t=None
        self.reuse_uncertainty=reuse_uncertainty
        self.shared_arc_recovery=shared_arc_recovery
        self.guide_full_only=guide_full_only
        self.bipolar_guide_area=bipolar_guide_area
        self.contrast_guide_area=contrast_guide_area
        self.partial_arc_information=partial_arc_information
        self.side_axis_budget=side_axis_budget;self.last_side_qualified=False
        self.native_histogram=native_histogram
        self.count_current=None
        if jit_counts:
            from native_counts_numba import NumbaCounter
            self.count_current=NumbaCounter()
        self.fit_current=fit_rotor_geometry;self.split_current=split_contour
        self.optimized_compute=optimized_compute
        self.measure_body=body_measurement
        self.body_workspace=None
        if optimized_compute:
            from functools import partial
            from fast_geometry_compute import fit_rotor_geometry_fast,split_contour_fast
            self.fit_current=fit_rotor_geometry_fast;self.split_current=split_contour_fast
            self.measure_body=partial(body_measurement,skip_unqualified_lines=True)
        if body_workspace:
            from functools import partial
            from native_body_workspace import BodyWorkspace
            self.body_workspace=BodyWorkspace()
            self.measure_body=partial(body_measurement,skip_unqualified_lines=optimized_compute,body_workspace=self.body_workspace)
        self.uncertainty_current=None
        if jit_conics:
            if not optimized_compute:raise ValueError('Fused conic uncertainty requires optimized compute')
            from functools import partial
            from native_geometry_numba import JITUncertainty,JITLine
            self.uncertainty_current=JITUncertainty()
            line=JITLine();self.uncertainty_current.provenance['line_kernel']=line.provenance
            self.fit_current=partial(self.fit_current,uncertainty_fn=self.uncertainty_current,line_fn=line)

    def quick_conic(self,contour,offset,body,t,max_major):
        """Current AMS conic; reuse conservative uncertainty for at most2ticks.

        Full SVD uncertainty is refreshed at10Hz. Shape/centre/residual are still
        measured NOW. This approximates uncertainty, not observed hub positions.
        """
        if not self.reuse_uncertainty or self.tick%3==1 or not self.previous or t-self.previous_t>.10:return None
        points=np.asarray(contour,float).reshape(-1,2);p=_fit(points)
        if not _valid(p,points,max_major,None):return None
        delta=np.asarray(body['center'])-self.previous_body
        prior=min(self.previous,key=lambda r:np.linalg.norm(p[:2]+offset-[r['cx'],r['cy']]-delta))
        if prior.get('fit_condition') is None or prior.get('mode')=='line':return None
        size=np.array([2*p[2]/prior['major'],2*p[3]/prior['minor']])
        angle=abs((np.rad2deg(p[4])-prior['angle']+90)%180-90)
        error=np.linalg.norm(p[:2]+offset-[prior['cx'],prior['cy']]-delta)
        if size.min()<.85 or size.max()>1.15 or angle>5 or error>max(8.,.25*prior['major']):return None
        distance=abs(_distance(points,p));keep=distance<max(1.4,.048*p[2])
        if keep.sum()<10:return None
        coverage=_coverage(points[keep],p);support=float(keep.mean());median=float(np.median(distance[keep]))
        if coverage<.48 or support<.6:return None
        g=dict(prior);g.pop('cx');g.pop('cy')
        sigma=max(prior['center_sigma_px']*1.05,median,(1-coverage)*p[2]*.15)
        g.update(center=p[:2],major=float(2*p[2]),minor=float(2*p[3]),angle=float(np.rad2deg(p[4])),
            ratio=float(p[3]/p[2]),normal=np.array([-np.sin(p[4]),np.cos(p[4])])*np.sqrt(max(0.,1-(p[3]/p[2])**2)),
            mode='ellipse' if coverage>=.82 and support>=.88 else 'arc',
            inferred_shape=coverage<.82 or support<.88,visible_arc_fraction=coverage,support_fraction=support,
            fit_residual_px=median,center_sigma_px=sigma,center_covariance_px2=(np.eye(2)*sigma*sigma).tolist(),
            uncertainty_source='recent_full_fit_inflated_current_shape_and_residual_checked',
            uncertainty_approximate=True)
        return g

    def step(self,events,t,body,q,velocity,frequency_pixels_unique=False,defer_rates=False,defer_geometry=False):
        self.tick+=1
        empty=dict(state='insufficient',observed_count=0,observed_rotors=[],confidence=0.)
        if not body.get('detected'):
            self.last_side_qualified=False
            return self.model.step(t,body,empty),[],dict(reason='no_current_body'),None
        if defer_geometry:
            self.last_side_qualified=False
            return self.model.step(t,body,empty),[],dict(reason='scheduled_unresolved_small_target_rescan',
                geometry_observation_deferred=True,cached_geometry_not_a_current_measurement=True,rotors=[]),None
        bounds=roi_bounds(body['bbox']);x,y,X,Y=bounds
        if self.count_current is not None:on,off=self.count_current(events,bounds,t,self.selector)
        else:on,off=count_pair(events,bounds,t,self.selector,cpp_histogram=self.native_histogram)
        density=np.minimum(on,off)
        # Same stage6 thresholds and side qualification, on the small native ROI.
        local_box=np.asarray(body['bbox'])-np.tile(bounds[:2],2)
        side={}
        side_fast=False
        support_ratio=None
        if not self.legacy_visibility:
            # Current image evidence, not last frame's label, decides whether
            # a side-axis shortcut may suppress all current conic extraction.
            from curvature_visibility import masks
            candidate_mask,_=masks(on,off)
            moments=cv2.moments(candidate_mask,binaryImage=True)
            if moments['m00']>0:
                covariance=np.array([[moments['mu20'],moments['mu11']],[moments['mu11'],moments['mu02']]])/moments['m00']
                values=np.linalg.eigvalsh(covariance)
                support_ratio=float(np.sqrt(max(values[0],0.)/max(values[1],1e-9)))
        if self.side_axis_budget and self.last_side_qualified and self.previous_t is not None and t-self.previous_t<=.12 and self.tick%3!=1:
            side=self.measure_body(density,local_box)
            side_fast=bool(side.get('valid') and side.get('confidence',0)>=.45)
            if not self.legacy_visibility:
                side_fast=bool(side_fast and support_ratio is not None and support_ratio<=.16)
        # Reuse neighbor10 exactly: spatially align each polarity BEFORE min.
        # min(blur(ON),blur(OFF)) differs materially from blur(min(ON,OFF)).
        raw_mask=None;contours=[]
        if not side_fast:
            feature=np.minimum(cv2.GaussianBlur(on,(3,3),.8),cv2.GaussianBlur(off,(3,3),.8))
            feature[(on+off)<2]=0
            blurred=cv2.GaussianBlur(feature,(3,3),.8)
            raw_mask=np.uint8(blurred>=4.)*255
            # Existing morphology: remove isolated gaps without fitting their edges.
            mask=cv2.morphologyEx(raw_mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contours=sorted([c for c in contours if len(c)>=16 and cv2.contourArea(c)>=20],key=cv2.contourArea,reverse=True)[:12]
        rotors=[];split_count=0
        # Degenerate side views need an axis, not repeated ill-conditioned conics.
        # A thin whole-body projection alone cannot exclude four visible discs.
        # Rebuild the complete template at10Hz even in the cheap side branch.
        # Cheap complete/qualified-arc fits each current tick. The body-axis LSD
        # is a fallback AFTER checking for four discs, never a preemptive route.
        if contours:
            for c in contours[:6]:
                # Fast current complete conics first. Expensive partial-arc
                # proposals are confined to the validated curvature path below.
                max_major=min(500.,1.5*max(X-x,Y-y))
                g=self.quick_conic(c,bounds[:2],body,t,max_major)
                if g is None:
                    options=dict(max_major=max_major,allow_arc=False)
                    if self.optimized_compute:options['minimum_major']=max([r['major'] for r in rotors],default=0.)/2.5
                    g=self.fit_current(c,**options)
                if g is not None:
                    g['arc_center_qualified']=bool(g['mode']=='arc' and g.get('visible_arc_fraction',0)>=.48 and
                        g.get('support_fraction',0)>=.6 and g.get('fit_condition',1e9)<=1500 and
                        g.get('center_sigma_px',1e9)<=3 and g.get('axis_angle_sigma_deg',90)<=15)
                    rotors.append(record(g,bounds[:2]))
            if rotors:
                # Existing merge110 size-ratio convention removes tiny body
                # details; independent discs still have to pass joint geometry.
                largest=max(r['major'] for r in rotors)
                rotors=[r for r in rotors if r['major']>=largest/2.5]
            # Original concavity/arc validation runs at10Hz, never on every side tick.
            if self.legacy_visibility and len(rotors)!=4 and self.tick%15==1:
                for c in contours[:1]:
                    arcs,_=self.split_current(c)
                    if arcs is None:continue
                    replacements=[]
                    for arc in arcs:
                        p=arc['parameters'];replacements.append(dict(cx=float(p[0]+x),cy=float(p[1]+y),
                            major=float(2*p[2]),minor=float(2*p[3]),ratio=float(p[3]/p[2]),angle=float(np.rad2deg(p[4])),
                            mode='arc',inferred_shape=True,curvature_validated=True,current_observation=True,
                            fov_clipped=False,center_sigma_px=arc['sigma'],fit_residual_px=arc['rms'],geometry_quality=.6))
                    # Replace only the original component whose contour contains its fit centre.
                    rotors=[r for r in rotors if cv2.pointPolygonTest(c,(r['cx']-x,r['cy']-y),False)<0]+replacements
                    split_count+=1
        merge_audit=dict(accepted_pairs=0,reason='four_already_observed_or_current_projection_flat')
        local_opportunity=False
        if getattr(self,'local_conic_curvature',False) and not side_fast:
            from local_conic_opportunity import nonflat_contour_present
            local_opportunity=nonflat_contour_present(contours)
        if not self.legacy_visibility and not side_fast and len(rotors)!=4 and support_ratio is not None and (support_ratio>.16 or local_opportunity):
            interval=getattr(self,'curvature_interval_s',0.)
            last=getattr(self,'last_curvature_t',None)
            due=not interval or last is None or round((t-last)*1e6)>=round(interval*1e6)-1
            if due:
                from curvature_visibility import recover
                rotors,merge_audit=recover(on,off,rotors,bounds[:2],reconsider_small_cores=self.reconsider_small_cores,
                    selection=getattr(self,'arc_pair_selection','native_area'))
                self.last_curvature_t=t;split_count+=merge_audit['accepted_pairs']
            else:
                merge_audit=dict(accepted_pairs=0,reason='bounded_curvature_search_rate_current_conics_retained',
                    expensive_search_deferred=True,current_cheap_conics_and_axis_still_measured=True,
                    cached_rotors_not_reported_as_current=True,curvature_interval_s=interval)
        if getattr(self,'local_conic_curvature',False):
            merge_audit['local_nonflat_contour_opportunity']=local_opportunity
            merge_audit['local_opportunity_is_not_rotor_confirmation']=True
        isolated_rims=dict(accepted=0,reason='disabled_or_current_projection_flat')
        if self.isolated_rim_refit and not self.legacy_visibility and not side_fast and support_ratio is not None and support_ratio>.16:
            from isolated_rim_visibility import recover_isolated_rims
            rotors,isolated_rims=recover_isolated_rims(on,off,rotors,bounds[:2])
        arc_recovery=dict(attempted=False,accepted=0)
        if self.shared_arc_recovery and self.tick%3==1 and body.get('confidence',0)>=.45:
            area_mask=None
            if self.bipolar_guide_area:
                occupancy=((on>0)&(off>0)).astype(np.float32)
                area_mask=np.uint8(cv2.GaussianBlur(occupancy,(3,3),.8)>=.5)*255
            rotors,arc_recovery=recover_shared_arcs(self.model,t,body,raw_mask,bounds[:2],rotors,
                full_only=self.guide_full_only,area_mask=area_mask,contrast_area=self.contrast_guide_area)
        if self.projected_fov_guard:
            from ellipse_visibility import ellipse_fov_clipped
            rotors=[dict(r,fov_clipped=ellipse_fov_clipped(r),fov_check_uses_rotated_conic_extent=True) for r in rotors]
        geom=self.geometry.step(t,body,rotors)
        # Curvature validates actual open arcs; permit the existing four-point model.
        qualified=[r for r in rotors if not(self.projected_fov_guard and r.get('fov_clipped')) and
            (r.get('mode')=='ellipse' or r.get('curvature_validated') or r.get('arc_center_qualified'))]
        if len(rotors) in (3,4) and len(qualified)==len(rotors) and geom['state'] not in ('four','three_completed'):
            adapted=[dict(r,mode='ellipse',inferred_shape=False) for r in rotors]
            geom=self.geometry.step(t,body,adapted);geom['observed_rotors']=rotors
        shared_shape=True
        if not self.legacy_visibility and len(qualified)==4 and len(rotors)==4:
            # Extend146's existing three-disk affine-plane ratio guard to a
            # four-region hypothesis. Four individually good arcs can include
            # a leg or a still-merged disk and need not define one rigid plane.
            from research.stage146.branches146 import PARAMS as branch_params
            ratios=np.array([r['ratio'] for r in qualified]);sizes=np.array([r['major'] for r in qualified])
            shared_shape=bool(ratios.max()/ratios.min()<=branch_params['three_max_ratio_spread'] and sizes.max()/sizes.min()<=2.)
            if not shared_shape:geom.update(state='four_unresolved',confidence=0.,reason='four_regions_not_one_consistent_projected_disk_plane')
        structure=self.model.step(t,body,geom)
        normal_constraints=dict(updated=False,reason='disabled')
        if self.partial_arc_information and not side_fast:
            normal_constraints=update_partial_arcs(self.model,t,body,raw_mask,bounds[:2],rotors)
            if normal_constraints['updated']:
                structure['model_centers']=self.model._hubs().tolist()
                structure['model_conditioned_center']=self.model.beta[[0,3]].tolist()
                structure['model_conditioned_center_covariance']=self.model.P[np.ix_([0,3],[0,3])].tolist()
                structure['five_circles']['rotors']=structure['model_centers']
                if structure['current_geometric_center'] is not None:
                    structure['current_geometric_center']=self.model.beta[[0,3]].tolist()
                    structure['five_circles']['body']=structure['current_geometric_center']
                structure['continuous_normal_constraints']=normal_constraints
        visible_plane=False
        axis_candidates=qualified
        if getattr(self,'qualified_plane_switch',False):
            # Reuse the existing current arc-fit uncertainty limits. A broad
            # or ill-conditioned conic hypothesis must not veto a fresh line
            # measurement merely because an ellipse can be drawn through it.
            axis_candidates=[r for r in qualified if r.get('current_observation',True) and
                not r.get('association_ambiguous') and r.get('center_sigma_px',float('inf'))<=3 and
                r.get('axis_angle_sigma_deg',float('inf'))<=15]
        if (not self.legacy_visibility or structure.get('axis_xy') is None) and len(axis_candidates)>=2 and np.median([r['ratio'] for r in axis_candidates])>.16:
            normal,agreement=plane_axis(axis_candidates)
            if agreement>=.85:
                visible_plane=True
                refresh=False
                if getattr(self,'qualified_plane_switch',False):
                    from axis_amplitude_bridge import current_axis_qualified
                    refresh=not current_axis_qualified(structure)
                if structure.get('axis_xy') is None or refresh:
                    structure.update(axis_xy=normal.tolist(),geometric_weight=.35,
                        pose_source='current_qualified_visible_conics',pose_without_complete_five_circle=True)
        if structure['current_geometric_center'] is None and not visible_plane and not side:
            side=self.measure_body(density,local_box)
        if side.get('valid') and side.get('confidence',0)>=.45 and structure['current_geometric_center'] is None and (self.legacy_visibility or not visible_plane):
            n=np.asarray(side['normal_axis_xy'])
            # Whole-body elongation is an unsigned axis, not proof of exactly
            # two physical motors. Keep this unresolved mode distinct.
            flat=bool(qualified and np.median([r['ratio'] for r in qualified])<=.16)
            mode='edge_on' if flat else 'axis_only'
            if hasattr(self.model,'core'):self.model.core.mode=mode
            else:self.model.mode=mode
            structure.update(mode=mode,axis_xy=n.tolist(),geometric_weight=.35,
                reason='stage6_current_ROI_count_axis',axis_support=dict(side,center=(np.asarray(side['center'])+[x,y]).tolist()))
            if hasattr(self.model,'observe_axis'):
                structure=self.model.observe_axis(structure,n,side['confidence'],t)
        rates=rotor_rates(q,rotors,t,velocity,1280,unique_pixels=frequency_pixels_unique) if rotors and not defer_rates else []
        self.previous=rotors;self.previous_body=np.asarray(body['center']);self.previous_t=t
        self.last_side_qualified=bool(side.get('valid') and side.get('confidence',0)>=.45 and
            (self.legacy_visibility or not visible_plane and support_ratio is not None and support_ratio<=.16))
        if not self.legacy_visibility:
            four_current=bool(shared_shape and len(qualified)==4 and len(rotors)==4 and
                len(cv2.convexHull(np.array([[r['cx'],r['cy']] for r in qualified],np.float32)))==4 and visible_plane)
            structure.update(view_evidence='four_disks_visible' if four_current else
                'partial_nonflat_disks' if visible_plane else 'line_projection' if self.last_side_qualified else 'unresolved',
                side_view_confirmed=bool(self.last_side_qualified),current_rotor_hypothesis_count=len(rotors),
                four_region_shared_shape_qualified=shared_shape,
                view_evidence_is_full_3d_pose=False)
        diag=dict(roi=bounds.tolist(),current_window_us=20000,threshold=4.,contours=len(contours),
            curvature_splits=split_count,shared_arc_recovery=arc_recovery,
            curvature_visibility=merge_audit,current_support_eigen_ratio=support_ratio,
            isolated_rim_visibility=isolated_rims,
            partial_arc_information=normal_constraints,
            side=side,observed_regions=len(rotors),rotors=rotors,roi_pixels=int((X-x)*(Y-y)))
        if self.side_axis_budget:
            diag.update(current_axis_budget_tick=side_fast,full_conic_refresh_interval_ticks=3,
                rotor_count_unresolved_on_axis_tick=side_fast,
                scheduled_tick_is_not_a_new_rotor_or_hidden_pose_observation=True)
        return structure,rates,diag,(density,bounds)
