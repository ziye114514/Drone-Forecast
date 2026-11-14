"""Current native ON/OFF concave-arc grouping before side-view decisions.

Use the existing147 notch/arc uncertainty and size/angle conventions. Binary
bipolar occupancy is an outline proposal, NOT an extra periodic detector.
Original open arcs only; no watershed cut chord becomes a measured boundary.
"""
from itertools import combinations
import cv2
import numpy as np
from batched_arc_fit import fit_arc_exact as fit_arc_fast
from research.stage147.curvature147 import notches,PARAMS
from event_drone_forecast.geometry import _fit,_distance,_coverage,_uncertainty


def direct_arc(points,extent):
    """Fast path for a clean concavity-bounded arc;147 quality gates retained."""
    q=_fit(points)
    if q is None:return None
    keep=abs(_distance(points,q))<=1.8
    if keep.mean()<.90 or keep.sum()<22:return None
    q=_fit(points[keep])
    if q is None or not (10<=2*q[2]<=extent*1.6 and q[3]>=2.5 and q[3]/q[2]>=.12):return None
    keep=abs(_distance(points,q))<=1.8;p=points[keep]
    if keep.mean()<.90 or len(p)<22:return None
    rms=float(np.sqrt(np.mean(_distance(p,q)**2)));coverage=_coverage(p,q)
    if rms>=.8 or coverage<PARAMS['minimum_arc_coverage']:return None
    sigma,angle,condition=_uncertainty(p,q)
    if sigma>PARAMS['maximum_center_sigma_px'] or condition>1500:return None
    return dict(parameters=q,points=p,rms=rms,coverage=coverage,sigma=sigma,angle_sigma=angle,condition=condition,support_fraction=float(keep.mean()))


def masks(on,off):
    f=np.minimum(cv2.GaussianBlur(on,(3,3),.8),cv2.GaussianBlur(off,(3,3),.8));f[(on+off)<2]=0
    dense=np.uint8(cv2.GaussianBlur(f,(3,3),.8)>=4)*255
    # Existing bipolar-guide representation. Occupancy preserves weak native
    # rim pixels that a count4 core may erase; no reconstructed intensity.
    outline=np.uint8(cv2.GaussianBlur(((on>0)&(off>0)).astype(np.float32),(3,3),.8)>=.5)*255
    def clean(a):
        a=cv2.morphologyEx(a,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
        return cv2.morphologyEx(a,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    return clean(dense),clean(outline)


def contours(mask):
    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    return sorted((c for c in cs if len(c)>=22 and cv2.contourArea(c)>=20),key=cv2.contourArea,reverse=True)


def seeded_outline(on,off,seed_mask):
    """Only current bipolar components touched by existing rotor support."""
    _,outline=masks(on,off)
    count,labels=cv2.connectedComponents(outline,connectivity=8)
    seeds=cv2.dilate(np.uint8(seed_mask),cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=2)>0
    touched=np.unique(labels[seeds]);touched=touched[touched>0]
    lookup=np.zeros(count,bool);lookup[touched]=True
    return lookup[labels]&((on+off)>0)


def pool_arcs(c):
    p=c.reshape(-1,2);cuts=notches(c);ids=sorted(a['index'] for a in cuts);out=[]
    trials=0
    if len(ids)<2:return out,cuts,trials
    # Bounded arc adjacency: concave endpoints may bound one convex disk arc
    # while the complementary path contains a leg or a third object.
    pairs=sorted(combinations(ids,2),key=lambda ij:-min(ij[1]-ij[0],len(p)-(ij[1]-ij[0])))[:12]
    for i,j in pairs:
        for a,b in [(i,j),(j,i+len(p))]:
            segment=p[np.arange(a+3,b-2)%len(p)]
            if len(segment)<22:continue
            trials+=1;fit=fit_arc_fast(segment,max(np.ptp(p,axis=0)))
            if fit is None or fit['parameters'][3]/fit['parameters'][2]<=.16:continue
            if any(np.linalg.norm(fit['parameters'][:2]-f['parameters'][:2])<3 and
                   abs(np.log(fit['parameters'][2]/f['parameters'][2]))<.15 for f in out):continue
            out.append(fit)
    return out,cuts,trials


def pair_score(parent,children,occupancy):
    a,b=[f['parameters'] for f in children];major=2*np.array([a[2],b[2]])
    gap=abs((np.rad2deg(a[4]-b[4])+90)%180-90)
    if major.max()/major.min()>PARAMS['maximum_size_ratio'] or gap>PARAMS['maximum_pair_angle_deg']:return None
    if np.linalg.norm(a[:2]-b[:2])<PARAMS['minimum_center_separation']*major.max():return None
    p=parent.reshape(-1,2);single=_fit(p)
    if single is None:return None
    first=abs(_distance(p,single));second=np.minimum(abs(_distance(p,a)),abs(_distance(p,b)))
    support=float(np.mean(second<=1.8))
    # The parent may also contain a leg/airframe path: require independent
    # current child arcs, not that two ellipses explain every parent pixel.
    sa,sb=[set(map(tuple,np.rint(f['points']).astype(int))) for f in children]
    overlap=len(sa&sb)/max(1,min(len(sa),len(sb)))
    if overlap>.20:return None
    inflation=float(max(single[2]/np.mean([a[2],b[2]]),single[3]/np.mean([a[3],b[3]])))
    if inflation<1.6:return None
    direction=np.array([np.cos(single[4]),np.sin(single[4])])
    supported_points=np.concatenate([f['points'] for f in children])
    span=float(np.ptp(supported_points@direction)/max(np.ptp(p@direction),1.))
    # A pair of small leg/body arcs cannot replace an extended rotor assembly.
    # Same60% supported-data convention as each child fit.
    if span<.60:return None
    union=np.zeros_like(occupancy);cover=[]
    for q in (a,b):
        part=np.zeros_like(occupancy);cv2.ellipse(part,tuple(np.rint(q[:2]).astype(int)),tuple(np.maximum(1,np.rint(q[2:4]).astype(int))),float(np.rad2deg(q[4])),0,360,1,-1)
        fraction=float(np.sum((part>0)&(occupancy>0))/max(1,np.sum(part>0)))
        if fraction<.60:return None
        cover.append(fraction);union|=part
    area=int(np.sum((union>0)&(occupancy>0)))
    return dict(boundary_support=support,parent_axis_inflation=inflation,child_arc_overlap=overlap,parent_axis_span_fraction=span,child_occupancy_fraction=cover,
                explained_native_area=area,angle_gap_deg=float(gap))


def recover(on,off,existing,offset,reconsider_small_cores=False,selection='native_area'):
    if selection not in ('native_area','boundary_likelihood'):raise ValueError('Unknown arc pair selection')
    dense,outline=masks(on,off);parents=contours(outline)[:3];cores=contours(dense)[:6]
    rotors=list(existing);audit=[];total_trials=0
    for parent in parents:
        measured_inside=[r for r in existing if (r.get('mode')=='ellipse' or r.get('arc_center_qualified'))
                         and r.get('ratio',0)>.16 and cv2.pointPolygonTest(parent,(float(r['cx']-offset[0]),float(r['cy']-offset[1])),False)>=0]
        explained=None;covered=0
        if len(measured_inside)>=2:
            if reconsider_small_cores:
                region=np.zeros_like(outline);cv2.drawContours(region,[parent],-1,1,-1)
                occupied=(region>0)&(outline>0);union=np.zeros_like(outline)
                for r in measured_inside:
                    cv2.ellipse(union,tuple(np.rint([r['cx']-offset[0],r['cy']-offset[1]]).astype(int)),
                        tuple(np.maximum(1,np.rint([r['major']/2,r['minor']/2]).astype(int))),r['angle'],0,360,1,-1)
                covered=int(np.sum(occupied&(union>0)));explained=covered/max(1,int(occupied.sum()))
            # Two dense activity cores do not necessarily cover two rotor
            # rims. Reuse the existing60% supported-data convention rather
            # than accepting the mere count of individually fitted conics.
            if not reconsider_small_cores or explained>=.60:
                audit.append(dict(accepted=False,reason='two_current_independent_arcs_already_resolve_parent',
                    current_parent_support_explained=explained));continue
        arcs,cuts,trials=pool_arcs(parent);total_trials+=trials
        # Current dense-core arcs often give the unobscured lower rim; the
        # occupancy path supplies the upper rim erased by the count floor.
        for core in cores:
            if cv2.pointPolygonTest(parent,tuple(map(float,core.reshape(-1,2).mean(0))),False)<0:continue
            extra,_,tr=pool_arcs(core);arcs.extend(extra);total_trials+=tr
        best=None
        for children in combinations(arcs,2):
            proof=pair_score(parent,children,outline)
            if proof is None:continue
            score=proof['explained_native_area']
            if selection=='boundary_likelihood':
                # Foreground ENCLOSED by an ellipse is not evidence that its
                # boundary is correct. Large inflated candidates can win an
                # area ranking even when the measured rim supports another
                # admissible pair. Compare the SAME original contour using
                # a capped residual; unrelated legs remain bounded outliers.
                residual=np.minimum(*[np.abs(_distance(parent.reshape(-1,2),f['parameters'])) for f in children])
                score=-float(np.mean(np.minimum(residual**2,1.8**2)))
                proof['capped_parent_boundary_mse_px2']=-score
            if best is None or score>best[0]:best=(score,children,proof)
        if best is not None and len(measured_inside)>=2 and reconsider_small_cores and best[2]['explained_native_area']<=covered:
            best=None
        record=dict(notches=len(cuts),arc_candidates=len(arcs),trials=trials,accepted=best is not None,
            current_parent_support_explained=explained,reconsidered_current_core_pair=bool(explained is not None and explained<.60))
        if best is None:audit.append(record);continue
        _,children,proof=best;record.update(proof)
        rotors=[r for r in rotors if cv2.pointPolygonTest(parent,(float(r['cx']-offset[0]),float(r['cy']-offset[1])),False)<0]
        for f in children:
            q=f['parameters'];center=q[:2]+offset
            extent=np.sqrt(np.array([[np.cos(q[4])**2,np.sin(q[4])**2],
                [np.sin(q[4])**2,np.cos(q[4])**2]])@(q[2:4]**2))
            clipped=bool(np.any(center-extent<0) or np.any(center+extent>=[1280,720])) if reconsider_small_cores else False
            rotors.append(dict(cx=float(center[0]),cy=float(center[1]),major=float(2*q[2]),minor=float(2*q[3]),ratio=float(q[3]/q[2]),angle=float(np.rad2deg(q[4])),
                mode='arc',inferred_shape=True,current_observation=True,curvature_validated=True,arc_center_qualified=True,
                fov_clipped=clipped,center_sigma_px=f['sigma'],axis_angle_sigma_deg=f['angle_sigma'],fit_condition=f['condition'],
                fit_residual_px=f['rms'],visible_arc_fraction=f['coverage'],support_fraction=f['support_fraction'],geometry_quality=.6,
                curvature_source='native_bipolar_outline_and_dense_core_open_arcs',no_cut_chord=True))
        record['child_centers']=[(f['parameters'][:2]+offset).tolist() for f in children];audit.append(record)
    return rotors,dict(parents=audit,accepted_pairs=sum(a['accepted'] for a in audit),arc_fit_trials=total_trials,
        outline_is_periodicity_measurement=False,count_floor_unchanged=4,new_outline_is_bipolar_occupancy=True,
        pair_selection=selection)
