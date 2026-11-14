"""Reuse the frozen stage4 affine skeleton to aid stage33 rotor association."""
from pathlib import Path
import itertools,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage4'),str(ROOT/'research/stage33')]
import tracker as skeleton
from slots33 import ReplaceStale,RotorSensing,PaperConfig,geometry_record

PARAMS=dict(skeleton_parameters=skeleton.PARAMS,body_current_expiry_s=.16,body_margin_fraction=.25,
    maximum_shape_change_gate_factor=1.7,shape_fit_error_gate_factor=.5,min_update_flatness=.08,
    meaning='Reuse existing4 least-squares affine shape and33 unmatched lifecycle. Body box supplies only current common translation and candidate gating, not true center/pose. Joint accepted current observations may retain their track IDs; never synthesize rotor measurements or frequencies.',
    limits='No mechanical motor identity truth in real data. Temporal layout can support continuity but cannot resolve symmetry/edge-on ambiguity. No reset suppression without accepted current measurements.')

class LayoutSlots(ReplaceStale):
    def __init__(self,cfg):
        super().__init__(cfg);self.A=None;self.ids=None;self.layout_t=None;self.last_full_t=None
        self.body_at_layout=None;self.pending=None;self.streak=0

    def inside(self,rotors,body,t):
        if not body or not body.get('detected') or t-body['t']>PARAMS['body_current_expiry_s']:return False
        b=np.array(body['bbox'],float);m=(b[2:]-b[:2])*PARAMS['body_margin_fraction']
        return all(np.all(np.array([r['cx'],r['cy']])>=b[:2]-m) and np.all(np.array([r['cx'],r['cy']])<=b[2:]+m) for r in rotors)

    def step(self,observation,history,t,body=None):
        current=observation.get('rotors',[]);lookup={q.id:q for q in self.tracks if t-q.seen<=self.cfg.track_expiry_s}
        if self.A is not None and (t-self.last_full_t>skeleton.PARAMS['support_expiry_s'] or any(i not in lookup for i in self.ids)):
            self.A=None;self.ids=None;self.pending=None;self.streak=0
        diag=dict(layout_ready=self.A is not None,joint_association=False,reason='not_ready',initializing_streak=self.streak)
        overrides=[];sentinel=object()
        if self.A is not None and len(current)==4 and self.inside(current,body,t):
            predicted=skeleton.X@self.A.T
            if self.body_at_layout is not None:
                predicted+=np.array(body['center'])-self.body_at_layout
            else:
                dt=t-self.layout_t;predicted+=np.mean([lookup[i].velocity for i in self.ids],axis=0)*dt
            span=float(np.linalg.norm(np.ptp(predicted,axis=0)));gate=max(10.,span*skeleton.PARAMS['association_fraction'])
            points=np.array([[r['cx'],r['cy']] for r in current]);best=None
            valid,_=skeleton.initialization(current)
            if valid is not None:
                for order in itertools.permutations(range(4)):
                    ordered=points[list(order)];A=skeleton.fit_affine(np.arange(4),ordered);fitted=skeleton.X@A.T
                    error=float(np.mean(np.linalg.norm(fitted-ordered,axis=1)));change=float(np.max(np.linalg.norm(fitted-predicted,axis=1)))
                    # Reject reflection across a collapsed view; do not exchange
                    # opposite physical motors merely to minimize image distance.
                    determinant=float(np.linalg.det(A[:,:2])*np.linalg.det(self.A[:,:2]))
                    if error>=gate*PARAMS['shape_fit_error_gate_factor'] or change>=gate*PARAMS['maximum_shape_change_gate_factor'] or determinant<=0:continue
                    loss=float(np.mean(np.sum((ordered-predicted)**2,axis=1)))
                    if best is None or loss<best[0]:best=(loss,order,A,error,change)
            if best is not None:
                loss,order,A,error,change=best
                # Base association sees the accepted joint correspondence; all
                # measured rotor dictionaries and original update arithmetic stay.
                for slot,index in enumerate(order):
                    q=lookup[self.ids[slot]];previous=q.__dict__.get('position',sentinel);point=points[index].copy()
                    overrides.append((q,previous));q.position=lambda when,p=point:p.copy()
                diag.update(joint_association=True,reason='current_four_joint_fit',cost=loss,fit_error_px=error,max_change_px=change,order=list(order))
            else:diag['reason']='current_shape_or_motion_rejected'
        elif self.A is not None:diag['reason']='needs_four_current_and_body'
        try:result=super().step(observation,history,t)
        finally:
            for q,previous in overrides:
                if previous is sentinel:del q.__dict__['position']
                else:q.position=previous
        visible=[q for q in self.tracks if abs(q.seen-t)<1e-6]
        init,reason=skeleton.initialization([q.rotor for q in visible])
        qualified=init is not None and self.inside([q.rotor for q in visible],body,t)
        if qualified:
            pts,A,diam=init;center=pts.mean(0)
            if self.A is None:
                self.streak=self.streak+1 if self.pending is not None and np.linalg.norm(center-self.pending)<50 else 1
                self.pending=center
                if self.streak>=skeleton.PARAMS['init_frames']:
                    indices=[]
                    for point in pts:indices.append(min(visible,key=lambda q:np.linalg.norm(np.array([q.rotor['cx'],q.rotor['cy']])-point)).id)
                    self.ids=indices;self.A=A;diag['initialized_now']=True
            elif set(q.id for q in visible)==set(self.ids):
                mapping={q.id:q for q in visible};points=np.array([[mapping[i].rotor['cx'],mapping[i].rotor['cy']] for i in self.ids])
                new_A=skeleton.fit_affine(np.arange(4),points)
                if np.linalg.det(new_A[:,:2])*np.linalg.det(self.A[:,:2])>0:self.A=new_A
                else:self.A=None;self.ids=None;self.streak=0
            else:self.A=None;self.ids=None;self.streak=0
            if self.A is not None:
                self.layout_t=t;self.last_full_t=t;self.body_at_layout=np.array(body['center']) if body else None
        elif self.A is None:self.streak=0;self.pending=None
        diag.update(layout_ready_after=self.A is not None,initializing_streak=self.streak,initialization_reason=reason,
            model_ids=None if self.ids is None else list(self.ids),predicted_points_are_measurements=False)
        self.audit['layout']=diag
        return result
