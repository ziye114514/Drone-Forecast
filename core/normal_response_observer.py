"""Unknown force with a learnable component constrained by current rotor plane.

Keep the two-force partition a=k*u+d, but model d=e+h*n. The unmeasured
normal-response h is learned from position innovations; external e is not
rotated with the plane. In coordinates[d,h], a new measured normal changes d
by h*delta_n after the position update, never backwards over an elapsed step.
No nominal thrust amplitude is injected: h starts at zero with the existing
1000px/s² weak response prior. Constant normal cannot identify h.
"""
import numpy as np
from layout_reference_observer import LayoutReferenceObserver
from axis_amplitude_bridge import current_axis_qualified
from normal_turn_forecast import NormalTurnForecast


class NormalResponseObserver(LayoutReferenceObserver):
    state_dim=12
    response_index=11
    requires_current_structure=True
    preserve_unmeasured_projection=False

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.normal=None;self.normal_t=None;self.normal_observation_kind=None
        self.normal_trend=NormalTurnForecast();self.current_structure={};self.current_motion={}

    def _initialize(self,z):
        super()._initialize(z);self.P[self.response_index,self.response_index]=1e6

    def _set_input(self,u,reanchor=False,transfer_delta=None,transfer_cov=None):
        delta=np.asarray(u,float)-self.u if reanchor else None if transfer_delta is None else np.asarray(transfer_delta,float)
        super()._set_input(u,reanchor,transfer_delta,transfer_cov)
        if delta is not None and self.normal is not None:
            direction=self.normal/max(float(self.normal@self.normal),1e-12)
            projected=float(direction@delta);k=self.x[self.gain_index]
            self.x[self.response_index]-=k*projected
            J=np.eye(self.state_dim);J[self.response_index,self.gain_index]=-projected
            self.P=J@self.P@J.T
            if transfer_cov is not None:
                moment=k*k+self.P[self.gain_index,self.gain_index]
                cross=moment*np.asarray(transfer_cov)@direction
                self.P[4:6,self.response_index]+=cross;self.P[self.response_index,4:6]+=cross
                self.P[self.response_index,self.response_index]+=moment*float(direction@transfer_cov@direction)

    def _transport_normal(self,n,projection_scale):
        delta=projection_scale*n-self.normal;h=self.x[self.response_index]
        J=np.eye(self.state_dim);J[4:6,self.response_index]=delta
        J[self.response_index,self.response_index]=projection_scale
        self.x[4:6]+=h*delta;self.x[self.response_index]*=projection_scale;self.P=J@self.P@J.T
        h=self.x[self.response_index];tangent=np.array([-n[1],n[0]])
        self.P[4:6,4:6]+=(h*h+self.P[self.response_index,self.response_index])*np.deg2rad(5)**2*np.outer(tangent,tangent)
        return bool(np.any(delta))

    def step(self,t,body,u,*args,**kwargs):
        structure=kwargs.pop('structure',{}) or {};motion=kwargs.get('motion') or {}
        was_two=self.two_point_initialized
        c,valid,f,s=super().step(t,body,u,*args,**kwargs)
        if self.two_point_initialized and not was_two:self.P[self.response_index,self.response_index]=getattr(self,'normal_variance_after_two_point',1e6)
        n=np.asarray(structure.get('axis_xy') if structure.get('axis_xy') is not None else [],float)
        good=bool(c is not None and current_axis_qualified(structure) and n.shape==(2,) and np.isfinite(n).all() and np.linalg.norm(n)>1e-6)
        used=False;reanchor=False;projection_scale=1.
        kind='current_conic_projection' if structure.get('pose_source') in ('current_qualified_visible_conics','current_metric_template_relock') or (structure.get('mode') in ('four','three','partial_two') and structure.get('continuous',{}).get('updated')) else 'azimuth_or_transported_projection_prior'
        if good:
            if n[1]>0:n=-n
            if self.normal is not None:
                gap=t-self.normal_t
                angle=np.arccos(np.clip(self.normal@n/max(np.linalg.norm(self.normal)*np.linalg.norm(n),1e-12),-1.,1.))
                reanchor=bool(gap>.12+1e-8 or angle>np.deg2rad(30))
                if self.preserve_unmeasured_projection and (kind!='current_conic_projection' or self.normal_observation_kind!='current_conic_projection'):
                    projection_scale=float(np.linalg.norm(self.normal)/np.linalg.norm(n))
                if not reanchor:
                    used=self._transport_normal(n,projection_scale)
                elif projection_scale!=1.:
                    J=np.eye(self.state_dim);J[self.response_index,self.response_index]=projection_scale
                    self.x[self.response_index]*=projection_scale;self.P=J@self.P@J.T
            self.normal=n.copy();self.normal_t=float(t);self.normal_observation_kind=kind
        if c is not None:
            a=self.x[4:6]+self.x[self.gain_index]*self.u;length=float(np.linalg.norm(a))
            if length>3000:a=a*3000/length
            c=dict(c,acceleration=a.tolist());f=dict(f,acceleration_px_s2=a.tolist(),disturbance_px_s2=self.x[4:6].tolist())
            h=float(self.x[self.response_index]);component=np.zeros(2) if self.normal is None else h*self.normal
            f.update(unobserved_normal_response_px_s2=component.tolist(),unobserved_normal_response_gain=h,
                unobserved_normal_response_sigma=float(np.sqrt(max(0.,self.P[self.response_index,self.response_index]))),
                external_unknown_not_rotated=(self.x[4:6]-component).tolist(),
                normal_response_is_measured_thrust=False,unknown_force_has_directional_structure=True)
        estimate=motion.get('estimate') or {};dt=motion.get('dt_s')
        flow=dict(valid=False)
        if dt and motion.get('current_observation') and not estimate.get('peak_at_search_boundary') and estimate.get('score',0)>0:
            flow=dict(valid=True,velocity=[estimate['dx']/dt,estimate['dy']/dt])
        trend=self.normal_trend.observe(t,structure,c,dict(observation=flow))
        turn=dict(applied=bool(good and trend.get('qualified') and c is not None),trend=trend,
            estimated_unobserved_normal_part_rotated=True,external_unknown_rotated=False)
        if turn['applied']:
            visible=np.zeros(2) if f.get('input_u') is None else f['effective_gain_px_s2']*np.asarray(f['input_u'])
            f['normal_turn_vector_px_s2']=(visible+np.asarray(f['unobserved_normal_response_px_s2'])).tolist()
        s.update(normal_response_state_updated=used,normal_reference_reanchored_without_impulse=reanchor,
            normal_response_starts_at_zero=True,normal_turn_forecast=turn)
        if self.preserve_unmeasured_projection:
            s['normal_projection_qualification']=dict(kind=kind,hidden_amplitude_coordinate_scale=projection_scale,
                unknown_projection_change_not_a_measured_force=True,azimuth_change_preserved=True,
                joint_disturbance_amplitude_covariance_transformed=True)
        return c,valid,f,s
