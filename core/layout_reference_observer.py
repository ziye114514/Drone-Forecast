"""One native-rotor track with an uncertain layout-to-output reference offset.

The first layout centre initializes only b = layout - p. Its anticorrelation
with p is retained, so it cannot teleport position or act as a second absolute
anchor. Later layout observations constrain p+b. Native displacement and layout
measurements use fixed symmetric information weighting, not an independence
claim about observations from the same events. Covariances remain proxies.
"""
import numpy as np
from coherent_gap_observer import ShortGapObserver


def layout_packet(structure,geometry,allow_line=False,line_update_allowed=True):
    c=structure.get('current_geometric_center')
    if c is None or not structure.get('joint_center_is_current_measurement'):
        if not allow_line:return None
        side=structure.get('axis_support') or {};bounds=geometry.get('roi')
        if not side.get('valid') or side.get('confidence',0)<.45 or structure.get('mode') not in ('axis_only','edge_on'):return None
        if bounds is None or bounds[0]<=0 or bounds[1]<=0 or bounds[2]>=1280 or bounds[3]>=720:return None
        center=np.asarray(side.get('center'),float);scatter=np.asarray(side.get('center_multiscale_covariance_proxy_px2'),float)
        if center.shape!=(2,) or scatter.shape!=(2,2) or not np.isfinite(center).all() or not np.isfinite(scatter).all():return None
        covariance=(scatter+scatter.T)/2+np.eye(2)*36.
        return dict(center=center.tolist(),covariance_proxy=covariance.tolist(),epoch=[structure.get('geometry_epoch'),'side_line'],
            source='current_native_rotor_line_midpoint',reference_kind='side_line',update_allowed=bool(line_update_allowed),
            current_observation=True,COM_truth=False,inferred_rotor_frequency_created=False,
            inferred_hidden_hubs_are_measurements=False,center_is_visible_projection_not_mass_GT=True)
    p=np.asarray(c,float)
    if p.shape!=(2,) or not np.isfinite(p).all():return None
    # Never turn an axis or two-point model-conditioned prior into a centre.
    if structure.get('mode') not in ('four','three','relock_pending'):return None
    sigmas=[float(r.get('center_sigma_px',6.)) for r in geometry.get('rotors',[]) if r.get('current_observation',True)]
    variance=max([s*s for s in sigmas]+[36.])
    covariance=np.asarray(structure.get('model_conditioned_center_covariance',np.zeros((2,2))),float)
    if covariance.shape!=(2,2) or not np.isfinite(covariance).all():return None
    covariance=(covariance+covariance.T)*.5+np.eye(2)*variance
    if np.linalg.eigvalsh(covariance).min()<=0:return None
    return dict(center=p.tolist(),covariance_proxy=covariance.tolist(),epoch=structure.get('geometry_epoch'),
        source=structure.get('center_source'),current_observation=True,COM_truth=False,
        inferred_rotor_frequency_created=False)


class LayoutReferenceObserver(ShortGapObserver):
    state_dim=11
    geometry_data_mode='layout'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.layout_initialized=False;self.layout_t=None;self.layout_epoch=None
        self.packet=None;self.layout_used=False;self.layout_gate=None
        self.layout_anchor=None;self.native_anchor=None;self.body_now=None

    def _rebase_observation_reference(self,shift):
        self.x[:2]+=shift;self.x[7:9]+=shift
        J=np.zeros((self.state_dim,2));J[:2]=np.eye(2);J[7:9]=np.eye(2)
        if self.layout_initialized:
            self.x[9:11]-=shift;J[9:11]=-np.eye(2)
            if self.native_anchor is not None:self.native_anchor+=shift
        self.P+=36.*J@J.T

    def _initialize_layout(self,packet,t,body):
        c=np.asarray(packet['center'],float);R=np.asarray(packet['covariance_proxy'],float)
        # b=c-p; no p/v/d/k change and no1/N false absolute precision gain.
        J=np.eye(self.state_dim);J[9:11]=0.;J[9,0]=-1.;J[10,1]=-1.
        self.x[9:11]=c-self.x[:2];self.P=J@self.P@J.T;self.P[9:11,9:11]+=R
        self.layout_initialized=True;self.layout_t=float(t);self.layout_epoch=packet.get('epoch')
        self.layout_anchor=c.copy();self.native_anchor=np.asarray(body['center'],float).copy()

    def _measurement(self,z,body,motion,lag):
        H,R,measurement,used=super()._measurement(z,body,motion,lag)
        packet=self.packet
        fresh=bool(packet is not None and self.layout_initialized and self.layout_t is not None and
            0<self.t-self.layout_t<=.4+1e-8 and packet.get('epoch')==self.layout_epoch)
        if not fresh or not self.relative_used or not packet.get('update_allowed',True):return H,R,measurement,used
        Hg=np.zeros((2,self.state_dim));Hg[:,:2]=np.eye(2);Hg[:,9:11]=np.eye(2)
        Rg=np.asarray(packet['covariance_proxy'],float);value=np.asarray(packet['center'],float)
        if self.geometry_data_mode=='native-control':
            value=self.layout_anchor+(np.asarray(body['center'],float)-self.native_anchor)
        residual=value-Hg@self.x;S=Hg@self.P@Hg.T+Rg
        self.layout_gate=float(residual@np.linalg.solve(S,residual))
        if self.layout_gate>9.21034037197618:return H,R,measurement,used
        self.layout_used=True
        # Weighted observation information is CI of the two posterior
        # information forms sharing the SAME prior. No full-state inversion,
        # and no independent-noise stack; the native factor is also diluted.
        n=len(measurement);weighted=np.zeros((n+2,n+2));weighted[:n,:n]=2*R;weighted[n:,n:]=2*Rg
        return np.vstack([H,Hg]),weighted,np.r_[measurement,value],used

    def step(self,t,body,u,*args,**kwargs):
        packet=(kwargs.get('motion') or {}).get('layout_observation')
        if (packet is not None and packet.get('reference_kind')=='side_line' and not packet.get('update_allowed',True) and
            self.layout_initialized and packet.get('epoch')!=self.layout_epoch):
            # Warming a possible side reference must not replace a previously
            # established full-layout reference or alter healthy motion.
            packet=None
        self.packet=packet;self.layout_used=False;self.layout_gate=None
        self.body_now=body
        c,valid,f,s=self._joint_step(t,body,u,*args,**kwargs)
        initialized=False
        if c is not None and packet is not None and body.get('detected') and not body.get('center_provisional'):
            expired=self.layout_t is None or t-self.layout_t>.4+1e-8 or packet.get('epoch')!=self.layout_epoch
            if not self.layout_initialized or expired:
                self._initialize_layout(packet,t,body);initialized=True
            elif self.layout_used or not packet.get('update_allowed',True):self.layout_t=float(t)
        s.update(layout_reference_initialized=initialized,layout_observation_used=self.layout_used,
            layout_innovation_chi2=self.layout_gate,layout_reference_is_uncertain_offset=True,
            layout_offset_xy=None if not self.layout_initialized else self.x[9:11].tolist(),
            output_reference='initial_native_rotor_support_anchor',layout_center_is_output_position=False,
            layout_bias_cross_covariance_preserved=True,geometry_data_mode=self.geometry_data_mode,
            geometry_and_native_information_weight=.5 if self.layout_used else None,
            first_layout_is_absolute_position_measurement=False)
        return c,valid,f,s

    def _joint_step(self,t,body,u,*args,**kwargs):
        return super().step(t,body,u,*args,**kwargs)


class NativeControlLayoutObserver(LayoutReferenceObserver):
    """Offline source-value control: same eligible geometry, native displacement."""
    geometry_data_mode='native-control'


class CoherentLayoutReferenceObserver(LayoutReferenceObserver):
    gap_correlation_s=None


class LegacyGapLayoutReferenceObserver(LayoutReferenceObserver):
    def _joint_step(self,t,body,u,*args,**kwargs):
        from relative_rotor_observer import RelativeRotorForceObserver
        return RelativeRotorForceObserver.step(self,t,body,u,*args,**kwargs)
