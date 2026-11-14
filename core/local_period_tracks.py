"""Causal local periodic-region continuity; regions are not motor identities."""
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment


class LocalPeriodTracks:
    gap_s=.12
    log_gate=float(np.log(4./3.)-.08)

    def __init__(self):
        self.tracks={};self.serial=0;self.epoch=0;self.last_t=None;self.last_evidence=None
        self.log_relative=0.

    def step(self,t,local):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing source cutoffs required')
        self.last_t=t
        if self.last_evidence is not None and t-self.last_evidence>self.gap_s+1e-8:
            self.tracks.clear();self.log_relative=0.;self.epoch+=1;self.last_evidence=None
        self.tracks={k:r for k,r in self.tracks.items() if t-r['t']<=self.gap_s+1e-8}
        regions=[]
        for r in local.get('regions',[]):
            f=r.get('frequency_hz');cell=r.get('cell')
            if f is None or cell is None or not 60<=f<=1200:continue
            candidates=np.unique([f]+[v for v in r.get('other_supported_frequencies',[]) if 60<=v<=1200])
            regions.append(dict(cell=np.asarray(cell,float),candidates=candidates,original=float(f)))
        ids=sorted(self.tracks);matches=[];assigned=set();continued=[]
        if ids and regions:
            distance=np.array([[np.linalg.norm(self.tracks[k]['cell']-r['cell']) for r in regions] for k in ids])
            cost=distance.copy();chosen={}
            for i,k in enumerate(ids):
                old=self.tracks[k]
                for j,r in enumerate(regions):
                    errors=abs(np.log(r['candidates']/old['f']));index=int(np.argmin(errors))
                    chosen[i,j]=(float(r['candidates'][index]),float(errors[index]))
                    if distance[i,j]>1.5 or errors[index]>self.log_gate:cost[i,j]=1e6
                    else:cost[i,j]+=.25*errors[index]/self.log_gate
            ii,jj=linear_sum_assignment(cost)
            for i,j in zip(ii,jj):
                if cost[i,j]>=1e6:continue
                k=ids[i];old=self.tracks[k];r=regions[j];f,error=chosen[i,j]
                increment=2*float(np.log(f/old['f']))
                matches.append(dict(track_id=k,cell=r['cell'].tolist(),frequency_hz=f,
                    previous_frequency_hz=old['f'],log_squared_increment=increment,
                    current_alternative_used=f!=r['original'],candidate_count=len(r['candidates']),
                    harmonic_ambiguity_unresolved=len(r['candidates'])>1))
                continued.append(increment);assigned.add(j)
                self.tracks[k]=dict(cell=r['cell'],f=f,t=float(t))
        for j,r in enumerate(regions):
            if j in assigned:continue
            self.serial+=1;self.tracks[self.serial]=dict(cell=r['cell'],f=r['original'],t=float(t))
        qualified=len(continued)>=2
        spread=float(np.median(abs(np.asarray(continued)-np.median(continued)))) if continued else None
        # The existing .08 frequency dispersion bound, squared-frequency units.
        if qualified and spread>2*.08:qualified=False
        if not continued and regions and self.last_evidence is not None:
            self.log_relative=0.;self.epoch+=1;self.last_evidence=None
        if qualified:
            self.log_relative+=float(np.median(continued));self.last_evidence=float(t)
        frequency=300.*np.exp(.5*self.log_relative) if qualified else None
        if frequency is not None and not 60<=frequency<=1200:
            # This is a normalized gauge, not an absolute mechanical RPM.
            # Restart an unsupported gauge instead of remaining permanently
            # invalid while refreshing last_evidence on every rejected frame.
            qualified=False;frequency=None;self.tracks.clear();self.log_relative=0.
            self.epoch+=1;self.last_evidence=None
        return dict(qualified=qualified,frequency_hz=None if frequency is None else float(frequency),
            selection_source='continued_local_regions:'+str(self.epoch),
            normalized_change_proxy=True,mechanical_RPM_measured=False,metric_thrust_measured=False,
            same_source_spatial_tracking_confirmed=qualified,
            current_regions=len(regions),continued_regions=len(continued),matches=matches,
            source_epoch=self.epoch,relative_squared_proxy=float(np.exp(self.log_relative)),
            current_increment_MAD=spread,current_only=True,regions_are_not_rotors=True,
            missing_fundamental_invented=False,source_count_changes_not_thrust_steps=True,
            reason='current_region_relative_change' if qualified else 'need_two_current_continued_regions')


class InnovationUnknownForce:
    """One2D force state; not another position/velocity trajectory predictor."""
    def __init__(self):
        self.d=None;self.P=np.eye(2)*1e6;self.components={};self.k=200.;self.t=None

    def step(self,t,a,components,n,gain,covariance,zero_input=False):
        a=np.asarray(a,float);raw={k:float(v)*np.asarray(n,float) for k,v in components.items()} if n is not None else {}
        current={k:(np.zeros(2) if zero_input else v) for k,v in raw.items()}
        u=np.sum(list(current.values()),axis=0) if current else np.zeros(2)
        first=self.d is None;dt=0. if self.t is None else float(t-self.t)
        if dt<0:raise ValueError('Causal force timestamps required')
        if first:self.d=a-gain*u
        # New/lost observations change ownership, not physical acceleration.
        transfer=np.zeros(2)
        for k in set(current)-set(self.components):transfer+=current[k]
        for k in set(self.components)-set(current):transfer-=self.components[k]
        if not first:self.d-=self.k*transfer
        if not first:self.d-=(gain-self.k)*u
        self.P+=np.eye(2)*(2*1000.**2/.35)*max(dt,0.)
        R=np.asarray(covariance,float);S=self.P+R;innovation=a-(self.d+gain*u)
        # Joseph covariance update; raw/current motion is not assumed exact.
        K=np.linalg.solve(S,self.P).T
        self.d+=K@innovation;I=np.eye(2)-K;self.P=I@self.P@I.T+K@R@K.T
        self.P=.5*(self.P+self.P.T);net=self.d+gain*u
        norm=float(np.linalg.norm(net))
        if norm>3000.:net*=3000./norm
        self.components=current;self.k=float(gain);self.t=float(t)
        return net,dict(unknown=self.d.tolist(),unknown_covariance_proxy=self.P.tolist(),
            innovation=innovation.tolist(),transfer_u=transfer.tolist(),known_current_u=u.tolist(),
            current_known_changes_not_algebraically_cancelled=True,unknown_can_update_every_frame=True,
            net_acceleration_cap_applied=norm>3000.,extra_position_filters=0)
