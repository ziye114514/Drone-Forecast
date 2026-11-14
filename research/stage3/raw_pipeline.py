"""Three-arm RAW replay; identical positions/RPM, different normal measurements."""
import copy
import numpy as np
from measurement import audit_ellipse,ConsensusAxis,ROOT
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_bank import PaperForecastBank

class AuditedPerception(PaperPerception):
    def _candidates(self,contours,activity_map,t,b,thin_only=False):
        found=super()._candidates(contours,activity_map,t,b,thin_only)
        for r in found:
            best=None
            for c in contours:
                if len(c)<6:continue
                points=c.reshape(-1,2).astype(float)*b
                low=points.min(axis=0);high=points.max(axis=0)
                distance=np.linalg.norm((low+high)/2-r['center'])
                if best is None or distance<best[0]:best=(distance,points)
            if best is None or best[0]>max(10.,r['major']*.7):
                r['measurement_audit']=dict(accepted=False,reason='no_contour_support',normal_sigma_deg=30.,quality_multiplier=0.)
            else:
                p=np.r_[r['center'],r['major']/2,r['minor']/2,np.deg2rad(r['angle'])]
                r['measurement_audit']=audit_ellipse(best[1],p,self.paper_cfg.camera_matrix)
        return found

    def detect(self,events,t):
        o,total=super().detect(events,t)
        if o['detected']:
            for serialized,measured in zip(o['rotors'],self.previous_rotors):
                serialized['measurement_audit']=measured.get('measurement_audit',dict(accepted=False,reason='no_audit',quality_multiplier=0.))
        return o,total

class CachedAxis:
    def __init__(self,axis):self.axis=axis;self.prepared_t=None
    @property
    def differential_gain(self):
        # Compatibility with the frozen bank's reporting field. This axis
        # does not claim calibrated body-motor correspondence or learn it.
        return 0.
    def __getattr__(self,name):return getattr(self.axis,name)
    def advance(self,*args):self.prepared_t=args[0];return self.axis.step(*args)
    def step(self,t,*args):
        assert t==self.prepared_t
        return self.axis.n

class ConsensusBank(PaperForecastBank):
    def __init__(self,cfg,audit=False):
        self.use_audit=audit;super().__init__(cfg)
        self.axes={name:CachedAxis(ConsensusAxis(cfg)) for name in ('paper_nominal','paper_adaptive')}

    def step(self,observation,sensing):
        if self.last_seen is not None and observation['t']-self.last_seen>.5:self.__init__(self.cfg,self.use_audit)
        rotors=copy.deepcopy([r for r in sensing['rotors'] if r['observed_now']])
        for r in rotors:
            if not self.use_audit:r.pop('measurement_audit',None)
        for axis in self.axes.values():axis.advance(observation['t'],rotors,sensing['differential_proxy'],True)
        o=copy.deepcopy(observation);accepted=all(a.diagnostics['accepted'] for a in self.axes.values())
        q=1.
        if self.use_audit:
            good=[r.get('measurement_audit',{}).get('quality_multiplier',0.) for r in rotors]
            q=float(np.mean(good)) if good else 0.
        # A past propagated normal is not permission to learn from a newly
        # rejected observation. No accepted current consensus means CV fallback.
        o['geometry_quality']=o.get('geometry_quality',0)*q if accepted else 0.
        result=super().step(o,sensing)
        if result is not None:
            result['normal_measurements']={name:a.diagnostics for name,a in self.axes.items()}
            result['normal_admission_quality']=o['geometry_quality']
            if not accepted:
                assert not result['physics_active']
                assert not self.models['paper_adaptive'].update_allowed
        return result
