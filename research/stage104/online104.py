"""Add planar-center forecasts while preserving every existing93 forecast."""
from pathlib import Path
import copy,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'research/stage93'),str(ROOT/'research/stage23'),str(ROOT/'research/stage77')]
from online93 import OnlineExtras as Previous,Kinematic
from robust23 import Predictor as History
from separated77 import SeparatedBank
from increment72 import Gain
from rigid104 import PlanarCenter,smooth

NAMES=['rigid_center_kf','rigid_center_history','rigid_center_view_force','rigid_center_constant_force','rigid_center_zero_force']


class OnlineExtras(Previous):
    def __init__(self,w,h,context):
        super().__init__(w,h);self.context=context;self.center104=PlanarCenter();self.kinematic104=Kinematic()
        self.history104=History();self.last_seen104=None;self.reset_force()
    def reset_force(self):
        self.banks104={k:SeparatedBank('mmae') for k in ['event','constant','zero']}
        self.gains104={k:Gain() for k in ['event','constant']};self.previous104={}
    def measure(self,t,events,body):
        result=super().measure(t,events,body)
        assert abs(self.context['t']-t)<1e-8
        self.obs104,self.diag104=self.center104.step(t,body,self.context['sensed']['rotors'],self.current['body'])
        return result
    def forecast(self,q,obs,rotor_quality):
        q=super().forecast(q,obs,rotor_quality);return self.add_forecasts(q,rotor_quality)
    def add_forecasts(self,q,rotor_quality):
        t=q['t'];obs=self.obs104;z=obs.get('center') if obs.get('detected') else None
        if self.last_seen104 is not None and t-self.last_seen104>.5:
            self.reset_force()
        if z is not None:self.last_seen104=t
        k=self.kinematic104.step(t,z);history,_=self.history104.step(t,z)
        # Current normal sources are blended by measured view/quality; a body
        # line is not allowed to become a substitute rotor frequency.
        inc=q['increment_diagnostic'];normal=None
        nd=inc['direction'];rotor=q['feature_diagnostic']['models']['adaptive_axis_only']['normal']
        body=self.current['body'];rv=None
        if q['feature_diagnostic'].get('axis_age_s') is not None and abs(q['feature_diagnostic']['axis_age_s'])<1e-8 and rotor is not None:
            rv=np.asarray(rotor[:2],float)
            if np.linalg.norm(rv)>.1:rv/=np.linalg.norm(rv)
            else:rv=None
        bv=np.asarray(body['normal_axis_xy'],float) if body.get('valid') and body.get('confidence',0)>.45 and nd['current'] and nd['source']=='confirmed_body_line' else None
        if rv is not None and bv is not None and rv@bv<0:rv=-rv
        if rv is not None and rv[1]>0:rv=-rv
        if bv is not None and bv[1]>0:bv=-bv
        face_weight=self.diag104['measurement'].get('view_weight',0.) if self.diag104['current'] else 0.
        if bv is not None:
            normal=bv if rv is None else face_weight*rv+(1.-face_weight)*bv
        elif rv is not None:normal=rv
        if normal is not None:normal/=max(np.linalg.norm(normal),1e-9)
        f=inc['features']['periodic'];gd={};coeff={};fresh=False
        if history['valid']:
            base=history['coefficients']['trusted_acc'];acc=np.asarray(base['acceleration'])
            for kind,model in self.gains104.items():
                active=bool(normal is not None and f['active'])
                current=bool(active and f['current'] and (bv is not None or (rv is not None and rotor_quality>.45)))
                feedback=bool(z is not None and obs.get('confidence',0)>.5)
                _,gd[kind]=model.step(t,base,normal,1. if kind=='constant' else f['ratio'],active,current,feedback,z)
            past=self.previous104.get('event');fresh=bool(past is not None and past['updates']>=10 and gd['event']['active'] and
                gd['event']['delta'] is not None and current)
            for kind,bank in self.banks104.items():
                key='constant' if kind=='constant' else 'event';old=self.previous104.get(key)
                correction=np.zeros(2)
                if fresh and kind!='zero' and old is not None and gd[key]['delta'] is not None:
                    correction=old['gain_px_s2']*np.asarray(gd[key]['delta'])
                bank.step(t,z,np.zeros(2),acc+correction,fresh)
                coeff[kind]=copy.deepcopy(base)
                if fresh and bank.bank is not None:coeff[kind],_=bank.coefficient()
        q=dict(q,coefficients=dict(q['coefficients']),predictions=dict(q['predictions']))
        if q['valid'] and history['valid']:
            q['coefficients']['rigid_center_kf']=k
            q['coefficients']['rigid_center_history']=history['coefficients']['trusted_acc']
            for kind,name in [('event','rigid_center_view_force'),('constant','rigid_center_constant_force'),('zero','rigid_center_zero_force')]:
                q['coefficients'][name]=coeff[kind]
            from hybrid74 import prior
            for name in NAMES:q['predictions'][name]={str(h):prior.evaluate(q['coefficients'][name],h) for h in [.1,.2,.4]}
        q['rigid_center_diagnostic']=copy.deepcopy(self.diag104)
        q['rigid_center_diagnostic'].update(view_face_weight=face_weight,view_direction=None if normal is None else normal.tolist(),
            force_current=fresh,physics_gain_reused_from_old_branch=False,independent_force_recalibrated=True,gains=gd,
            history_valid=history['valid'])
        self.previous104=copy.deepcopy(gd)
        return q
