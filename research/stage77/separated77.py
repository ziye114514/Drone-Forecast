"""Separate filtering controls from forecast corrections; isolate mode states.

All gains, measurement noise and transition probabilities remain74. Matched
constant inputs share the EVENT branch's availability clock; a zero constant
regressor is not made artificially unavailable just because it never excites.
"""
from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage74'))
from hybrid74 import ControlledFilter,DualBank,IMMEstimator,prior
VENDOR=ROOT/'external/filterpy_mmae_upstream'
SNAPSHOT=json.loads((VENDOR/'UPSTREAM_SNAPSHOT.json').read_text('utf-8'))
for item in SNAPSHOT['files']:
    assert hashlib.sha256((VENDOR/item['path']).read_bytes()).hexdigest()==item['sha256']
_source=(VENDOR/'mmae.py').read_text('utf-8');_old='from filterpy.common import pretty_str';assert _source.count(_old)==1
_ns=dict(__name__='filterpy_mmae77')
exec(compile(_source.replace(_old,"def pretty_str(name, value): return str(name)+' = '+str(value)"),str(VENDOR/'mmae.py'),'exec'),_ns)
AuthorMMAE=_ns['MMAEFilterBank']
SPECS={f'{backend}_{control}_fresh{suffix}':dict(backend=backend,state_control=control=='state',constant=bool(suffix))
       for backend in ['imm','mmae'] for control in ['state','forecast'] for suffix in ['','_constant']}
ADDED=['rollout_refresh','history_when_missing']+list(SPECS)
PARAMS=dict(transition=[[.97,.03],[.03,.97]],initial_modes=[.9,.1],
    sigma_stable=400.,sigma_agile=800.,sigma_position=6.,minimum_updates=10,
    current_input='Both current axis support and current qualified relative periodic input required. '
                  'Original age expiry retained as diagnostic, no stale force launch.',
    fallback='Same trusted_acc when current physical input is unavailable, unchanged output coverage.',
    matched_controls='Event eligibility clock used for both event and constant branches. '
                     'Their separately learned gain/delta values remain unchanged.',
    future='Current trusted history acceleration + previous-tick gain * current feature delta. '
           'Interval prediction still uses previous history, or zero in forecast-only arms.',
    mmae_covariance='Pinned author MMAE weights/state executed. Its scalar-state zip covariance '
                    'defect is detected by a vector-state control; wrapper recomputes exact '
                    'mixture covariance and predicted mixture outside unchanged upstream source.',
    scope='Fixed structural ablations informed by76 failure audit. Known development recordings; '
          'no held-out or new mechanical RPM claims. No parameter search.')

class CompatibleFilter(ControlledFilter):
    def __init__(self,z):super().__init__(z);self.z=np.asarray(z,float).copy()
    def update(self,z,R=None,H=None):
        assert R is None and H is None
        super().update(z)
        if z is not None:self.z=np.asarray(z).copy()

def mixture(filters,weights):
    x=sum(w*f.x for w,f in zip(weights,filters));P=np.zeros_like(filters[0].P)
    for w,f in zip(weights,filters):
        d=f.x-x;P+=w*(f.P+np.outer(d,d))
    return x,(P+P.T)/2

class SeparatedBank:
    def __init__(self,backend):self.backend=backend;self.bank=None;self.last_t=None;self.future_a=np.zeros(2)
    def step(self,t,z,state_a,future_a,active):
        self.future_a=np.clip(future_a,-2500.,2500.)
        if self.bank is None:
            if z is None:return
            filters=[CompatibleFilter(z),CompatibleFilter(z)]
            self.bank=IMMEstimator(filters,[.9,.1],np.array([[.97,.03],[.03,.97]])) if self.backend=='imm' else AuthorMMAE(filters,np.array([.9,.1]),4)
            self.last_t=t;return
        dt=t-self.last_t;assert dt>0;self.last_t=t
        for j,f in enumerate(self.bank.filters):
            f.dt=dt;f.a=np.zeros(2) if j==0 else np.clip(state_a,-2500.,2500.);f.sigma=800. if j==1 and active else 400.
        if self.backend=='mmae':self.bank.p=self.bank.p@np.array([[.97,.03],[.03,.97]])
        self.bank.predict()
        if z is not None:self.bank.update(np.asarray(z,float))
        if self.backend=='mmae':
            # Author p/state operations are retained; repair only the audited
            # scalar-vs-vector covariance loop and refresh coast mixture.
            self.bank.x,self.bank.P=mixture(self.bank.filters,self.bank.p)
            self.bank.x_post=self.bank.x.copy();self.bank.P_post=self.bank.P.copy()
        assert np.isfinite(self.bank.x).all() and np.linalg.eigvalsh(self.bank.P).min()>-1e-7
    def coefficient(self):
        weights=self.bank.mu if self.backend=='imm' else self.bank.p
        saved=[f.a.copy() for f in self.bank.filters]
        self.bank.filters[0].a=np.zeros(2);self.bank.filters[1].a=self.future_a.copy()
        facade=SimpleNamespace(imm=SimpleNamespace(filters=self.bank.filters,mu=weights,x=self.bank.x,P=self.bank.P))
        try:c,d=DualBank.coefficient(facade)
        finally:
            for f,a in zip(self.bank.filters,saved):f.a=a
        d.update(backend=self.backend,state_control=saved[1].tolist(),future_control=self.future_a.tolist())
        return c,d

class SeparatedForecast:
    def __init__(self):self.reset()
    def reset(self):
        self.models={m:SeparatedBank(spec['backend']) for m,spec in SPECS.items()};self.refresh=SeparatedBank('imm')
        self.previous_history=np.zeros(2);self.previous_gain={};self.last_seen=None
    def step(self,original,obs):
        q=copy.deepcopy(original);t=q['t'];z=obs.get('center') if obs.get('detected') else None
        if self.last_seen is not None and t-self.last_seen>.5:self.reset()
        if z is not None:self.last_seen=t
        inc=q['increment_diagnostic'];event=inc['models']['increment_periodic'];gain=self.previous_gain.get('increment_periodic')
        eligible=bool(event['active'] and gain is not None and gain['updates']>=10 and event['delta'] is not None)
        fresh=bool(eligible and inc['direction']['current'] and inc['features']['periodic']['current'])
        current_history=np.asarray(q['coefficients']['trusted_acc']['acceleration']) if q['valid'] else np.zeros(2)
        diagnostic=dict(eligible=eligible,current=fresh,models={})
        for name,spec in SPECS.items():
            key='increment_periodic_constant' if spec['constant'] else 'increment_periodic'
            gd=self.previous_gain.get(key);delta=inc['models'][key]['delta'];innovation=np.zeros(2)
            if fresh and gd is not None and delta is not None:innovation=gd['gain_px_s2']*np.asarray(delta)
            old_control=self.previous_history+innovation if fresh and spec['state_control'] else np.zeros(2)
            future=current_history+innovation if fresh else np.zeros(2)
            model=self.models[name];model.step(t,z,old_control,future,fresh)
            if model.bank is not None:
                coefficient,d=model.coefficient();d.update(active=fresh,innovation_a=innovation.tolist())
                if q['valid']:
                    q['coefficients'][name]=coefficient if fresh else copy.deepcopy(q['coefficients']['trusted_acc'])
            else:d=dict(active=False,mode_probability=None)
            diagnostic['models'][name]=d
        # Refresh-only arm preserves74 eligibility and interval control. This
        # isolates the error of using t-1 history for future t+h propagation.
        innovation=gain['gain_px_s2']*np.asarray(event['delta']) if eligible else np.zeros(2)
        self.refresh.step(t,z,self.previous_history+innovation if eligible else np.zeros(2),current_history+innovation if eligible else np.zeros(2),eligible)
        if self.refresh.bank is not None:
            coefficient,d=self.refresh.coefficient();diagnostic['models']['rollout_refresh']=d
            if q['valid']:q['coefficients']['rollout_refresh']=coefficient
        else:diagnostic['models']['rollout_refresh']=dict(mode_probability=None)
        if q['valid']:
            old=q['dual_branch_diagnostic']['dual_cv_event_increment'];choice='dual_cv_event_increment' if old['physical_input_active'] else 'trusted_acc'
            q['coefficients']['history_when_missing']=copy.deepcopy(q['coefficients'][choice])
            for name in ADDED:q['predictions'][name]={str(h):prior.evaluate(q['coefficients'][name],h) for h in prior.PARAMS['horizons']}
        diagnostic['models']['history_when_missing']=dict(fallback=not q['dual_branch_diagnostic']['dual_cv_event_increment']['physical_input_active'])
        q['separated_diagnostic']=diagnostic
        if q['valid']:self.previous_history=current_history.copy()
        self.previous_gain=copy.deepcopy(inc['models'])
        return q

def controls():
    # Nonzero vector means expose upstream scalar zip covariance; verify the
    # wrapper's standard between-component covariance exactly by hand.
    f=[CompatibleFilter([1,2]),CompatibleFilter([5,6])];f[0].x=np.array([1.,2.,3.,4.]);f[1].x=np.array([5.,6.,7.,8.])
    for ff in f:ff.P=np.eye(4);ff.likelihood=1.
    bank=AuthorMMAE(f,np.array([.25,.75]),4)
    for ff in f:ff.update=lambda z,R=None,H=None:None
    bank.update([0,0]);x,P=mixture(f,bank.p)
    expected=np.eye(4)+3*np.ones((4,4));np.testing.assert_allclose(P,expected,atol=1e-12)
    assert not np.allclose(bank.P,P)
    # An independent neutral mode must survive wrong controls in the other
    # mode. Compare it to a separately updated identical filter.
    mmae=SeparatedBank('mmae');reference=CompatibleFilter([0.,0.]);imm=SeparatedBank('imm')
    maximum=0.
    for j in range(80):
        t=j*.02;z=[t*40.,5*np.sin(t)];a=np.array([2200.,-1800.])
        mmae.step(t,z,a,a,True);imm.step(t,z,a,a,True)
        if j:
            reference.dt=.02;reference.predict();reference.update(z)
        error=float(np.max(abs(mmae.bank.filters[0].x-reference.x)));maximum=max(maximum,error)
        np.testing.assert_allclose(mmae.bank.filters[0].x,reference.x,atol=1e-10,rtol=0)
    assert np.linalg.norm(imm.bank.filters[0].x-reference.x)>.01
    c,d=mmae.coefficient();assert np.isfinite(np.array(c['rollout_positions'])).all()
    return dict(upstream_vector_covariance_defect_detected=True,wrapper_mixture_covariance_matches_hand_calculation=True,
        independent_cv_protected_from_wrong_other_control=True,protected_cv_max_error=maximum,
        imm_cross_mode_state_difference_detected=True,finite_rollout=True,
        caveat='Structural/numerical controls, not a field accuracy guarantee.')
