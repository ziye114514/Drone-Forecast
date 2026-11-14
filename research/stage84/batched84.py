"""Batch unchanged IMM/MMAE future means/covariance over26 horizons.

Filter prediction/update/interaction and learned physics are unmodified.
Only the read-only rollout projection loop is replaced by stacked matmul.
"""
from pathlib import Path
from types import FunctionType
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage82'),str(ROOT/'research/stage80')]
import stream82
import exact80
from hybrid74 import DualBank,HybridForecast
from separated77 import SeparatedBank,SeparatedForecast

TIMES=np.linspace(0,.5,26)
F=np.tile(np.eye(4),(len(TIMES),1,1));F[:,0,2]=TIMES;F[:,1,3]=TIMES
B=np.zeros((len(TIMES),4,2));half=.5*TIMES*TIMES
B[:,0,0]=half;B[:,1,1]=half;B[:,2,0]=TIMES;B[:,3,1]=TIMES
FT=F.transpose(0,2,1);BBT=B@B.transpose(0,2,1)
for array in [TIMES,F,B,FT,BBT]:array.flags.writeable=False
PARAMS=dict(horizons26_unchanged=True,all49_retained=True,filter_state_math_unchanged=True,
    numerical_tolerance_absolute=1e-8,numerical_tolerance_relative=0.,
    scope='Only read-only future mean/covariance projections batched. No state '
          'update, likelihood, physics learning, input eligibility or horizon changes. '
          'Small FP differences allowed only in named rollout arrays/predictions.')

class BatchedDualBank(DualBank):
    def coefficient(self):
        means=[(F@f.x+B@f.a)[:,:2] for f in self.imm.filters]
        points=sum(mu*x for mu,x in zip(self.imm.mu,means));covariance=np.zeros((26,2,2))
        for mu,f,x in zip(self.imm.mu,self.imm.filters,means):
            C=(F@f.P@FT+f.sigma**2*BBT)[:,:2,:2];e=x-points
            covariance+=mu*(C+e[:,:,None]*e[:,None,:])
        c=dict(position=self.imm.x[:2].tolist(),velocity=self.imm.x[2:].tolist(),
            acceleration=sum(mu*f.a for mu,f in zip(self.imm.mu,self.imm.filters)).tolist(),
            rollout_times=TIMES.tolist(),rollout_positions=points.tolist())
        d=dict(mode_probability=self.imm.mu.tolist(),innovation_nis=[f.nis for f in self.imm.filters],
            component_acceleration=[f.a.tolist() for f in self.imm.filters],position_covariance=self.imm.P[:2,:2].tolist(),
            rollout_position_covariance=covariance.tolist())
        return c,d

def clone(fn,**changed):return FunctionType(fn.__code__,dict(fn.__globals__,**changed),fn.__name__,fn.__defaults__,fn.__closure__)
class BatchedSeparatedBank(SeparatedBank):coefficient=clone(SeparatedBank.coefficient,DualBank=BatchedDualBank)
class BatchedHybrid(exact80.FastHybrid):reset=clone(HybridForecast.reset,DualBank=BatchedDualBank)
class BatchedSeparated(exact80.FastSeparated):reset=clone(SeparatedForecast.reset,SeparatedBank=BatchedSeparatedBank)
class OnlineExtras(stream82.OnlineExtras):
    def __init__(self,w,h):
        super().__init__(w,h);self.hybrid=BatchedHybrid();self.separated=BatchedSeparated()

def compare(a,b,path=(),changes=None):
    if changes is None:changes=[]
    if a==b:return changes
    if isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys(),path
        for k in a:compare(a[k],b[k],path+(str(k),),changes)
    elif isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b),path
        for k,(x,y) in enumerate(zip(a,b)):compare(x,y,path+(str(k),),changes)
    else:
        # Constrain numerical tolerance to rollout projections and explicit
        # predictions, never weights/state/learning/inputs or mode selection.
        allowed=('rollout_positions' in path or 'rollout_position_covariance' in path or 'predictions' in path)
        assert allowed and isinstance(a,(int,float)) and isinstance(b,(int,float)),path
        error=abs(float(a)-float(b));assert error<=1e-8,(path,error)
        changes.append(dict(path='/'.join(path),absolute_error=error))
    return changes

def controls():
    from types import SimpleNamespace
    from hybrid74 import ControlledFilter
    rng=np.random.default_rng(8400);maximum=0.;differing=0
    for j in range(120):
        filters=[ControlledFilter([0.,0.]),ControlledFilter([0.,0.])]
        for f in filters:
            f.x=rng.normal(0,1200,4);A=rng.normal(0,100,(4,4));f.P=A@A.T
            f.a=rng.uniform(-2500,2500,2);f.sigma=800. if j%2 else 400.
        weights=rng.dirichlet([1.,1.]);x=sum(w*f.x for w,f in zip(weights,filters));P=sum(w*f.P for w,f in zip(weights,filters))
        bank=SimpleNamespace(imm=SimpleNamespace(filters=filters,mu=weights,x=x,P=P))
        old=DualBank.coefficient(bank);new=BatchedDualBank.coefficient(bank)
        changes=compare(list(old),list(new));differing+=len(changes)
        maximum=max(maximum,max([r['absolute_error'] for r in changes],default=0.))
        assert np.linalg.eigvalsh(np.asarray(new[1]['rollout_position_covariance'])).min()>-1e-7
    return dict(random_banks=120,rollout_and_covariance_max_absolute_error=maximum,differing_numeric_fields=differing,
        state_weights_controls_exact=True,all_rollout_covariance_PSD=True,
        scope='Projection arithmetic controls, not prediction accuracy. Real data states and all49 checks follow.')
