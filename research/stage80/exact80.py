"""Exact optimizations: per-window background memo and forecast copy-on-write.

Freeze source modules remain unchanged. Only full forecast-envelope cloning
changes; individual model coefficients, queues and numerical states still use
the original deepcopy. Never mutate a shared prior coefficient/diagnostic.
"""
from pathlib import Path
from types import FunctionType,SimpleNamespace
import copy,inspect,sys,textwrap
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage78'),str(ROOT/'research/stage77'),str(ROOT/'research/stage74'),str(ROOT/'research/stage73'),str(ROOT/'research/stage72'),str(ROOT/'research/stage70'),str(ROOT/'research/stage69')]
from online78 import OnlineExtras as OriginalExtras
from online73 import _tone_globals
import activity68
from force69 import ForceForecast
from guard70 import Guard
import guard70
from increment72 import IncrementForecast,ADDED as DELTAS
from hybrid74 import HybridForecast
from separated77 import SeparatedForecast

PARAMS=dict(parameters_changed=False,models_removed=False,
    activity='Same68 arithmetic and73 carriers. Background tone at repeated frequency '
             'computed once per current window, then reused exactly.',
    envelope='Clone root, coefficients mapping and predictions mapping; retain '
             'read-only prior values. All small coefficient/state/queue deepcopies '
             'remain unchanged. Test old outputs and prior snapshots immutable.',
    implementation='External function globals adapter; no patch to frozen modules '
                   'or global Python copy module. All49 predictions retained.')

def clone_envelope(x,*args,**kwargs):
    if isinstance(x,dict) and 'coefficients' in x and 'predictions' in x and 'valid' in x:
        assert not args and not kwargs
        return dict(x,coefficients=dict(x['coefficients']),predictions=dict(x['predictions']))
    return copy.deepcopy(x,*args,**kwargs)

def adapted(fn,**extra):
    namespace=dict(fn.__globals__,copy=SimpleNamespace(deepcopy=clone_envelope),**extra)
    return FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)

source=textwrap.dedent(inspect.getsource(activity68.Activity.step))
old='f,p=spectrum(signals);candidates=[];accepted=[]'
assert source.count(old)==1;source=source.replace(old,old+';background_tones={}')
old='tt=tone(a,freq);bg=tone(background,freq)'
assert source.count(old)==1
source=source.replace(old,'tt=tone(a,freq)\n            if freq not in background_tones:background_tones[freq]=tone(background,freq)\n            bg=background_tones[freq]')
namespace=dict(activity68.Activity.step.__globals__,tone=_tone_globals['tone'])
exec(compile(source,'<80 per-window exact background memo>','exec'),namespace)
class CachedActivity(activity68.Activity):step=namespace['step']
class FastForce(ForceForecast):step=adapted(ForceForecast.step)
class FastGuard(Guard):step=adapted(Guard.step)
class FastIncrement(IncrementForecast):step=adapted(IncrementForecast.step)
class FastIncrementGuard(Guard):step=adapted(Guard.step,ADDED=DELTAS)
class FastHybrid(HybridForecast):step=adapted(HybridForecast.step)
class FastSeparated(SeparatedForecast):step=adapted(SeparatedForecast.step)

class OnlineExtras(OriginalExtras):
    def __init__(self,w,h):
        super().__init__(w,h);self.activity=CachedActivity(w,h)
        self.force=FastForce();self.guard=FastGuard();self.increment=FastIncrement()
        self.increment_guard=FastIncrementGuard();self.hybrid=FastHybrid();self.separated=FastSeparated()

def controls():
    q=dict(valid=True,t=0.,coefficients=dict(cv=dict(position=[1.,2.],velocity=[3.,4.],acceleration=[0.,0.])),
           predictions=dict(cv={'0.1':[1.3,2.4]}),diagnostic=dict(a=[1,2,3]))
    before=copy.deepcopy(q);c=clone_envelope(q);c['coefficients']['new']=dict(x=2);c['predictions']['new']={'0.1':[1,2]}
    assert q==before and c is not q and c['coefficients'] is not q['coefficients']
    assert c['coefficients']['cv'] is q['coefficients']['cv'] and c['diagnostic'] is q['diagnostic']
    coefficient=clone_envelope(q['coefficients']['cv']);coefficient['position'][0]=99
    assert q==before
    return dict(envelope_mapping_isolated=True,old_values_shared_read_only=True,
                individual_coefficient_deepcopy_retained=True,no_global_copy_patch=True,
                required_followup='Actual RAW49 equality and earlier-row immutability, not just this small control.')
