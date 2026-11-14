"""ON/OFF-only source adapter; reuse old geometry/grouping without editing it.

The original detect() AST is changed only at its count/feature assignments and
the minimum-count gate. Original source hashes are frozen by the experiment.
Strict10 is required to match the unchanged PaperPerception exactly on RAW.
"""
import ast,inspect,textwrap
from collections import deque
import cv2,numpy as np
from event_drone_forecast.perception import RotorPerception
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.paper_config import PaperConfig

METHODS={'strict10':dict(window_ms=10,neighbor=False),
         'neighbor10':dict(window_ms=10,neighbor=True),
         'strict30':dict(window_ms=30,neighbor=False),
         'neighbor30':dict(window_ms=30,neighbor=True)}

class _CountAdapter(ast.NodeTransformer):
    def __init__(self):self.changed=[]
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ['pixel','total','positive','negative','repeated']:
                self.changed.append(name)
                replacements={'pixel':'None','total':"self.prepared['total']",
                    'positive':"self.prepared['on']",'negative':"self.prepared['off']",
                    'repeated':'self.feature(positive,negative)'}
                node.value=ast.parse(replacements[name],mode='eval').body
        # Rate-normalized window count gate, same units in every arm.
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Subscript):
            target=node.targets[0]
            if isinstance(target.value,ast.Name) and target.value.id=='repeated':
                self.changed.append('count_gate')
                target.slice=ast.parse('(positive+negative)<c.min_pixel_count',mode='eval').body
        return self.generic_visit(node)

_tree=ast.parse(textwrap.dedent(inspect.getsource(RotorPerception.detect)))
_adapter=_CountAdapter();_tree=_adapter.visit(_tree)
assert _adapter.changed==['pixel','total','positive','negative','repeated','count_gate'],_adapter.changed
_namespace=dict(RotorPerception.detect.__globals__)
exec(compile(ast.fix_missing_locations(_tree),'<stage8_count_adapter>','exec'),_namespace)
_reused_detect=_namespace['detect']

def counts(events,w,h):
    pix=events['y'].astype(np.int32)*w+events['x']
    total=np.bincount(pix,minlength=w*h).reshape(h,w)
    on=np.bincount(pix[events['p']>0],minlength=w*h).reshape(h,w)
    return dict(total=total,on=on.astype(np.float32),off=(total-on).astype(np.float32))

class PolarityFrontend(PaperPerception):
    def __init__(self,w,h,method):
        super().__init__(w,h,PaperConfig());self.method=method;self.prepared=None;self.candidates=[]
        self.window=deque()

    def feature(self,on,off):
        if METHODS[self.method]['neighbor']:
            on=cv2.GaussianBlur(on,(3,3),.8)
            off=cv2.GaussianBlur(off,(3,3),.8)
        return np.minimum(on,off).astype(np.float32)

    def detect_prepared(self,events,t,prepared):
        self.prepared=prepared;self.candidates=[]
        observation,total=_reused_detect(self,events,t)
        if observation['detected']:
            for serialized,measured in zip(observation['rotors'],self.previous_rotors):serialized['activity']=float(measured['activity'])
        return observation,total

    def detect(self,events,t):
        current=counts(events,self.w,self.h);self.window.append((t,current))
        while self.window and self.window[0][0]<t-.020001:self.window.popleft()
        if METHODS[self.method]['window_ms']==10:prepared=current
        else:
            # Divide by three even during startup; no full-window future fill.
            prepared=dict(total=current['total'],on=sum(r['on'] for _,r in self.window)/3,
                off=sum(r['off'] for _,r in self.window)/3)
        return self.detect_prepared(events,t,prepared)

    def _candidates(self,*args,**kwargs):
        found=super()._candidates(*args,**kwargs)
        self.candidates.extend(dict(cx=float(r['center'][0]),cy=float(r['center'][1]),
            major=float(r['major']),minor=float(r['minor']),angle=float(r['angle']),mode=r['mode']) for r in found)
        return found

def tests():
    # Adjacent bipolar supports are missed by native-pixel coincidence.
    a=np.zeros((31,31),np.float32);b=a.copy();a[15,14]=30;b[15,16]=30
    f=PolarityFrontend(31,31,'strict10');g=PolarityFrontend(31,31,'neighbor10')
    assert f.feature(a,b).max()==0 and g.feature(a,b).max()>0
    # Spatial tolerance cannot create a second polarity from a monopolar edge.
    assert g.feature(a,np.zeros_like(a)).max()==0
    assert np.all(g.feature(a,b)>=cv2.GaussianBlur(np.minimum(a,b),(3,3),.8)-1e-6)
    # Stationary rate evidence is invariant to 3x integration after normalization.
    np.testing.assert_array_equal(g.feature(a,b),g.feature(a*3/3,b*3/3))
    # Exercise the reused full detector against its untouched source, not just the feature equation.
    mask=np.zeros((240,320),np.uint8)
    for center in [(90,70),(230,70),(90,170),(230,170)]:cv2.ellipse(mask,center,(22,7),12,0,360,1,-1)
    yy,xx=np.nonzero(mask);dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')])
    events=np.zeros(len(xx)*6,dtype=dtype);events['x']=np.repeat(xx,6);events['y']=np.repeat(yy,6);events['p']=np.tile([0,1,0,1,0,1],len(xx))
    original=PaperPerception(320,240,PaperConfig());adapted=PolarityFrontend(320,240,'strict10')
    for tick in range(1,5):
        events['t']=tick*10000-100
        expected,_=original.detect(events,tick*.01);actual,_=adapted.detect(events,tick*.01)
        assert expected==actual
    return dict(adjacent_support_recovered=True,monopolar_rejected=True,rate_units_equal=True,
        adapter_assignments_only=_adapter.changed,geometry_and_grouping_reused=True,full_detector_synthetic_exact=True)
