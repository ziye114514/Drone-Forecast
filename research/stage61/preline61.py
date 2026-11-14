"""Route genuinely thin contours to existing line geometry before ellipse fitting."""
from pathlib import Path
import ast,copy,inspect,sys,textwrap
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage57'),str(ROOT/'research/stage60')]
from gating57 import CandidateFrontend
from frontend60 import LineFrontend
from event_drone_forecast.rotor_sensing import PaperPerception
from event_drone_forecast.geometry import _line
PARAMS=dict(thin_width_ratio=.14,min_line_major_px=12.,shape_quality_cap=.4,geometry_quality_cap=.4,
    scope='Use unchanged geometry._line on preclassified thin contours, reject too short/invalid line rather than fall back to ill-defined ellipse. All original broad-contour/partial/grouping/frequency code retained. No RNG seed changes in inference, no forced four points or real rotor identity claim.')

class Insert(ast.NodeTransformer):
    def __init__(self):self.changed=0
    def visit_Try(self,node):
        if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fitEllipse' for n in ast.walk(node)):
            self.changed+=1
            before=ast.parse('''was_thin, measured_line = self.measure_preline(contour, activity_map, b)
if was_thin:
    if measured_line is not None:
        found.append(measured_line)
    continue
''').body
            return before+[node]
        return self.generic_visit(node)

tree=ast.parse(textwrap.dedent(inspect.getsource(PaperPerception._candidates)));insertion=Insert();tree=ast.fix_missing_locations(insertion.visit(tree));assert insertion.changed==1
namespace=dict(PaperPerception._candidates.__globals__);exec(compile(tree,'<61 pre-ellipse line route>','exec'),namespace)
DERIVED_SOURCE=ast.unparse(tree);modified_candidates=namespace['_candidates']

class PrelinePaper(PaperPerception):
    _candidates=modified_candidates
    def measure_preline(self,contour,activity_map,b):
        p=np.asarray(contour,float).reshape(-1,2)*b;c=p.mean(0)
        values,vectors=np.linalg.eigh(np.cov(p.T));u=vectors[:,-1];v=vectors[:,0]
        length=float(np.ptp((p-c)@u));width=max(1.,float(np.quantile((p-c)@v,.95)-np.quantile((p-c)@v,.05)))
        if length<=0 or width/max(length,1)>PARAMS['thin_width_ratio']:return False,None
        self.preline_audit['thin_routed']+=1;m=_line(p,self.w*.5)
        if m is None:
            self.preline_audit['thin_rejected']+=1;return True,None
        x,y,w,h=cv2.boundingRect(contour);activity=float(activity_map[y:y+h,x:x+w].sum())
        if activity<30:
            self.preline_audit['thin_rejected']+=1;return True,None
        m.update(bbox=np.array([x*b,y*b,(x+w)*b,(y+h)*b],float),activity=activity,
            shape_quality=min(PARAMS['shape_quality_cap'],m['shape_quality']),
            geometry_quality=min(PARAMS['geometry_quality_cap'],m['geometry_quality']),
            pre_ellipse_line=True,verified_rotor_identity=False)
        self.preline_audit['line_measured']+=1;return True,m

class StableFrontend(CandidateFrontend,PrelinePaper):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.preline_audit=dict(thin_routed=0,thin_rejected=0,line_measured=0)
    def detect_prepared(self,*args,**kwargs):
        self.preline_audit=dict(thin_routed=0,thin_rejected=0,line_measured=0)
        return super().detect_prepared(*args,**kwargs)

class StablePairedFrontend(LineFrontend,PrelinePaper):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.preline_audit=dict(thin_routed=0,thin_rejected=0,line_measured=0)
    def detect_prepared(self,*args,**kwargs):
        self.preline_audit=dict(thin_routed=0,thin_rejected=0,line_measured=0)
        return super().detect_prepared(*args,**kwargs)

def checks():
    # Actual wrapper composition must intercept the old PolarityFrontend super call.
    assert StableFrontend.mro().index(PrelinePaper)<StableFrontend.mro().index(PaperPerception)
    assert StablePairedFrontend.mro().index(PrelinePaper)<StablePairedFrontend.mro().index(PaperPerception)
    from pathlib import Path
    import json
    probe=json.loads((ROOT.parent/'output/stage60_paired_lines_20261021/degenerate_contour_probe.json').read_text(encoding='utf-8'))
    real=next(r for r in probe['contours'] if r['bbox']==[443,293,1,12]);contour=np.array(real['points'],np.int32).reshape(-1,1,2)
    activity=np.ones((720,1280),np.float32)*3;front=StableFrontend(1280,720,'neighbor10')
    original=cv2.fitEllipse
    def forbidden(unused):raise AssertionError('Thin real contour reached fitEllipse')
    cv2.fitEllipse=forbidden
    try:
        result=front._candidates([contour],activity,1.,1,thin_only=True);assert not result and front.preline_audit['thin_rejected']==1
        points=np.c_[np.full(21,30),np.arange(20,41)];long=np.r_[points,points[::-1]].astype(np.int32).reshape(-1,1,2)
        f=StableFrontend(128,128,'neighbor10');one=f._candidates([long],np.ones((128,128),np.float32)*3,1.,1,thin_only=True)
        assert len(one)==1 and np.allclose(one[0]['center'],[30,30]) and one[0]['major']==20
    finally:cv2.fitEllipse=original
    # Broad geometry executes the entire unchanged frontend and grouping path.
    mask=np.zeros((240,320),np.uint8)
    for c in [(90,70),(230,70),(90,170),(230,170)]:cv2.ellipse(mask,c,(22,9),12,0,360,1,-1)
    yy,xx=np.nonzero(mask);dtype=np.dtype([('x','u2'),('y','u2'),('p','i2'),('t','i8')]);e=np.zeros(len(xx)*6,dtype);e['x']=np.repeat(xx,6);e['y']=np.repeat(yy,6);e['p']=np.tile([0,1,0,1,0,1],len(xx))
    old=CandidateFrontend(320,240,'neighbor10');new=StableFrontend(320,240,'neighbor10')
    for tick in range(1,6):
        e['t']=tick*10000-100;t=tick*.01;old.set_hint(e,None,t,'off');new.set_hint(e,None,t,'off')
        a,_=old.detect(e,t);b,_=new.detect(e,t);assert a==b and new.preline_audit['thin_routed']==0
    return dict(actual_collinear_contour_never_calls_fitEllipse=True,too_short_real_line_rejected=True,
                known_long_line_center_exact=True,broad_full_frontend_geometry_exact=True,actual_MRO_wrappers_checked=True)
