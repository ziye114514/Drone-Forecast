"""Binary bipolar support, keeping original state, line and conic gates.

The AST adapters change only the support mask/count gate and the matching
merge-mask provenance. Standard occupancy/majority filtering, not novel.
"""
from pathlib import Path
import ast,inspect,sys,textwrap
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'research/stage110'),str(ROOT/'research/stage55')]
from merge110 import MergeRecovery
from roi55 import GeometryROI
import polarity

PARAMS=dict(modes=['count_global','count_roi','occupancy_global','occupancy_roi'],
    bipolar='Both polarities must occur at the SAME pixel in current20ms; no count floor or synthetic event.',
    support='Gaussian3x3 sigma.8 of binary bipolar occupancy >=.5 (local majority), then original2x2 opening/3x3 closing with110 merge protection.',
    activity='Binary occupied pixels in occupancy arms; original normalized count activity in count arms. Original minimum activity30 retained; units differ explicitly.',
    roi='Reuse55 current body-box validation,25% margin,160ms age. Candidate-centroid gate after fitting, never crop contours. ROI arms require valid current hint; otherwise no candidates. Global arms do not.',
    unchanged='All original grouping, recovery, line-first fitting,104 qualifications, five-frame initialization and117/119/121 model logic.',
    threshold=.5,nn_required_for_roi=True,forced_four=False)

class MaskAdapter(ast.NodeTransformer):
    def __init__(self):self.changed=[]
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Subscript) and isinstance(node.targets[0].value,ast.Name) and node.targets[0].value.id=='repeated':
            node.targets[0].slice=ast.parse('self.count_gate(positive,negative,c.min_pixel_count)',mode='eval').body;self.changed.append('count_gate')
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='raw_mask':
            node.value=ast.parse('self.support_mask(blurred,threshold)',mode='eval').body;self.changed.append('raw_mask')
        return self.generic_visit(node)
tree=ast.parse(ast.unparse(polarity._tree));adapter=MaskAdapter();tree=ast.fix_missing_locations(adapter.visit(tree))
assert adapter.changed==['count_gate','raw_mask'],adapter.changed
ns=dict(polarity._reused_detect.__globals__);exec(compile(tree,'<124 support-mask-only adapter>','exec'),ns);detect124=ns['detect']

class MergeMask(ast.NodeTransformer):
    def __init__(self):self.changed=[]
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ['density','positive','threshold','raw']:
                node.value=ast.parse(dict(density='None',positive='None',threshold='self.support_threshold',raw='self.current_raw_mask.copy()')[name],mode='eval').body
                self.changed.append(name)
        return self.generic_visit(node)
merge_tree=ast.parse(textwrap.dedent(inspect.getsource(MergeRecovery.contours_before_merge)));merge_adapter=MergeMask()
merge_tree=ast.fix_missing_locations(merge_adapter.visit(merge_tree));assert merge_adapter.changed==['density','positive','threshold','raw']
merge_ns=dict(MergeRecovery.contours_before_merge.__globals__);exec(compile(merge_tree,'<124 actual merge mask provenance>','exec'),merge_ns)

class SupportFrontend(MergeRecovery):
    contours_before_merge=merge_ns['contours_before_merge']
    def __init__(self,w,h,mode):
        super().__init__(w,h,'merge_safe');assert mode in PARAMS['modes'];self.support_mode=mode
        self.region=GeometryROI(w,h,'body');self.current_roi=None;self.roi_diag=None;self.capture=False;self.support_threshold=None
    def set_hint(self,events,body,t):
        _,self.roi_diag=self.region.apply(events[:0],body,t)
        self.current_roi=self.roi_diag['roi'] if self.roi_diag['mode']=='body_roi' else None
    def count_gate(self,on,off,minimum):
        return np.zeros(on.shape,bool) if self.support_mode.startswith('occupancy') else (on+off)<minimum
    def feature(self,on,off):
        return ((on>0)&(off>0)).astype(np.float32) if self.support_mode.startswith('occupancy') else super().feature(on,off)
    def support_mask(self,blurred,threshold):
        self.support_threshold=PARAMS['threshold'] if self.support_mode.startswith('occupancy') else float(threshold)
        self.current_raw_mask=np.uint8(blurred>=self.support_threshold)*255
        if self.capture:self.current_density=blurred.copy()
        return self.current_raw_mask
    def _candidates(self,*args,**kwargs):
        found=super()._candidates(*args,**kwargs)
        if self.support_mode.endswith('_roi'):
            if self.current_roi is None:found=[]
            else:
                lo=np.array(self.current_roi[:2]);hi=np.array(self.current_roi[2:])
                found=[r for r in found if np.all(r['center']>=lo) and np.all(r['center']<hi)]
        return found
    def detect_prepared(self,events,t,prepared):
        # Reuse full RecoveryFrontend wrapper; inject the mask adapter into the
        # PolarityFrontend module only for this synchronous call, restoring it.
        original=polarity._reused_detect;polarity._reused_detect=detect124
        try:return super().detect_prepared(events,t,prepared)
        finally:polarity._reused_detect=original
