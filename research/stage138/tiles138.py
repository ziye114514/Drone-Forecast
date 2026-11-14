"""Morphology/contours on the bounding support plus a four-pixel halo.

The threshold is still computed globally. Every foreground pixel is retained;
this is a compute ROI around ALL support, not a body-box measurement crop.
Global contour coordinates and global activity-map size remain unchanged.
"""
from pathlib import Path
import ast,copy,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage137')]
import fast137
from fast137 import occupancy124,polarity,SupportFrontend,MergeRecovery
from merge110 import PARAMS as MERGE
PARAMS=dict(halo_px=4,full_fallback_area_fraction=.8,threshold='Original full-image statistic and masks retained.',
    support='Bounding box of all current threshold foreground, padded4px. Retains every connected component; coordinates offset back to sensor.',
    cache='One opening shared by closed-parent/independent-child merge logic; parent label map uses same tile.',
    measurement='No body-box clipping, no new support, no changed contour ranking or qualification.')

class TileOps:
    def open_tile(self,raw):
        h,w=raw.shape;x,y,bw,bh=cv2.boundingRect(raw);pad=PARAMS['halo_px']
        if bw:
            x0=max(0,x-pad);y0=max(0,y-pad);x1=min(w,x+bw+pad);y1=min(h,y+bh+pad)
        else:x0=y0=0;x1=y1=1
        fallback=(x1-x0)*(y1-y0)>=w*h*PARAMS['full_fallback_area_fraction']
        if fallback:x0=y0=0;x1=w;y1=h
        self.mask_origin=(x0,y0);self.raw_tile=np.ascontiguousarray(raw[y0:y1,x0:x1])
        self.current_open_tile=cv2.morphologyEx(self.raw_tile,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
        self.tile_diagnostic=dict(roi=[x0,y0,x1,y1],area_fraction=(x1-x0)*(y1-y0)/(w*h),full_fallback=fallback,all_foreground_retained=True)
        return self.current_open_tile
    def close_tile(self,opened):return cv2.morphologyEx(opened,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    def contours_before_merge(self,contours,activity,b):
        children,_=cv2.findContours(self.current_open_tile,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE,offset=self.mask_origin)
        audit=dict(mode=self.merge_mode,threshold=self.support_threshold,closed_components=len(contours),open_components=len(children),split_parents=0,retained_children=0,no_fixed_rotor_count=True)
        if self.merge_mode=='open_only':self.merge_audit=audit;return children
        ox,oy=self.mask_origin;parent_map=np.zeros(self.current_open_tile.shape,np.int32)
        for i,c in enumerate(contours):cv2.drawContours(parent_map,[c],-1,i+1,-1,offset=(-ox,-oy))
        groups={}
        for child in children:
            x,y=child[0,0];parent=int(parent_map[y-oy,x-ox])-1
            if parent>=0:groups.setdefault(parent,[]).append(child)
        selected=[]
        for i,parent in enumerate(contours):
            area=cv2.contourArea(parent);qualified=[c for c in groups.get(i,[]) if self._eligible(c,activity,b,area)];sizes=[max(cv2.boundingRect(c)[2:]) for c in qualified]
            good=len(qualified)>=2 and sum(cv2.contourArea(c) for c in qualified)>=MERGE['minimum_children_coverage']*area
            good=good and max(sizes,default=1)/max(min(sizes,default=1),1)<=MERGE['maximum_child_size_ratio']
            if good:selected.extend(qualified);audit['split_parents']+=1;audit['retained_children']+=len(qualified)
            else:selected.append(parent)
        self.merge_audit=audit;return selected

class TileAdapter(ast.NodeTransformer):
    def __init__(self):self.changed=[]
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='mask':
            text=ast.unparse(node.value)
            if text=='raw_mask.copy()':node.value=ast.parse('self.open_tile(raw_mask)',mode='eval').body;self.changed.append('open_once')
            elif 'MORPH_OPEN' in text:node.value=ast.Name('mask',ast.Load());self.changed.append('reuse_open')
            elif 'MORPH_CLOSE' in text:node.value=ast.parse('self.close_tile(mask)',mode='eval').body;self.changed.append('close_tile')
        return self.generic_visit(node)
    def visit_Call(self,node):
        if isinstance(node.func,ast.Attribute) and node.func.attr=='findContours':
            assert isinstance(node.args[0],ast.Name)
            if node.args[0].id=='raw_mask':node.args[0]=ast.parse('self.raw_tile',mode='eval').body
            node.keywords.append(ast.keyword(arg='offset',value=ast.parse('self.mask_origin',mode='eval').body));self.changed.append('contour_offset')
        return self.generic_visit(node)

def detector(tree):
    adapter=TileAdapter();tree=ast.fix_missing_locations(adapter.visit(copy.deepcopy(tree)));assert adapter.changed==['open_once','reuse_open','close_tile','contour_offset','contour_offset'],adapter.changed
    namespace=dict(occupancy124.ns);exec(compile(tree,'<138 global-support tile morphology>','exec'),namespace);return namespace['detect']
count_detect=detector(occupancy124.tree);occupancy_detect=detector(fast137.tree)

class TileFrontend(TileOps,SupportFrontend):
    def detect_prepared(self,events,t,prepared):
        original=polarity._reused_detect;polarity._reused_detect=occupancy_detect if self.support_mode.startswith('occupancy') else count_detect
        try:return MergeRecovery.detect_prepared(self,events,t,prepared)
        finally:polarity._reused_detect=original

def fronts(w,h):return dict(count=TileFrontend(w,h,'count_global'),occupancy=TileFrontend(w,h,'occupancy_roi'))
