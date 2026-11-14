"""An ellipse extending outside the sensor is not a full observed disk.

Conservative eligibility guard, not an occlusion detector or geometry truth.
Keeps the actual measurement in the input list, marking its inferred boundary.
"""
from pathlib import Path
import copy,math,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage127')]
from independent127 import Independent,NAMES
PARAMS=dict(rule='Full ellipse must lie inside pixel-center FOV[0,w-1]x[0,h-1]. Analytic rotated-ellipse bounds, no fitted perimeter outside sensor is declared complete.',
    action='Keep current measurement but mark mode=arc, inferred_shape=True, visible_arc_fraction=None, support_fraction=None; original104/119 then reject it as complete four/three input. No point deleted or invented.',
    geometry='Projected fitted-ellipse bounds, not physical rotor boundary truth. Conservative eligibility, not calibrated uncertainty or an occlusion segmentation.',
    models='Original127 Independent models and shared KF; same full-four five-frame qualifications, shape fit and tracking. Both count and occupancy supports checked.',default_promoted=False)
def bounded(rotors,w,h):
    result=[];audit=[]
    for r in rotors:
        q=copy.deepcopy(r);a=q['major']/2;b=q['minor']/2;theta=math.radians(q['angle']);dx=math.hypot(a*math.cos(theta),b*math.sin(theta));dy=math.hypot(a*math.sin(theta),b*math.cos(theta))
        box=[q['cx']-dx,q['cy']-dy,q['cx']+dx,q['cy']+dy];clipped=box[0]<0 or box[1]<0 or box[2]>w-1 or box[3]>h-1
        if clipped and q.get('mode','ellipse')=='ellipse':
            q.update(mode='arc',inferred_shape=True,visible_arc_fraction=None,support_fraction=None,fov_clipped=True,full_disk_observed=False)
        audit.append(dict(fitted_bounds_px=box,ellipse_outside_sensor=clipped,full_ellipse_eligible=q.get('mode','ellipse')=='ellipse' and not q.get('inferred_shape',False)))
        result.append(q)
    return result,audit
class Guarded:
    def __init__(self,w,h):self.w=w;self.h=h;self.model=Independent()
    def step(self,t,body,axis,count,occupancy):
        c,ca=bounded(count,self.w,self.h);o,oa=bounded(occupancy,self.w,self.h);result=self.model.step(t,body,axis,c,o)
        result.update(boundary_audit=dict(count=ca,occupancy=oa),guarded_current_inputs=dict(count=c,occupancy=o));return result
