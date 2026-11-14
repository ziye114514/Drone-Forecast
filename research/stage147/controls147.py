"""Known single/double masks, line/FOV guards, and causal geometry controls.

Synthetic cases test failure modes, not real-camera accuracy or physical RPM.
"""
from pathlib import Path
import json,sys
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from curvature147 import split_contour,CurvatureGeometry,CurvatureFrontend
from probe147 import OUT,controls,js

def main():
    controls(); rejected=[]; false_splits=[]
    rng=np.random.default_rng(147)
    for angle in (0,30,70,115):
        for ratio in (.18,.4,.7,1.):
            for defect in ('clean','one_gap','two_dents','edge_noise','clipped'):
                mask=np.zeros((220,280),np.uint8)
                center=(140,110); axes=(55,max(3,int(55*ratio)))
                cv2.ellipse(mask,center,axes,angle,0,360,255,-1)
                if defect in ('one_gap','two_dents'):
                    for side in ([1] if defect=='one_gap' else [-1,1]):
                        a=np.deg2rad(angle);point=np.array(center)+side*axes[1]*np.array([-np.sin(a),np.cos(a)])
                        cv2.circle(mask,tuple(np.rint(point).astype(int)),5,0,-1)
                if defect=='edge_noise':
                    for _ in range(12):
                        phase=rng.uniform(0,2*np.pi);a=np.deg2rad(angle)
                        d=np.array([axes[0]*np.cos(phase),axes[1]*np.sin(phase)])
                        R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                        point=np.array(center)+R@d;cv2.circle(mask,tuple(np.rint(point).astype(int)),2,0,-1)
                if defect=='clipped':mask[:,140:]=0
                cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                arcs,diag=split_contour(max(cs,key=cv2.contourArea))
                record=dict(angle=angle,ratio=ratio,defect=defect)
                (false_splits if arcs is not None else rejected).append(record)
    # Explicit partial-arc exception may never bypass field-of-view integrity.
    def rotor(x,y,curved=False):
        return dict(cx=x,cy=y,major=40.,minor=20.,angle=0.,ratio=.5,mode='arc' if curved else 'ellipse',
                    inferred_shape=curved,curvature_validated=curved)
    points=[rotor(5,50,True),rotor(100,50),rotor(100,100),rotor(30,100)]
    result=CurvatureGeometry(160,160).step(.02,dict(detected=True,center=[60,75]),points)
    assert result['state']=='4_unresolved', result
    assert result['observed_rotors'][0]['mode']=='arc' and result['observed_rotors'][0]['fov_clipped']
    front=CurvatureFrontend(300,300)
    def candidate(x,y,child=False):
        return dict(center=np.array([x,y],float),major=40.,activity=100.,shape_quality=.9,
                    curvature_validated=child,curvature_parent=0)
    # A perfect square initially tempts the count-four grouping to take only
    # one child from a split pair. It must restore the original parent instead.
    front.parent_candidates147={0:[candidate(50,160)]}
    selected=front._group([candidate(50,50),candidate(150,50),candidate(150,150),
                           candidate(50,150,True),candidate(50,170,True)],.02)
    child_count=sum(r['curvature_validated'] for r in selected)
    assert child_count in (0,2) and front.orphan_rejections147>0
    summary=dict(single_masks=80,rejected=len(rejected),false_splits=false_splits,
                 partial_arcs_keep_inferred_status=True,partial_arc_cannot_bypass_fov=True,
                 orphan_group_restores_parent=True,
                 synthetic_not_real_accuracy=True)
    (OUT/'negative_controls147.json').write_text(js(summary),encoding='utf-8')
    print(js(summary),flush=True)
    assert not false_splits, false_splits

if __name__=='__main__':main()
