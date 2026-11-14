import unittest,sys
from pathlib import Path
import numpy as np
from study import camera_circle,project,circle_normal_candidates,normal_error,DTYPE
from activity_fallback import ActivityTracker as OriginalActivity
from activity_fallback_fixed import ActivityTracker

def cluster(x=200,y=200):
    xx,yy=np.meshgrid(np.arange(x,x+20),np.arange(y,y+20))
    e=np.zeros(xx.size*3,DTYPE);e['x']=np.repeat(xx.ravel(),3);e['y']=np.repeat(yy.ravel(),3);return e

class ValidationChecks(unittest.TestCase):
    def test_analytic_conic_matches_projected_3d_rim(self):
        w=camera_circle(65,40,120,True);phase=np.linspace(0,2*np.pi,100,endpoint=False)
        points=project(w['H'],np.column_stack((np.cos(phase),np.sin(phase))))
        x,y,a,b,angle=w['ellipse'];d=points-[x,y]
        u=d[:,0]*np.cos(angle)+d[:,1]*np.sin(angle);v=-d[:,0]*np.sin(angle)+d[:,1]*np.cos(angle)
        np.testing.assert_allclose((u/a)**2+(v/b)**2,1,atol=1e-10)

    def test_cone_inverse_has_true_normal_without_using_fit(self):
        for tilt in (0,20,80,88):
            w=camera_circle(tilt,40,36,True)
            self.assertLess(min(normal_error(q,w['normal']) for q in circle_normal_candidates(w['ellipse'],w['K'])),.001)

    def test_reacquisition_after_loss(self):
        tracker=ActivityTracker(1280,720);e=cluster()
        for t in (.01,.02,.03):r=tracker.step(e,t)
        self.assertTrue(r['detected'])
        tracker.step(np.empty(0,DTYPE),.5)
        for t in (.51,.52,.53):r=tracker.step(cluster(400,300),t)
        self.assertTrue(r['detected'])
        np.testing.assert_allclose(r['center'],[410,310])
        self.assertEqual(r['geometry_quality'],0.)

    def test_old_reset_bug_reproduces(self):
        tracker=OriginalActivity(1280,720)
        for t in (.01,.02,.03):tracker.step(cluster(),t)
        tracker.step(np.empty(0,DTYPE),.5)
        for t in (.51,.52,.53,.54):r=tracker.step(cluster(400,300),t)
        self.assertFalse(r['detected'])

if __name__=='__main__':unittest.main()
