import unittest,itertools
import numpy as np
from measurement import *
from experiments import scene,circle,observed_points,K,rotor_record
from event_drone_forecast.paper_config import PaperConfig

class MeasurementTests(unittest.TestCase):
    def test_four_exact_circle_candidates_resolve_offaxis_plane(self):
        w=scene(45,40,120,True);ellipses=[circle(w,q)[1] for q in [(-2,-2),(2,-2),(-2,2),(2,2)]]
        result=joint_normal(ellipses,K,robust=True)
        self.assertTrue(result['accepted']);self.assertLess(axis_error(result['normal'],w['n']),.001)

    def test_order_of_rotors_does_not_change_joint_axis(self):
        w=scene(65,40,72,True);ellipses=[circle(w,q)[1] for q in [(-2,-2),(2,-2),(-2,2),(2,2)]]
        reference=joint_normal(ellipses,K,robust=True)['normal']
        for ids in itertools.permutations(range(4)):
            r=joint_normal([ellipses[j] for j in ids],K,robust=True)
            self.assertTrue(r['accepted']);self.assertLess(axis_error(reference,r['normal']),.001)

    def test_single_circle_not_declared_disambiguated(self):
        p=circle(scene(45,40,120,True))[1]
        self.assertFalse(joint_normal([p],K,robust=True)['accepted'])

    def test_noisy_full_circle_is_auditable(self):
        w=scene(45,40,120,False);points,truth=observed_points(w,np.random.default_rng(1),noise=.25)
        d=audit_ellipse(points,fit_ams(points),K)
        self.assertTrue(d['accepted']);self.assertLess(d['normal_spread_deg'],8)

    def test_line_and_insufficient_support_do_not_provide_full_pose(self):
        self.assertFalse(audit_ellipse(np.zeros((10,2)),[0,0,60,20,0],K)['accepted'])
        self.assertFalse(audit_ellipse(np.zeros((30,2)),[0,0,60,.5,0],K)['accepted'])

    def test_unknown_intrinsics_are_not_silently_created(self):
        axis=ConsensusAxis(PaperConfig());w=scene(45,40,120,False)
        rs=[rotor_record(circle(w,q)[1]) for q in [(-2,-2),(2,-2),(-2,2),(2,2)]]
        axis.step(0,rs)
        self.assertFalse(axis.diagnostics['calibration_supplied'])
        self.assertIn('unvalidated',axis.diagnostics['interpretation'])

if __name__=='__main__':unittest.main()
