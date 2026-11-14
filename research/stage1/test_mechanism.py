import unittest,copy
import numpy as np
from mechanism import *

class MechanismChecks(unittest.TestCase):
    def test_id_only_permutation_preserves_force_and_predictions(self):
        world=make_world(3,duration=1.)
        a,_=run_trial(world,'clean',start=.4,end=.8)
        b,_=run_trial(world,'id_permutation_only',start=.4,end=.8)
        np.testing.assert_array_equal(a['prediction'],b['prediction'])
        np.testing.assert_array_equal(a['parameters'],b['parameters'])

    def test_frequency_doubling_quadruples_relative_force(self):
        world=make_world(3,duration=1.)
        a,_=run_trial(world,'clean',start=.4,end=.8)
        b,_=run_trial(world,'double_confident',start=.4,end=.8)
        k=25
        self.assertAlmostEqual(b['relative'][k]/a['relative'][k],4.)

    def test_low_quality_freezes_quality_arm_but_not_ordinary(self):
        world=make_world(3,duration=1.5)
        a,_=run_trial(world,'double_flagged',start=.8,end=1.2)
        t=a['times'];k=np.flatnonzero((t>=.8)&(t<1.2))
        self.assertTrue(a['allowed'][k,2].all());self.assertFalse(a['allowed'][k,3].any())
        np.testing.assert_array_equal(a['parameters'][k[0],3],a['parameters'][k[-1],3])

    def test_input_fully_restores_after_fault(self):
        world=make_world(7,duration=1.5)
        for scenario in SCENARIOS:
            a=sensor_sample(world,65,scenario,start=.5,end=1.1);b=sensor_sample(world,65,'clean',start=.5,end=1.1)
            for key in ('position','frequencies','normal','quality','four_visible'):
                np.testing.assert_equal(a[key],b[key])

    def test_world_not_mutated_by_corruptions(self):
        world=make_world(3,duration=1.);before=copy.deepcopy(world)
        for scenario in SCENARIOS:
            for i in range(len(world['times'])):sensor_sample(world,i,scenario,start=.2,end=.8)
        for key in world:np.testing.assert_equal(world[key],before[key])

    def test_prefix_does_not_read_future(self):
        world=make_world(9,duration=1.5);changed=copy.deepcopy(world)
        for key in ('noisy_position','noisy_frequency','noisy_angle','noisy_aspect'):
            changed[key][51:]+=20
        a,_=run_trial(world,'clean',limit=1.)
        b,_=run_trial(changed,'clean',limit=1.)
        np.testing.assert_array_equal(a['prediction'],b['prediction'])

    def test_exact_rollout_matches_frozen_kernel(self):
        model=ProjectedThrustModel(PaperConfig(),True);model.parameters=np.array([230.,190.,.8,.3]);model.disturbance=np.array([4.,-7.])
        n=np.array([.2,-.9,np.sqrt(.15)]);p=np.array([4.,7.]);v=np.array([30.,-15.])
        reference=model.rollout(p,v,n,np.zeros(3),1.1,HORIZONS)[1]
        np.testing.assert_allclose(exact_rollout(p,v,n,1.1,model),reference,atol=1e-8)

    def test_frequency_gain_scale_symmetry(self):
        model=ProjectedThrustModel(PaperConfig(),False);model.parameters=np.array([220.,200.,0.,0.])
        a=exact_rollout([0,0],[10,2],[.2,-.9,.3],1.,model)
        model.parameters[0]/=4
        b=exact_rollout([0,0],[10,2],[.2,-.9,.3],4.,model)
        np.testing.assert_allclose(a,b)

if __name__=='__main__':unittest.main(verbosity=2)
