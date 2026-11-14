import unittest
from guarded_position import GuardedPosition

def measured(t,detected=True):
    return dict(t=t,detected=detected,center=[200.,200.],bbox=[160.,160.,240.,240.],confidence=.9,rotors=[],
        observed_rotor_count=4,geometry_quality=.8,n_events=100)

class GuardTests(unittest.TestCase):
    def test_unknown_activity_cannot_acquire(self):
        r=GuardedPosition().step(measured(.1,False),measured(.1))
        self.assertFalse(r['detected'])

    def test_original_measurement_preserved(self):
        g=GuardedPosition();o=measured(.1);r=g.step(o,measured(.1))
        for key in o:self.assertEqual(r[key],o[key])

    def test_consistent_activity_bridges_recent_loss(self):
        g=GuardedPosition();g.step(measured(.1),measured(.1));r=g.step(measured(.2,False),measured(.2))
        self.assertTrue(r['detected']);self.assertEqual(r['geometry_quality'],0.)

    def test_far_activity_rejected(self):
        g=GuardedPosition();g.step(measured(.1),measured(.1));a=measured(.2);a['center']=[700.,600.]
        self.assertFalse(g.step(measured(.2,False),a)['detected'])

    def test_old_confirmation_cannot_track_forever(self):
        g=GuardedPosition();g.step(measured(.1),measured(.1))
        self.assertFalse(g.step(measured(.7,False),measured(.7))['detected'])

if __name__=='__main__':unittest.main()
