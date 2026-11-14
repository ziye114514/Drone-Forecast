import unittest,copy
import numpy as np
from raw_pipeline import ConsensusBank
from experiments import K,scene,circle,rotor_record
from event_drone_forecast.paper_config import PaperConfig
from event_drone_forecast.paper_bank import PaperForecastBank

def inputs(t,axis_only=False):
    w=scene(45,40,120,False)
    rotors=[dict(rotor_record(circle(w,q)[1]),observed_now=True,axis_only=axis_only,
                 measurement_audit=dict(accepted=not axis_only,normal_sigma_deg=2.,quality_multiplier=1.)) for q in [(-2,-2),(2,-2),(-2,2),(2,2)]]
    o=dict(t=t,detected=True,center=[640+20*t,360+10*t],confidence=.95,geometry_quality=.9,rotors=rotors)
    s=dict(rotors=rotors,four_visible=True,four_frequencies=True,thrust_ratio=1.,frequency_quality=.95,differential_proxy=0.)
    return o,s

class IntegrationTests(unittest.TestCase):
    def test_full_measurement_reaches_frozen_bank_reporting(self):
        bank=ConsensusBank(PaperConfig(camera_matrix=K.tolist()),True)
        for t in np.arange(80)*.01:r=bank.step(*inputs(float(t)))
        self.assertTrue(r['physics_active']);self.assertEqual(r['image_differential_gain']['paper_adaptive'],0.)
        self.assertTrue(r['normal_measurements']['paper_adaptive']['accepted'])

    def test_rejected_new_observation_cannot_train_on_old_normal(self):
        bank=ConsensusBank(PaperConfig(),True);bank.step(*inputs(0.))
        r=bank.step(*inputs(.01,True))
        self.assertFalse(r['physics_active']);self.assertFalse(bank.models['paper_adaptive'].update_allowed)

    def test_cv_is_unchanged_by_normal_ablation(self):
        banks=[PaperForecastBank(PaperConfig()),ConsensusBank(PaperConfig(),False),ConsensusBank(PaperConfig(),True)]
        for t in np.arange(20)*.01:
            o,s=inputs(float(t));r=[b.step(copy.deepcopy(o),copy.deepcopy(s)) for b in banks]
            for a in r[1:]:self.assertEqual(a['states']['cv'],r[0]['states']['cv'])

if __name__=='__main__':unittest.main()
