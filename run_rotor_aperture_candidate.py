"""Optional current-peak aperture covariance; no new detector or force law."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import ApertureObservedPredictor
if __name__=='__main__':main(predictor_type=ApertureObservedPredictor,accuracy_first=True)
