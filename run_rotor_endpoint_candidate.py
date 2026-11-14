"""Accuracy-first rotor candidate with shared-image endpoint error states."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import EndpointObservedPredictor

if __name__=='__main__':main(predictor_type=EndpointObservedPredictor,accuracy_first=True)
