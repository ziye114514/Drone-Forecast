"""Shared headless-plane motion on contrast-normalized current rotor evidence."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import LocalContrastPlaneObservedPredictor
if __name__=='__main__':main(predictor_type=LocalContrastPlaneObservedPredictor,accuracy_first=True)
