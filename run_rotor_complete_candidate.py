"""Accuracy-first native rotor identity/position/reference qualification."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import CompleteObservedPredictor

if __name__=='__main__':main(predictor_type=CompleteObservedPredictor,accuracy_first=True)
