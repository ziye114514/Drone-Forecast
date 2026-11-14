"""Accuracy-first rotor candidate with qualified plane/line force coordinates."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import ProjectionQualifiedObservedPredictor

if __name__=='__main__':main(predictor_type=ProjectionQualifiedObservedPredictor,accuracy_first=True)
