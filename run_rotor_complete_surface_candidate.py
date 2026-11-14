"""Reciprocal image cost only for two complete native rotor supports."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import CompleteSymmetricSurfaceObservedPredictor
if __name__=='__main__':main(predictor_type=CompleteSymmetricSurfaceObservedPredictor,accuracy_first=True)
