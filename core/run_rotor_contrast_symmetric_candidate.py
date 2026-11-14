"""Local rotor-count contrast and inverse-consistent native translation."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import LocalContrastSymmetricObservedPredictor
if __name__=='__main__':main(predictor_type=LocalContrastSymmetricObservedPredictor,accuracy_first=True)
