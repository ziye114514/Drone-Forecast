"""Pure-rotor motion appearance uncertainty, using the existing joint shift."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import PolarityPartialObservedPredictor

if __name__=='__main__':main(predictor_type=PolarityPartialObservedPredictor,accuracy_first=True)
