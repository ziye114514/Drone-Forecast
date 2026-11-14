"""Native rotor count matching with local photometric offset/scale removal."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import LocalContrastObservedPredictor
if __name__=='__main__':main(predictor_type=LocalContrastObservedPredictor,accuracy_first=True)
