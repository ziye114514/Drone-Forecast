"""Preserve ON and OFF as channels in native rotor-region registration."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import MultichannelObservedPredictor
if __name__=='__main__':main(predictor_type=MultichannelObservedPredictor,accuracy_first=True)
