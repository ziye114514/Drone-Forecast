"""Optional positive normal-drive prior; no metric thrust measurement claim."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import PositivePriorObservedPredictor

if __name__=='__main__':main(predictor_type=PositivePriorObservedPredictor,accuracy_first=True)
