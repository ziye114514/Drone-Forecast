"""Pure-rotor partial-input and uncertainty candidate; all packets retained."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import PartialForceObservedPredictor

if __name__=='__main__':main(predictor_type=PartialForceObservedPredictor,accuracy_first=True)
