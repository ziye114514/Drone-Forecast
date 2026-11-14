"""Native rotor observation recovery; no accuracy/FPS success promised."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import ReferenceGuardedObservedPredictor

if __name__=='__main__':main(predictor_type=ReferenceGuardedObservedPredictor,accuracy_first=True)
