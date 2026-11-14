"""Past-velocity temporal rectification for motion; original geometry/period."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import TimestampedMotionObservedPredictor
if __name__=='__main__':main(predictor_type=TimestampedMotionObservedPredictor,accuracy_first=True)
