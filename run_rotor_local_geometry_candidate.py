"""Optional accuracy-first local-conic opportunity, with old entries retained."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import LocalGeometryObservedPredictor

if __name__=='__main__':main(predictor_type=LocalGeometryObservedPredictor,accuracy_first=True)
