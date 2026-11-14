"""Joint bidirectional translation cost, unchanged rotor/force architecture."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import SymmetricSurfaceObservedPredictor
if __name__=='__main__':main(predictor_type=SymmetricSurfaceObservedPredictor,accuracy_first=True)
