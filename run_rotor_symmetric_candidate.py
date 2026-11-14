"""Symmetric native rotor translation; same five-circle and force model."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import SymmetricObservedPredictor
if __name__=='__main__':main(predictor_type=SymmetricObservedPredictor,accuracy_first=True)
