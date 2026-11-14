"""Reuse native shared deformation only at a currently supported reference."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import SupportedPlaneObservedPredictor
if __name__=='__main__':main(predictor_type=SupportedPlaneObservedPredictor,accuracy_first=True)
