"""Optional joint2D peak refinement and aperture covariance on native rotors."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import JointPeakObservedPredictor
if __name__=='__main__':main(predictor_type=JointPeakObservedPredictor,accuracy_first=True)
