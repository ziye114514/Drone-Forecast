"""Joint2D peak vertex inside the measured3x3 neighborhood, old covariance."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import FullJointObservedPredictor
if __name__=='__main__':main(predictor_type=FullJointObservedPredictor,accuracy_first=True)
