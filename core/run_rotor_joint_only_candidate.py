"""Joint2D subpixel peak only; preserve polarity470 measurement covariance."""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import JointOnlyObservedPredictor
if __name__=='__main__':main(predictor_type=JointOnlyObservedPredictor,accuracy_first=True)
