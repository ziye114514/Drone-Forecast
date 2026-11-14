"""Shared background reference with correlated image-endpoint errors."""
from run_rotor_learned_candidate import main
from background_reference_predictor import BackgroundEndpointPredictor
if __name__=='__main__':main(predictor_type=BackgroundEndpointPredictor,accuracy_first=True)
