"""Rotor physics in a causal shared-background translation reference."""
from run_rotor_learned_candidate import main
from background_reference_predictor import BackgroundTranslationPredictor
if __name__=='__main__':main(predictor_type=BackgroundTranslationPredictor,accuracy_first=True)
