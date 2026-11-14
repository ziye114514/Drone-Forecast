"""Native rotor prediction with background-evidence coordinate selection."""
from run_rotor_learned_candidate import main
from background_selection_predictor import BackgroundSelectionPredictor
if __name__=='__main__':main(predictor_type=BackgroundSelectionPredictor,accuracy_first=True)
