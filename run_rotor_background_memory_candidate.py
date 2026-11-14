"""One native front, background qualification with existing force memory."""
from run_rotor_learned_candidate import main
from background_selection_predictor import BackgroundMemorySelectionPredictor
if __name__=='__main__':main(predictor_type=BackgroundMemorySelectionPredictor,accuracy_first=True)
