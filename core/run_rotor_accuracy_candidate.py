"""Accuracy research: pure rotors,20ms envelope forecast, all packets retained.

The20ms observation target matches the declared native reference protocol;
it is not instantaneous mass-centre truth or a realtime/FPS certificate.
"""
from run_rotor_learned_candidate import main
from observed_rotor_predictor import AccuracyObservedPredictor

if __name__=='__main__':main(predictor_type=AccuracyObservedPredictor,accuracy_first=True)
