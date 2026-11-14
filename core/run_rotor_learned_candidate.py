"""Experimental native-rotor learned-response candidate; no35px/30FPS promise."""
from pathlib import Path
from functools import partial
from run_rotor_only import make_parser,run
from learned_rotor_predictor import LearnedRotorPredictor


def main(predictor_type=LearnedRotorPredictor,accuracy_first=False):
    p=make_parser();p.description=__doc__
    p.add_argument('--response-model',type=Path,default=Path(__file__).resolve().parents[1]/'models/native_response_plane_v1.json',
        help='Fixed source-trained native unknown-force prior; never selected by recording ID')
    p.set_defaults(rotor_gpu_front=True,rotor_native_gpu=True,rotor_fast_geometry=True,
        rotor_remote_front=True,rotor_no_phase=True,rotor_local_phase=True,
        rotor_subpixel_motion=True,rotor_entry_init=True,rotor_relative_observation=True,
        rotor_relock_support_guard=True,rotor_border_motion=True,
        rotor_gap_prediction='short',rotor_reference_mode='fixed',rotor_unknown_force_model='plane-adaptive',
        rotor_regional_graph=True,rotor_projected_fov_guard=True,rotor_curvature_hz=0.,
        rotor_prefetch=True,rotor_reader_ready=True,rotor_reader='cooperative',
        rotor_max_backlog_ms=66.667,paced=True,no_preview=True,
        rotor_inverse_normal_response=True,rotor_anisotropic_response=True)
    if accuracy_first:
        # Preserve every source observation for the accuracy stage. This is
        # offline evaluation, not a realtime/FPS achievement.
        p.set_defaults(paced=False,rotor_max_backlog_ms=0.)
    args=p.parse_args()
    if not args.gpu_python:p.error('Specify --gpu-python with the existing CUDA Python interpreter')
    if not args.rotor_native_geometry_dll:p.error('Specify the existing --rotor-native-geometry-dll')
    run(args,predictor_factory=partial(predictor_type,response_model=args.response_model))


if __name__=='__main__':main()
