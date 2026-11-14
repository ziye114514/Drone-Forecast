"""Selected cumulative research configuration; original entry stays available.

No30px/30Hz guarantee. Explicit existing GPU Python and locally built native
DLL required. Runtime records actual age, drops and valid publication rate.
"""
from run_rotor_only import make_parser,run

def main():
    p=make_parser();p.description=__doc__
    p.set_defaults(rotor_gpu_front=True,rotor_native_gpu=True,rotor_fast_geometry=True,
        rotor_remote_front=True,rotor_no_phase=True,rotor_local_phase=True,
        rotor_subpixel_motion=True,rotor_entry_init=True,rotor_relative_observation=True,
        rotor_relock_support_guard=True,rotor_border_motion=True,
        rotor_gap_prediction='short',rotor_reference_mode='fixed',rotor_unknown_force_model='plane-adaptive',
        rotor_regional_graph=True,rotor_projected_fov_guard=True,rotor_curvature_hz=0.,
        rotor_prefetch=True,rotor_reader_ready=True,rotor_reader='cooperative',
        rotor_max_backlog_ms=66.667,paced=True,no_preview=True)
    args=p.parse_args()
    if not args.gpu_python:p.error('Specify --gpu-python with the existing CUDA Python interpreter')
    if not args.rotor_native_geometry_dll:p.error('Build tools/build_uncertainty216.ps1 and specify --rotor-native-geometry-dll')
    run(args)

if __name__=='__main__':main()
