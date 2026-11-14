"""RAW-only research entry: five-circle structure + visibility switching (3)."""
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=22.);p.add_argument('--duration',type=float,default=2.)
    p.add_argument('--strategy',type=int,choices=[1,2,3],default=3)
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    p.add_argument('--gpu-python');p.add_argument('--detector-weights');p.add_argument('--detector-sha256');p.add_argument('--cache-dir')
    p.add_argument('--rotor-threshold',type=float,default=4.)
    a=p.parse_args();root=Path(__file__).resolve().parent
    from structure_v3 import StructureV3
    import run_optimized_forecast as entry
    if a.detector_weights is None:
        a.detector_weights=str(root/'models/live_position_accumulated_v1.onnx')
        a.detector_sha256='ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4'
    transformer=StructureV3(a.strategy)
    entry.run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames,
        rotor_inertia=True,rotor_threshold=a.rotor_threshold,pose_solver='lm',
        curved_rollout=True,skip_unobserved_geometry=True,preview_window_ms=5.,
        gpu_python=a.gpu_python,detector_weights=a.detector_weights,
        detector_sha256=a.detector_sha256,cache_dir=a.cache_dir,branch_transform=transformer)


if __name__=='__main__':main()
