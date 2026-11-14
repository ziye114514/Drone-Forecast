"""Local opt-in original author-core observation on the ACTUAL RAW main path."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(Path(__file__).parent),str(ROOT),str(ROOT/'research/stage79'),str(ROOT/'research/stage141')]
from phase_state import RotorPhaseObserver
from run_optimized_forecast import run
from forecast import configuration

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=22.);p.add_argument('--duration',type=float,default=3.)
    p.add_argument('--config',type=Path,default=ROOT/'configs/local.json')
    p.add_argument('--no-reanchor',action='store_true',help='Legacy explicit no whole-model reanchor; now the default')
    p.add_argument('--current-reanchor',action='store_true',help='Experimental whole-model reset; did not improve net coverage')
    p.add_argument('--partial-reanchor',action='store_true',help='Reinitialize only weak slots matched to CURRENT conics; preserve supported slots')
    p.add_argument('--shadow-recovery',action='store_true',help='Preserve primary tracks and validate current-geometry recovery candidates in parallel')
    args=p.parse_args();cfg=configuration(args.config)
    observer=RotorPhaseObserver(allow_current_reanchor=args.current_reanchor and not args.no_reanchor,
        allow_partial_reanchor=args.partial_reanchor,allow_shadow_recovery=args.shadow_recovery)
    try:
        run(args.raw,args.output,args.start,args.duration,False,False,
            pose_solver='lm',curved_rollout=True,parallel_geometry=True,skip_unobserved_geometry=True,
            gpu_python=cfg['gpu_python'],detector_weights=cfg['detector_weights'],
            detector_sha256=cfg['detector_sha256'],cache_dir=cfg['cache_dir'],
            preview_window_ms=5.,event_observer=observer)
    finally:observer.close()

if __name__=='__main__':main()
