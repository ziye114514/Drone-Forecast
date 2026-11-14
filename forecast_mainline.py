"""Recommended ongoing-development entry: accumulated version by default."""
import sys
from forecast import main
if __name__=='__main__':
    args=sys.argv[1:]
    if args and args[0]=='structure-fast':
        sys.argv=[sys.argv[0]]+args[1:]
        from run_fast_structure import main as fast_main
        raise SystemExit(fast_main())
    if args and args[0]=='structure':
        # User-selected mechanism3; independent/four-only arms remain controls.
        sys.argv=[sys.argv[0]]+args[1:]
        from run_five_circle_v3 import main as structure_main
        raise SystemExit(structure_main())
    if args and args[0] in ('run','doctor') and not any(v=='--profile' or v.startswith('--profile=') for v in args):
        args+=['--profile','live-accumulated']
    raise SystemExit(main(args,allowed_profiles=('live-accumulated','live-baseline'),default_profile='live-accumulated'))
