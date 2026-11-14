from pathlib import Path
import pstats
ROOT=Path(__file__).resolve().parents[2]
stats=pstats.Stats(str(ROOT.parent/'output/stage137_fast_geometry_20261024/baseline_profile_absolute.pstats'))
stats.strip_dirs().sort_stats('cumulative').print_stats(35)
stats.sort_stats('tottime').print_stats(22)
