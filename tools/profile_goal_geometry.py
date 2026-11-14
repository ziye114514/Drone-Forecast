"""Profile only the current geometry step in one real RAW run, not FPS."""
from pathlib import Path
import sys,cProfile,pstats,io,json,runpy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from fast_structure_front import FastStructure


def main(output,extra=()):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    profiler=cProfile.Profile();original=FastStructure.step
    def measured(self,*args,**kwargs):
        profiler.enable()
        try:return original(self,*args,**kwargs)
        finally:profiler.disable()
    FastStructure.step=measured
    try:
        sys.argv=[str(ROOT/'core/run_fast_structure.py'),'--raw','E:/dataset/train/6/media/event.raw',
            '--output',str(out/'raw'),'--start','26','--duration','2','--goal-profile','--paced',*extra]
        runpy.run_path(sys.argv[0],run_name='__main__')
    finally:FastStructure.step=original
    profiler.dump_stats(str(out/'geometry.pstats'))
    buf=io.StringIO();pstats.Stats(profiler,stream=buf).strip_dirs().sort_stats('cumulative').print_stats(30)
    (out/'geometry.txt').write_text(buf.getvalue(),'utf-8')
    (out/'scope.json').write_text(json.dumps(dict(profiled_geometry_only=True,FPS_is_not_a_benchmark=True,
        original_thresholds_and_source_inputs=True,wall_clock_profiling_may_change_async_reply_times=True)), 'utf-8')
    (out/'profile_goal_geometry.py').write_bytes(Path(__file__).read_bytes())
    print(buf.getvalue())


if __name__=='__main__':main(sys.argv[1],sys.argv[2:])
