"""One bounded full RAW profile; profiled times are not a speed benchmark."""
from pathlib import Path
import sys,json,cProfile,pstats,io,runpy
ROOT=Path(__file__).resolve().parents[1]


def main(output):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    # Snapshot the comparison sources before any next optimization edits.
    snapshot=out/'baseline_source';snapshot.mkdir()
    for f in ['run_fast_structure.py','fast_structure_front.py','src/event_drone_forecast/geometry.py',
        'research/stage147/curvature147.py','research/stage6/silhouette.py','shared_arc_recovery.py','partial_arc_information.py']:
        p=snapshot/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    sys.argv=[str(ROOT/'run_fast_structure.py'),'--raw','E:/dataset/train/6/media/event.raw',
        '--output',str(out/'raw'),'--start','110','--duration','2','--async-nn','--paced','--verify-frames','--reference-compute']
    prof=cProfile.Profile();prof.enable();runpy.run_path(sys.argv[0],run_name='__main__');prof.disable()
    prof.dump_stats(str(out/'raw.pstats'))
    buf=io.StringIO();pstats.Stats(prof,stream=buf).strip_dirs().sort_stats('cumulative').print_stats(35)
    (out/'profile.txt').write_text(buf.getvalue(),'utf-8');print(buf.getvalue(),flush=True)


if __name__=='__main__':main(sys.argv[1])
