"""Separate timed run without ORT profiling; retain profiled pilot as EP evidence."""
from common44 import *
def main():
    verify();supplement=OUT/'unprofiled_supplement.json';assert not supplement.exists() and (OUT/'benchmark.json').exists()
    save(supplement,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(Path(__file__)),reason='Original pilot recorded per-node ORT profiling while PyTorch did not profile. Keep its provider traces/parity evidence, but run separate fresh sessions with profiling off for timing fairness. Same inputs/order/FP32/default backend options/warmup/NMS; no threshold/sample tuning. Save separate output.'))
    source=Path(__file__).with_name('benchmark44.py').read_text(encoding='utf-8')
    replacements=[('so.enable_profiling=True','so.enable_profiling=False'),("profile_paths[name]=str(Path(session.end_profiling()).relative_to(OUT))","profile_paths[name]=None"),("for name,profile in profile_paths.items():\n        counts={}","for name,profile in profile_paths.items():\n        if profile is None:continue\n        counts={}"),("OUT/'benchmark.json'","OUT/'benchmark_no_profiling.json'")]
    for a,b in replacements:assert source.count(a)==1,a;source=source.replace(a,b)
    namespace={'__name__':'unprofiled_benchmark44','__file__':str(Path(__file__).with_name('benchmark44.py'))};exec(compile(source,namespace['__file__'],'exec'),namespace);namespace['main']()
if __name__=='__main__':main()
