"""Instrument the frozen42 worker without changing model/input computations."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]

def main():
    protocol=json.loads((ROOT.parent/'output/stage43_latency_profile_20261019/protocol.json').read_text(encoding='utf-8'))
    path=ROOT/'research/stage42/worker42.py';assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['source_hashes']['research/stage42/worker42.py']
    source=path.read_text(encoding='utf-8')
    changes=[("data=base64.b64decode(request['png'],validate=True);image=", "decode_started=time.perf_counter();data=base64.b64decode(request['png'],validate=True);image="),
        ("assert image.shape==(720,1280,3)","assert image.shape==(720,1280,3);decode_ms=(time.perf_counter()-decode_started)*1000"),
        ("inference_ms=elapsed)","inference_ms=elapsed,profile=dict(decode_ms=decode_ms,ultralytics_speed_ms=result.speed))")]
    for old,new in changes:assert source.count(old)==1,old;source=source.replace(old,new)
    namespace={'__name__':'profiled_worker42','__file__':str(path)};exec(compile(source,str(path),'exec'),namespace);namespace['main']()
if __name__=='__main__':main()
