"""Execute preserved53 rotor math using acknowledged shared native events."""
from pathlib import Path
import argparse,hashlib
from packets54 import PacketReader
ROOT=Path(__file__).resolve().parents[2]

def worker_source():
    path=ROOT/'research/stage53/rotor_worker53.py';source=path.read_text(encoding='utf-8')
    changes={
        "raw=Path(raw).resolve();output=Path(output).resolve()":"output=Path(output).resolve()",
        "requests+=1\n                last_record":"requests+=1;reader.set_request(request)\n                last_record",
        "record=last_record,request_wall_ms=":"record=last_record,event_packet_sha256=reader.payload_sha256,request_wall_ms=",
        "    finally:\n        summary=":"    finally:\n        stream.close();reader.close()\n        summary=",
        "inputs='RAW only; no sensing/geometry cache'":"inputs='Shared current native RAW events only; no file decoder, no sensing/geometry cache'"}
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    return source

def main(name,output,start,width,height):
    reader=PacketReader(name,width,height,start);path=ROOT/'research/stage53/rotor_worker53.py';namespace={'__name__':'shared_native54','__file__':str(path)}
    exec(compile(worker_source(),str(path),'exec'),namespace);original=namespace['fingerprints']
    def fingerprints():
        result=original()
        for p in [Path(__file__),Path(__file__).with_name('packets54.py')]:result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace.update(open_raw=lambda unused:reader,raw_chunks=lambda r,s,d,step:r.chunks(),fingerprints=fingerprints)
    try:namespace['main'](None,output,start,0.)
    finally:namespace.clear()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shared-events',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True);a=p.parse_args();main(a.shared_events,a.output,a.start,a.width,a.height)
