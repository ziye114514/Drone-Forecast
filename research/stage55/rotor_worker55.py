"""Reuse54 isolated current RAW worker; gate geometry only by past body box."""
from pathlib import Path
import argparse,hashlib,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage54'))
from packets54 import PacketReader
import rotor_worker54 as previous
from roi55 import GeometryROI,validate_hint_request

def worker_source():
    source=previous.worker_source();changes={
        'reader.set_request(request)':'reader.set_request(request);validate_hint_request(request);assert request["geometry_roi"]==geometry_roi_mode',
        'tick=time.perf_counter();obs,_=front.detect(events,t)':'tick=time.perf_counter();geometry_events,roi_diagnostic=geometry_filter.apply(events,request.get("body_hint"),t);obs,_=front.detect(geometry_events,t)',
        'processing_ms=elapsed,eof_empty_padding=padding)':'processing_ms=elapsed,eof_empty_padding=padding,roi_diagnostic=roi_diagnostic,geometry_event_count=len(geometry_events))'}
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    return source
def main(name,output,start,width,height,mode):
    reader=PacketReader(name,width,height,start);path=ROOT/'research/stage53/rotor_worker53.py';namespace={'__name__':'roi_worker55','__file__':str(path),
        'geometry_filter':GeometryROI(width,height,mode),'validate_hint_request':validate_hint_request,'geometry_roi_mode':mode}
    exec(compile(worker_source(),str(path),'exec'),namespace);original=namespace['fingerprints']
    def fingerprints():
        result=original()
        for p in [Path(__file__),Path(__file__).with_name('roi55.py'),ROOT/'research/stage54/packets54.py',ROOT/'research/stage54/rotor_worker54.py',ROOT/'research/stage17/frontend.py']:result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace.update(open_raw=lambda unused:reader,raw_chunks=lambda r,s,d,step:r.chunks(),fingerprints=fingerprints)
    try:namespace['main'](None,output,start,0.)
    finally:namespace.clear()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shared-events',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True);p.add_argument('--geometry-roi',choices=['off','body'],required=True);a=p.parse_args();main(a.shared_events,a.output,a.start,a.width,a.height,a.geometry_roi)
