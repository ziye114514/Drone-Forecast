"""Same isolated RAW worker, full-event contour extraction and candidate gate."""
from pathlib import Path
import argparse, hashlib, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage55'),str(ROOT/'research/stage54')]
import rotor_worker55 as previous
from packets54 import PacketReader
from roi55 import validate_hint_request
from gating57 import CandidateFrontend

def worker_source():
    source=previous.worker_source()
    changes={
        "front=RecoveryFrontend(w,h,'neighbor10')":"front=CandidateFrontend(w,h,'neighbor10')",
        'geometry_events,roi_diagnostic=geometry_filter.apply(events,request.get("body_hint"),t);obs,_=front.detect(geometry_events,t)':
        'roi_diagnostic=front.set_hint(events,request.get("body_hint"),t,geometry_roi_mode);obs,_=front.detect(events,t);roi_diagnostic.update(front.gate_counts)',
        'geometry_event_count=len(geometry_events)':'geometry_event_count=len(events)'}
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    return source

def main(name,output,start,width,height,mode):
    reader=PacketReader(name,width,height,start);path=ROOT/'research/stage53/rotor_worker53.py'
    namespace={'__name__':'candidate_worker57','__file__':str(path),'CandidateFrontend':CandidateFrontend,
               'validate_hint_request':validate_hint_request,'geometry_roi_mode':mode}
    exec(compile(worker_source(),str(path),'exec'),namespace);original=namespace['fingerprints']
    def fingerprints():
        result=original()
        for p in [Path(__file__),Path(__file__).with_name('gating57.py'),ROOT/'research/stage55/roi55.py',ROOT/'research/stage55/rotor_worker55.py',ROOT/'research/stage54/packets54.py',ROOT/'research/stage54/rotor_worker54.py']:
            result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace.update(open_raw=lambda unused:reader,raw_chunks=lambda r,s,d,step:r.chunks(),fingerprints=fingerprints)
    try:namespace['main'](None,output,start,0.)
    finally:namespace.clear()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shared-events',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True);p.add_argument('--geometry-roi',choices=['off','body'],required=True)
    a=p.parse_args();main(a.shared_events,a.output,a.start,a.width,a.height,a.geometry_roi)
