"""Original shared RAW worker with paired observations; no cached hints/labels."""
from pathlib import Path
import argparse,hashlib,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage57'))
import rotor_worker57 as previous
from packets54 import PacketReader
from roi55 import validate_hint_request
from frontend60 import LineFrontend

def worker_source():
    source=previous.worker_source();changes={
      "front=CandidateFrontend(w,h,'neighbor10')":"front=LineFrontend(w,h,'neighbor10','off' if geometry_roi_mode=='off' else 'paired')",
      'front.set_hint(events,request.get("body_hint"),t,geometry_roi_mode)':'front.set_hint(events,request.get("body_hint"),t,"body")',
      'roi_diagnostic.update(front.gate_counts)':'roi_diagnostic.update(front.gate_counts);roi_diagnostic["line_audit"]=front.line_audit'}
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    return source

def main(name,output,start,width,height,mode):
    reader=PacketReader(name,width,height,start);path=ROOT/'research/stage53/rotor_worker53.py'
    namespace={'__name__':'paired60','__file__':str(path),'LineFrontend':LineFrontend,'validate_hint_request':validate_hint_request,'geometry_roi_mode':mode}
    exec(compile(worker_source(),str(path),'exec'),namespace);original=namespace['fingerprints']
    def fingerprints():
        result=original()
        for name in ['rotor_worker60.py','frontend60.py','pairs60.py','profile60.py']:
            p=Path(__file__).with_name(name);result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace.update(open_raw=lambda unused:reader,raw_chunks=lambda r,s,d,step:r.chunks(),fingerprints=fingerprints)
    try:namespace['main'](None,output,start,0.)
    finally:namespace.clear()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shared-events',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True);p.add_argument('--geometry-roi',choices=['off','body'],required=True)
    a=p.parse_args();main(a.shared_events,a.output,a.start,a.width,a.height,a.geometry_roi)
