"""Stable line-first observations in the original shared RAW/frequency worker."""
from pathlib import Path
import argparse,hashlib,sys
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage60'))
import rotor_worker60 as previous
from packets54 import PacketReader
from roi55 import validate_hint_request
from preline61 import StablePairedFrontend
import runtime_full61

def worker_source():
    source=previous.worker_source();changes={
      'roi_diagnostic["line_audit"]=front.line_audit':'roi_diagnostic["line_audit"]=front.line_audit;roi_diagnostic["preline_audit"]=front.preline_audit',
      'step_us=10000,source_hashes=hashes':'step_us=10000,opencv_runtime=runtime_info,opencv_threads=cv2.getNumThreads(),source_hashes=hashes',
      'labels_read=False)':'labels_read=False,opencv_runtime=runtime_info,opencv_threads=cv2.getNumThreads())'}
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    return source

def main(name,output,start,width,height,mode):
    runtime_info=runtime_full61.verify();reader=PacketReader(name,width,height,start);path=ROOT/'research/stage53/rotor_worker53.py'
    namespace={'__name__':'stable61','__file__':str(path),'LineFrontend':StablePairedFrontend,'validate_hint_request':validate_hint_request,
       'geometry_roi_mode':mode,'runtime_info':runtime_info,'cv2':cv2}
    exec(compile(worker_source(),str(path),'exec'),namespace);original=namespace['fingerprints']
    def fingerprints():
        result=original()
        for p in [Path(__file__),Path(__file__).with_name('preline61.py'),Path(__file__).with_name('runtime_full61.py'),ROOT/'research/stage60/rotor_worker60.py',ROOT/'research/stage60/frontend60.py',ROOT/'research/stage60/pairs60.py',ROOT/'research/stage60/profile60.py']:
            result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace.update(open_raw=lambda unused:reader,raw_chunks=lambda r,s,d,step:r.chunks(),fingerprints=fingerprints)
    try:namespace['main'](None,output,start,0.)
    finally:namespace.clear()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shared-events',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True);p.add_argument('--geometry-roi',choices=['off','body'],required=True)
    a=p.parse_args();main(a.shared_events,a.output,a.start,a.width,a.height,a.geometry_roi)
