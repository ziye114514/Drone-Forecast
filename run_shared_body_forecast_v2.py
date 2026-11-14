"""Correct subprocess constants in the preserved47 transport adapter."""
from pathlib import Path
import argparse,hashlib,importlib.util,subprocess
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('preserved_shared47',ROOT/'run_shared_body_forecast.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
original_build=base.build
def build(*args,**kwargs):
    ns,memory,workers=original_build(*args,**kwargs);ns['subprocess'].PIPE=subprocess.PIPE
    for name in ['Popen','PIPE','TimeoutExpired']:assert hasattr(ns['subprocess'],name)
    original=ns['fingerprints']
    def fingerprints():
        result=original();result[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return result
    ns['fingerprints']=fingerprints;return ns,memory,workers
base.build=build
run=base.run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.);p.add_argument('--precision',type=int,choices=[16,32],default=32);p.add_argument('--body-only',action='store_true');p.add_argument('--verify-frames',action='store_true');p.add_argument('--preview',action='store_true');a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.precision,a.body_only,a.verify_frames,a.preview)
