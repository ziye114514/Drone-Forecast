"""Actual RAW validation of streamed82 using the identical fixed81 cases."""
from pathlib import Path
from types import FunctionType
import argparse,datetime,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage81'),str(ROOT/'research/stage82')]
import run_stream_forecast as entry
import study81 as frozen81
import study82 as module82
load=frozen81.load;save=frozen81.save;sha=frozen81.sha;rows=frozen81.rows
PREV=frozen81.PREV;OUT=ROOT.parent/'output/stage83_RAW_streamed_window_v2_20261021'
cases=frozen81.cases
def sources():
    ns=entry.build();h=ns['fingerprints']();ns.clear();h['research/stage83/study83.py']=sha(__file__)
    # The inherited harness runs unchanged in an isolated function namespace.
    h['research/stage81/study81.py']=sha(ROOT/'research/stage81/study81.py')
    return h
def freeze():
    module82.verify();assert (module82.OUT/'benchmark82.json').is_file() and not OUT.exists()
    p=load(frozen81.OUT/'protocol.json')
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=cases(),
        raw_metadata=p['raw_metadata'],reference_hashes=p['reference_hashes'],module82_checks_sha256=sha(module82.OUT/'checks82.json'),
        benchmark82_sha256=sha(module82.OUT/'benchmark82.json'),
        design='Same fixed seven cases and mathematical49 outputs as81/78. Fresh RAW/NN/rotor '
            'workers, no cached inference inputs. Reuse frozen81 checks in a private namespace '
            'with only output root/source verifier/entry command/deployment stage changed.',
        timing='Recorded matching prefixes across78/81/83; separately paired module82 '
            'benchmark. No simultaneous full-pipeline speed ratio guarantee. No accuracy change.',
        previous_failed_attempt=str(ROOT.parent/'output/stage83_RAW_streamed_window_20261021/failure_postmortem83.json'),
        repair='Keep audit-hook namespace alive through all protocol IO; original failed sources/protocol preserved. No numerical change.',default_promoted=False))
    print('83 actual RAW streamed deployment frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for f,m in p['raw_metadata'].items():
        st=Path(f).stat();assert m==dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return p

# Compile only the original check/run/summarize function definitions. Change
# literal paths/stage/result-file labels; keep all data checks and video loops.
# No source modification, global monkey-patch, or result-cache input to RAW.
import ast,inspect,textwrap
namespace=dict(frozen81.__dict__,ROOT=ROOT,OUT=OUT,PREV=PREV,entry=entry,verify=verify,sources=sources)
for name in ['check','run','summarize_all']:
    source=textwrap.dedent(inspect.getsource(getattr(frozen81,name)))
    source=source.replace('run_exact_forecast.py','run_stream_forecast.py').replace('regression_failure81.json','regression_failure83.json').replace('regression81.json','regression83.json').replace('checks81.json','checks83.json')
    source=source.replace("==81 and not proto", "==83 and not proto").replace('81 actual RAW','83 actual RAW').replace('81 fresh RAW','83 fresh RAW')
    if name=='check':assert '==83 and not proto' in source
    exec(compile(source,'<83 frozen81 '+name+'>','exec'),namespace)
check=namespace['check'];run=namespace['run'];summarize_all=namespace['summarize_all']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check','summarize']);p.add_argument('--case');a=p.parse_args()
    if a.action in ['run','check']:globals()[a.action](a.case)
    elif a.action=='summarize':summarize_all()
    else:globals()[a.action]()
