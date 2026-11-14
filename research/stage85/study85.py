"""Fresh RAW full49 regression after batched84 projection controls."""
from pathlib import Path
import argparse,datetime,inspect,sys,textwrap
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage81'),str(ROOT/'research/stage84')]
import run_batched_forecast as entry
import study81 as frozen81
import study84 as module84
load=frozen81.load;save=frozen81.save;sha=frozen81.sha;rows=frozen81.rows
PREV=frozen81.PREV;OUT=ROOT.parent/'output/stage85_RAW_batched_rollout_20261021'
cases=frozen81.cases
def sources():
    ns=entry.build();h=ns['fingerprints']();ns.clear();h['research/stage85/study85.py']=sha(__file__)
    h['research/stage81/study81.py']=sha(ROOT/'research/stage81/study81.py');return h
def freeze():
    module84.verify();assert (module84.OUT/'benchmark84.json').is_file() and not OUT.exists()
    p=load(frozen81.OUT/'protocol.json')
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),cases=cases(),
        raw_metadata=p['raw_metadata'],reference_hashes=p['reference_hashes'],module84_checks_sha256=sha(module84.OUT/'checks84.json'),
        benchmark84_sha256=sha(module84.OUT/'benchmark84.json'),
        design='Same fixed seven cases as81/83, fresh RAW/NN/rotor workers. '
            'All49/current states strict exact78 plus actual truncated prefixes '
            'and complete video decoding. No inference cache or future labels.',
        timing='Measured matching old78 prefixes, sequential recorded81/83/85; '
            'independent paired module84 benchmark separate. Accuracy unchanged.',default_promoted=False))
    print('85 actual RAW batched deployment frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for f,m in p['raw_metadata'].items():
        st=Path(f).stat();assert m==dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return p
namespace=dict(frozen81.__dict__,ROOT=ROOT,OUT=OUT,PREV=PREV,entry=entry,verify=verify,sources=sources)
for name in ['check','run','summarize_all']:
    source=textwrap.dedent(inspect.getsource(getattr(frozen81,name)))
    source=source.replace('run_exact_forecast.py','run_batched_forecast.py').replace('regression_failure81.json','regression_failure85.json').replace('regression81.json','regression85.json').replace('checks81.json','checks85.json')
    source=source.replace('==81 and not proto','==85 and not proto').replace('81 actual RAW','85 actual RAW').replace('81 fresh RAW','85 fresh RAW').replace('exact81','exact85')
    if name=='check':assert '==85 and not proto' in source
    exec(compile(source,'<85 frozen81 '+name+'>','exec'),namespace)
check=namespace['check'];run=namespace['run'];summarize_all=namespace['summarize_all']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','check','summarize']);p.add_argument('--case');a=p.parse_args()
    if a.action in ['run','check']:globals()[a.action](a.case)
    elif a.action=='summarize':summarize_all()
    else:globals()[a.action]()
