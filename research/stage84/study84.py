"""Reuse80 strict regression/paired benchmark harness for batched rollouts."""
from pathlib import Path
import argparse,datetime,inspect,sys,textwrap
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage80'),str(ROOT/'research/stage82')]
import study80 as frozen80
import study82 as module82
import batched84
load=frozen80.load;save=frozen80.save;sha=frozen80.sha;rows=frozen80.rows
PREV=frozen80.PREV;OUT=ROOT.parent/'output/stage84_batched_rollout_20261021'
def sources():
    files=[Path(__file__),Path(__file__).with_name('batched84.py'),Path(__file__).with_name('RELATED_WORK.md'),ROOT/'research/stage80/study80.py',ROOT/'research/stage80/exact80.py',ROOT/'research/stage82/stream82.py']
    return {str(p):sha(p) for p in files}
def freeze():
    module82.verify();assert not OUT.exists()
    assert (ROOT.parent/'output/stage83_RAW_streamed_window_v2_20261021/checks83.json').is_file()
    p=load(PREV/'protocol.json')
    save(OUT/'protocol.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources(),
        frozen78_sources=p['source_hashes'],cases=p['cases'],input_hashes={c:sha(PREV/c/'observations_forecasts.jsonl') for c in p['cases']},
        raw_metadata=p['raw_metadata'],parameters=batched84.PARAMS,controls=batched84.controls(),
        benchmark_cases=['clear6','side6','extra2'],benchmark_duration_s=2.,
        design='Reuse80 whole49/current/immutability regression and paired benchmark '
            'in isolated namespace. New batched future projection, same author '
            'filter state/update/probability/physics. Strict whole-tree equality '
            'to78 required first; preserve any failure rather than weakening checks.',
        benchmark='OriginalExtras=stream82, new=batch84; alternate execution order. '
            'No other project experiment running. Separate cProfile, not full pipeline.',default_promoted=False))
    print('84 batched future projection protocol frozen',flush=True)
def verify():
    p=load(OUT/'protocol.json');assert p['source_hashes']==sources()
    for f,h in p['frozen78_sources'].items():assert sha(ROOT/f)==h
    for f,m in p['raw_metadata'].items():
        st=Path(f).stat();assert m==dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    return p
namespace=dict(frozen80.__dict__,ROOT=ROOT,OUT=OUT,PREV=PREV,verify=verify,sources=sources,
    OnlineExtras=batched84.OnlineExtras,OriginalExtras=module82.OnlineExtras)
for name in ['base_forecast','evaluate_tick','run','summarize_all','benchmark']:
    source=textwrap.dedent(inspect.getsource(getattr(frozen80,name)))
    source=source.replace('checks80.json','checks84.json').replace('benchmark80.json','benchmark84.json')
    source=source.replace('old78','stream82_reference').replace('exact80','batch84')
    source=source.replace('80 paired','84 paired').replace('80 all','84 all')
    exec(compile(source,'<84 frozen80 '+name+'>','exec'),namespace)
run=namespace['run'];summarize_all=namespace['summarize_all'];benchmark=namespace['benchmark']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','summarize','benchmark']);p.add_argument('--case');a=p.parse_args()
    if a.action=='run':run(a.case)
    elif a.action=='summarize':summarize_all()
    else:globals()[a.action]()
