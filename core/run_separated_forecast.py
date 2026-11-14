"""RAW-only protected CV / physical forecasts with current-input admission."""
from pathlib import Path
import argparse,ast,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'research/stage78')]
import run_hybrid_forecast as previous
from online78 import OnlineExtras
def build():
    ns=previous.build();source=previous.previous.previous.pipeline_source()
    substitutions={
        "[('trusted_acc',(60,190,70)),('force_periodic',(190,80,210))]":
        "[('cv',(220,150,50)),('dual_cv_event_increment',(190,80,210)),('mmae_state_fresh',(40,220,240))]",
        'green=history purple=local-periodic-force':'blue=CV purple=old-IMM yellow=isolated',
        'q["event_force_diagnostic"]["models"]["force_periodic"]["active"]':'q["separated_diagnostic"]["current"]',
        '| updates={q["event_force_diagnostic"]["models"]["force_periodic"]["updates"]}':
        '| weight={((q["separated_diagnostic"]["models"]["mmae_state_fresh"]["mode_probability"] or [1.,0.])[1] if q["separated_diagnostic"]["current"] else 0.):.2f}',
        'event_force_stage=71':'event_force_stage=78','additional_forecasts=11':'additional_forecasts=34'
    }
    for old,new in substitutions.items():assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/'core/run_joint_forecast.py'),'exec'),ns)
    ns['OnlineExtras']=OnlineExtras;old=ns['fingerprints']
    def fingerprints():
        result=old()
        for p in [Path(__file__),ROOT/'research/stage78/online78.py',ROOT/'research/stage77/separated77.py',
                  ROOT/'external/filterpy_mmae_upstream/mmae.py',ROOT/'external/filterpy_mmae_upstream/LICENSE',ROOT/'external/filterpy_mmae_upstream/UPSTREAM_SNAPSHOT.json']:
            result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    ns['fingerprints']=fingerprints
    return ns
def run(raw,output,start=0.,duration=0.,preview=False,verify_frames=False):
    ns=build()
    def receive(worker,kind):
        line=worker.stdout.readline()
        if not line:raise RuntimeError(kind+' worker stopped; inspect log')
        try:return json.loads(line)
        except json.JSONDecodeError:
            Path(output,'ipc_failure.json').write_text(json.dumps(dict(kind=kind,raw_line=line,process_poll=worker.poll()),ensure_ascii=False),encoding='utf-8');raise
    ns['receive']=receive
    try:return ns['run'](raw,output,start,duration,32,preview,verify_frames)
    finally:ns.clear()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames)
