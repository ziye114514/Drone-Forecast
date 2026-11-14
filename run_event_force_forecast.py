"""Experimental RAW-only ON/OFF force prediction with matched controls and guard.

No cached detections, event features or truth are accepted. Existing62 core15
is retained, then68/69/70 add effective2D force and causal guarded forecasts.
"""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'research/stage71')]
import run_axis_forecast as previous
from independent62 import Managed62
from online71 import OnlineExtras
from roi55 import SemanticPublisher
import runtime_full61

def pipeline_source():
    source=previous.pipeline_source()
    substitutions={
        'history=HistoryPredictor()':'history=HistoryPredictor();event_pipeline=OnlineExtras(w,h)',
        '                original_history,_=history.step':
        '                current_activity,current_body_axis,current_activity_audit=event_pipeline.measure(t,e,body)\n                original_history,_=history.step',
        "q['association_diagnostic']=audit":"q['association_diagnostic']=audit;q=event_pipeline.forecast(q,obs,models[method].predictor.axis.quality)",
        '                elapsed=(time.perf_counter()-tick)*1000;':
        "                event_pipeline.commit(t,body,forecasts['delayed_feedback'])\n                elapsed=(time.perf_counter()-tick)*1000;",
        'row=dict(t=t,event_count=len(e),':'row=dict(event_activity=current_activity,body_axis=current_body_axis,event_measurement_audit=current_activity_audit,t=t,event_count=len(e),',
        "('adaptive_axis_only',(190,80,210))":"('force_periodic',(190,80,210))",
        'purple=axis-only':'purple=local-periodic-force',
        'q["feature_diagnostic"]["models"]["adaptive_axis_only"]["active"]':'q["event_force_diagnostic"]["models"]["force_periodic"]["active"]',
        'stats["delayed_feedback"]["actual_updates"]["adaptive_axis_only"]':'q["event_force_diagnostic"]["models"]["force_periodic"]["updates"]',
        "        (output/'protocol.json').write_text(json_text(protocol),encoding='utf-8')":
        "        protocol.update(event_force_stage=71,core15_stage62_unchanged=True,additional_forecasts=11,event_inputs='Current raw20ms + bounded160ms past events + current NN ROI + committed past CV velocity only',event_scope='Experimental effective2D force proxies, not true RPM/pose/wind. Guarded_events can fall back to pure history; that is not evidence of event predictive value.')\n        (output/'protocol.json').write_text(json_text(protocol),encoding='utf-8')"
    }
    for old,new in substitutions.items():
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    return source

def build(capacity_mib=32):
    runtime_full61.verify();path=ROOT/'run_joint_forecast.py'
    namespace=dict(__name__='event_force71',__file__=str(path),event_transport='shared',
        event_capacity_bytes=capacity_mib*1024*1024,Publisher=SemanticPublisher,geometry_roi_mode='off',OnlineExtras=OnlineExtras)
    exec(compile(pipeline_source(),str(path),'exec'),namespace)
    namespace['METHODS']=['delayed_feedback'];namespace['Managed52']=lambda:Managed62('axis_learn')
    old=namespace['fingerprints']
    def fingerprints():
        result=old()
        paths=[Path(__file__),ROOT/'run_axis_forecast.py',ROOT/'run_stable_paired_forecast.py',ROOT/'run_paired_forecast.py',
            ROOT/'research/stage62/independent62.py',ROOT/'research/stage64/compensation64.py',ROOT/'research/stage6/silhouette.py',
            ROOT/'research/stage68/activity68.py',ROOT/'research/stage66/periodicity66.py',
            ROOT/'research/stage69/force69.py',ROOT/'research/stage70/guard70.py',Path(__file__).parent/'research/stage71/online71.py']
        paths.extend(ROOT/'research/stage61'/n for n in ['preline61.py','runtime_full61.py','rotor_worker61.py'])
        for p in paths:result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace['fingerprints']=fingerprints
    return namespace

def run(raw,output,start=0.,duration=0.,preview=False,verify_frames=False):
    namespace=build()
    def receive(worker,kind):
        line=worker.stdout.readline()
        if not line:raise RuntimeError(kind+' worker stopped; inspect log')
        try:return json.loads(line)
        except json.JSONDecodeError:
            Path(output,'ipc_failure.json').write_text(json.dumps(dict(kind=kind,raw_line=line,process_poll=worker.poll()),ensure_ascii=False),encoding='utf-8');raise
    namespace['receive']=receive
    try:return namespace['run'](raw,output,start,duration,32,preview,verify_frames)
    finally:namespace.clear()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.preview,a.verify_frames)
