"""RAW-only full physical forecast with causal past-body-guided rotor geometry."""
from pathlib import Path
import argparse,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'research/stage55'),str(ROOT)]
import run_deploy_forecast as previous
from roi55 import SemanticPublisher

def pipeline_source():
    source=previous.pipeline_source('shared');changes={
        'last_print=began;summary=None':'last_print=began;summary=None;past_body_hint=None',
        "str(ROOT/'research/stage54/rotor_worker54.py')":"str(ROOT/'research/stage55/rotor_worker55.py')",
        "'--height',str(h)]":"'--height',str(h),'--geometry-roi',geometry_roi_mode]",
        'send(rotor,dict(id=n,cutoff=t,packet=packet_header))':'sent_body_hint=copy.deepcopy(past_body_hint);send(rotor,dict(id=n,cutoff=t,packet=packet_header,body_hint=sent_body_hint,geometry_roi=geometry_roi_mode))',
        "                position=body if body['detected'] else geometry;":"                if body['detected']:past_body_hint=copy.deepcopy(body)\n                position=body if body['detected'] else geometry;",
        'research_control_groups_removed=True,':'research_control_groups_removed=True,geometry_roi=geometry_roi_mode,body_hint_timing="Only prior NN result by packet start; no future20ms hint for earlier10ms geometry",',
        "body_boxes=result['boxes'],frame_sha256=frame_sha,":"body_boxes=result['boxes'],frame_sha256=frame_sha,body_hint_sent=sent_body_hint,roi_diagnostic=rotor_result['record']['roi_diagnostic'],event_fields_sha256=packet_header['named_fields_sha256'],event_fields_hash_ms=packet_header['named_fields_hash_ms'],"}
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    return source
def build(mode,capacity_mib):
    assert mode in ['off','body'];path=ROOT/'core/run_joint_forecast.py';namespace={'__name__':'body_roi55','__file__':str(path),'event_transport':'shared','event_capacity_bytes':capacity_mib*1024*1024,'Publisher':SemanticPublisher,'geometry_roi_mode':mode}
    exec(compile(pipeline_source(),str(path),'exec'),namespace);namespace['METHODS']=['delayed_feedback'];original=namespace['fingerprints']
    def fingerprints():
        result=original()
        for p in [Path(__file__),ROOT/'core/run_deploy_forecast.py',ROOT/'research/stage55/roi55.py',ROOT/'research/stage55/rotor_worker55.py',ROOT/'research/stage54/packets54.py',ROOT/'research/stage54/rotor_worker54.py',ROOT/'research/stage17/frontend.py']:result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace['fingerprints']=fingerprints;return namespace
def run(raw,output,start=0.,duration=0.,precision=32,preview=False,verify_frames=False,geometry_roi='body',event_capacity_mib=32):
    assert 1<=event_capacity_mib<=256;namespace=build(geometry_roi,event_capacity_mib)
    try:return namespace['run'](raw,output,start,duration,precision,preview,verify_frames)
    finally:namespace.clear()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.);p.add_argument('--precision',type=int,choices=[16,32],default=32);p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true');p.add_argument('--geometry-roi',choices=['off','body'],default='body');p.add_argument('--event-capacity-mib',type=int,default=32);a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.precision,a.preview,a.verify_frames,a.geometry_roi,a.event_capacity_mib)
