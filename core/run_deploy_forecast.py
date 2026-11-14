"""Complete physical model deployment: one final branch, optional one RAW decoder."""
from pathlib import Path
import argparse,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'research/stage54'))
from packets54 import Publisher

def pipeline_source(transport):
    assert transport in ['dual','shared'];source=(ROOT/'core/run_joint_forecast.py').read_text(encoding='utf-8')
    common={
        'source_hashes=hashes,rotor_worker=':'source_hashes=hashes,transport=event_transport,event_capacity_bytes=event_capacity_bytes if event_transport=="shared" else 0,research_control_groups_removed=True,rotor_worker=',
        'Old reset and stable controls saved alongside delayed feedback':'Final delayed-feedback branch retains all15 models; duplicate research controls removed',
        'noise.process_events(e,buffer);clean=buffer.numpy().copy()':'geometry_ms=(time.perf_counter()-tick)*1000;prepare_started=time.perf_counter();noise.process_events(e,buffer);clean=buffer.numpy().copy()',
        '                send(rotor,dict(id=n,cutoff=t));':'                preparation_ms=(time.perf_counter()-prepare_started)*1000;communication_started=time.perf_counter()\n                send(rotor,dict(id=n,cutoff=t));',
        "                if result['boxes']:":"                communication_ms=(time.perf_counter()-communication_started)*1000;physics_started=time.perf_counter()\n                if result['boxes']:",
        '                elapsed=(time.perf_counter()-tick)*1000;':'                physics_ms=(time.perf_counter()-physics_started)*1000\n                elapsed=(time.perf_counter()-tick)*1000;',
        "body_boxes=result['boxes'],frame_sha256=frame_sha,":"body_boxes=result['boxes'],frame_sha256=frame_sha,event_packet_sha256=packet_header['payload_sha256'] if event_transport=='shared' else None,event_packet_nbytes=packet_header['nbytes'] if event_transport=='shared' else 0,event_packet_publish_ms=packet_header['publish_ms'] if event_transport=='shared' else 0,parent_geometry_ms=geometry_ms,image_preparation_ms=preparation_ms,communication_ms=communication_ms,physics_ms=physics_ms,",
        'history_all_six_exact=True)':'history_all_six_exact=True,maximum_event_packet_bytes=event_packet.max_bytes if event_transport=="shared" else 0)'}
    for old,new in common.items():assert source.count(old)==1,old;source=source.replace(old,new)
    if transport=='shared':
        changes={
            'memory=None;frame=None;video=None;':'event_packet=None;memory=None;frame=None;video=None;',
            "        rlog=(output/'rotor_worker.log').open":"        event_packet=Publisher(event_capacity_bytes)\n        rlog=(output/'rotor_worker.log').open",
            "[sys.executable,str(ROOT/'research/stage53/rotor_worker53.py'),'--raw',str(raw),'--output',str(output),'--start',str(start),'--duration',str(duration)]":"[sys.executable,str(ROOT/'research/stage54/rotor_worker54.py'),'--shared-events',event_packet.memory.name,'--output',str(output),'--start',str(start),'--width',str(w),'--height',str(h)]",
            'send(rotor,dict(id=n,cutoff=t));send(gpu,':'packet_header=event_packet.publish(e,n,t,verify_frames);send(rotor,dict(id=n,cutoff=t,packet=packet_header));send(gpu,',
            "                assert not verify_frames or result['frame_sha256']==frame_sha":"                event_packet.acknowledge(n);assert not verify_frames or rotor_result['event_packet_sha256']==packet_header['payload_sha256']\n                assert not verify_frames or result['frame_sha256']==frame_sha",
            'Two independent bounded RAW readers: body20 and rotor10':'One bounded RAW20 reader; exact native packets split into RAW10 in an isolated worker',
            'duplicate RAW decoding and video':'one RAW decoding and shared event publication and video',
            '        stop(gpu);stop(rotor)':'        stop(gpu);stop(rotor)\n        if event_packet is not None:event_packet.close()'}
        for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    return source

def build(transport,capacity_mib):
    path=ROOT/'core/run_joint_forecast.py';namespace={'__name__':'deployment54','__file__':str(path),'event_transport':transport,'event_capacity_bytes':capacity_mib*1024*1024,'Publisher':Publisher}
    exec(compile(pipeline_source(transport),str(path),'exec'),namespace);namespace['METHODS']=['delayed_feedback'];original=namespace['fingerprints']
    def fingerprints():
        result=original()
        for p in [Path(__file__),ROOT/'research/stage54/packets54.py',ROOT/'research/stage54/rotor_worker54.py']:result[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return result
    namespace['fingerprints']=fingerprints;return namespace
def run(raw,output,start=0.,duration=0.,precision=32,preview=False,verify_frames=False,transport='shared',event_capacity_mib=32):
    assert transport in ['dual','shared'] and 1<=event_capacity_mib<=256;namespace=build(transport,event_capacity_mib)
    try:return namespace['run'](raw,output,start,duration,precision,preview,verify_frames)
    finally:namespace.clear()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=0.);p.add_argument('--duration',type=float,default=0.);p.add_argument('--precision',type=int,choices=[16,32],default=32);p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true');p.add_argument('--transport',choices=['dual','shared'],default='shared');p.add_argument('--event-capacity-mib',type=int,default=32);a=p.parse_args();run(a.raw,a.output,a.start,a.duration,a.precision,a.preview,a.verify_frames,a.transport,a.event_capacity_mib)
