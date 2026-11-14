"""Precommit this-round holdout before reading its labels; upstream not blind."""
from pathlib import Path
import sys,json,datetime,hashlib,zipfile,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT/'tools'))
from prepare_fred_transfer import parse,sha


def main(sequence=8):
    out=ROOT.parent/f'output/stability_audit_20261031/holdout{sequence}';out.mkdir(exist_ok=False)
    files=['run_fast_structure.py','fast_huber.py','native_huber.py','fast_geometry_compute.py','native_conic_residual.py',
        'curved_flow_motion.py','live_flow_acceleration.py','shared_image_motion.py','physical_publication.py','nn_recovery_budget.py','fast_raw_reader.py','position_net_force.py']
    protocol=dict(committed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sequence=sequence,
        selection='earliest integer4s interval >=20s: >=90 labels through +4.4s; >=80% fully in-FOV min-side16px; max label gap134ms',
        score_horizon_s=.4,origin_hz=5,all_misses_retained=True,
        parameters_chosen_using_this_sequence_errors=False,
        upstream_detector_training_overlap='Upstream README says entire FRED; NOT globally unseen detector evaluation',
        candidates=(['current realtime30 frozen baseline','same realtime30 + position-net-force'] if sequence==10 else ['realtime30 control','same realtime30 + jit-huber + jit-conic-residual'+(' + budgeted-nn-recovery' if sequence==9 else '')]),
        failed_precision_hypotheses_not_promoted=['body-core points','causal residual force','component acceleration','fitted endpoint velocity'],
        source_sha256={name:sha(ROOT/name) for name in files},detector_sha256=sha(ROOT/'models/live_position_accumulated_v1.onnx'))
    (out/'preregister.json').write_text(json.dumps(protocol,indent=2),'utf-8')
    if shutil.disk_usage(out).free<3*1024**3:raise RuntimeError('Need3GiB free')
    archive=Path(f'E:/FRED/{sequence}.zip')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();label=next(n for n in names if Path(n).name=='coordinates.txt')
        raw=next(n for n in names if n.endswith('/Event/events.raw') or n=='Event/events.raw')
        content=z.read(label);labels=parse(content.decode('utf-8-sig'));times=np.array([r['t'] for r in labels])
        if (np.diff(times)<0).any():raise ValueError('Nonmonotonic labels')
        unique=[r for r in labels if sum(abs(times-r['t'])<1e-8)==1]
        for start in range(20,110):
            group=[r for r in unique if start<=r['t']<=start+4.4]
            good=[r for r in group if 0<r['bbox'][0]<r['bbox'][2]<1280 and 0<r['bbox'][1]<r['bbox'][3]<720 and min(np.subtract(r['bbox'][2:],r['bbox'][:2]))>=16]
            if len(group)>=90 and len(good)>=.8*len(group) and max(np.diff([r['t'] for r in group]),default=99.)<=.134:break
        else:raise ValueError('No eligible interval; do not replace with error-based selection')
        dest=out/'input';dest.mkdir();(dest/'coordinates.txt').write_bytes(content)
        with z.open(raw) as source,(dest/'events.raw').open('wb') as target:shutil.copyfileobj(source,target,1024*1024)
    protocol.update(start=start,duration=4.,raw=str(dest/'events.raw'),raw_sha256=sha(dest/'events.raw'),
        annotation_sha256=hashlib.sha256(content).hexdigest(),labels_in_span=len(group),eligible_size_labels=len(good),archive=str(archive))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf-8')
    print(json.dumps({k:v for k,v in protocol.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 8)
