"""Published CPU core: discovery-only joint fit, frozen heldout scoring.

Not the paper protocol, not a complete tracker, not an RPM/FDE benchmark.
No author ground-truth initializer; geometry/sign/frequency from early events.
"""
from pathlib import Path
import os,sys,json,hashlib,numpy as np
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'research/helix_reference')]
from reference_api import AuthorTracker
from angular_phase_candidates import ring_phase_candidates,ring_phase_samples
from pure_visibility170 import window
from probe_chirp231 import fixture,ELLIPSE
from ellipse_visibility import ellipse_fov_clipped
OUT=ROOT.parent/'output/helix_cpu233_20261108'
DLL=ROOT.parent/'output/helix_cpu232_20261108/build/Release/helix_reference.dll'

from joint_ring_phase import observe_ring,projected_geometry

def probe(e,r,t,velocity=None):
    end=(round(t*1e6)-33333+33333//2)/1e6
    return observe_ring(e,r,t,velocity,author_dll=DLL,geometry_source_end_s=end,
        velocity_source_end_s=end if velocity is not None else None,compare_phase_only=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=False);names=['angular_phase_candidates.py','tools/probe_helix_cpu233.py','research/helix_reference/reference_api.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    for n in names:
        dst=OUT/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    (OUT/'protocol.json').write_text(json.dumps(dict(source_sha256=hashes,DLL_sha256=hashlib.sha256(DLL.read_bytes()).hexdigest(),
        GPU_used=False,author_core_pose_refinement=True,author_headers_modified=False,author_GT_initializer_used=False,
        validation_events_update_filter=False,initial_geometry_from='frozen225 early-half ONOFF geometry',
        stage='CPU short-window observation probe, not full paper reproduction or forecasting result'),indent=2),'utf-8')
    smoke=probe(fixture(2,0.),ELLIPSE,.033333)
    assert smoke['arms']['phase_only']['qualified'],smoke
    records=[];old=json.loads((ROOT.parent/'output/geometry_first225_20261108/results.json').read_text('utf-8'))
    for item in old['records']:
        n=item['recording'];t=item['t']
        if (n,t) not in [('12',12.),('15',13.5)]:continue
        e=window(ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}',t)
        a=item['arms']['early_motion'];velocity=a['motion_proposal']['velocity_px_s']
        for i,h in enumerate(a['hypotheses']):
            result=probe(e,h['ellipse'],t,velocity);records.append(dict(recording=n,t=t,hypothesis=i,ellipse=h['ellipse'],result=result))
            (OUT/'partial.json').write_text(json.dumps(records,indent=2),'utf-8')
            print(n,t,i,{k:v['qualified'] for k,v in result['arms'].items()},flush=True)
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    assert 'torch' not in sys.modules
    (OUT/'results.json').write_text(json.dumps(dict(smoke=smoke,records=records,source_changed=False,GPU_used=False,
        trajectory_prediction_gain_measured=False),indent=2),'utf-8')
