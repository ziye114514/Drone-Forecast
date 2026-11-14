"""Refactor parity and provenance guard, no GPU and no forecasting claims."""
from pathlib import Path
import os,sys,json,numpy as np
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from joint_ring_phase import observe_ring
from pure_visibility170 import window
from probe_helix_cpu233 import DLL
OUT=ROOT.parent/'output/joint235_20261108'
if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    old=json.loads((ROOT.parent/'output/pose234_20261108/results.json').read_text('utf-8'))
    row=next(r for r in old['records'] if r['recording']=='12' and r['t']==12. and r['geometry_arm']=='early_motion' and r['hypothesis']==0)
    t=row['t'];e=window(ROOT.parent/'output/phase185_20261107/cache_recording12',t);mid=(round(t*1e6)-33333+33333//2)/1e6
    r=observe_ring(e,row['ellipse'],t,row['velocity'],author_dll=DLL,geometry_source_end_s=mid,velocity_source_end_s=mid,compare_phase_only=True)
    assert r['arms']==row['result']['arms'] and r['control']==row['result']['control']
    for geometry_end,velocity_end in [(t,mid),(mid,t)]:
        try:observe_ring(e,row['ellipse'],t,row['velocity'],author_dll='missing_not_loaded.dll',geometry_source_end_s=geometry_end,velocity_source_end_s=velocity_end)
        except ValueError:pass
        else:raise AssertionError('Initialization containing validation information accepted')
    assert r['force_input_enabled'] is False and 'torch' not in sys.modules
    (OUT/'checks.json').write_text(json.dumps(dict(fixed_observation_parity=True,
        geometry_and_velocity_timestamp_guards=True,force_feedback_disabled=True,GPU_used=False),indent=2),'utf-8')
    print('Reusable CPU observation parity and provenance guards passed',flush=True)
