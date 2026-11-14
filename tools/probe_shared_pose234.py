"""All already-frozen225/226 hypotheses; unchanged author settings, CPU only."""
from pathlib import Path
import os,sys,json,hashlib
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from probe_helix_cpu233 import probe
from pure_visibility170 import window
OUT=ROOT.parent/'output/pose234_20261108'
if __name__=='__main__':
    OUT.mkdir(exist_ok=False);files=['tools/probe_shared_pose234.py','tools/probe_helix_cpu233.py',
        'angular_phase_candidates.py','research/helix_reference/reference_api.py']
    sha={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
    for n in files:
        dst=OUT/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    (OUT/'protocol.json').write_text(json.dumps(dict(source_sha256=sha,GPU_used=False,
        all_prior_225_226_hypotheses_included=True,qualification_thresholds_changed=False,
        validation_updates_filter_or_pose=False,independent_RPM_truth_available=False,
        trajectory_FDE_gain_claimed=False,author_settings_refit=False),indent=2),'utf-8')
    records=[]
    for folder in ['geometry_first225_20261108','geometry_first226_20261108']:
        src=ROOT.parent/f'output/{folder}/results.json';old=json.loads(src.read_text('utf-8'))
        for item in old['records']:
            n=item['recording'];t=item['t'];e=window(ROOT.parent/f'output/{"motion181_20261107" if n=="15" else "phase185_20261107"}/cache_recording{n}',t)
            for arm,a in item['arms'].items():
                velocity=(a.get('motion_proposal') or {}).get('velocity_px_s') if arm=='early_motion' else None
                for i,h in enumerate(a.get('hypotheses',[])):
                    result=probe(e,h['ellipse'],t,velocity)
                    records.append(dict(recording=n,t=t,geometry_arm=arm,hypothesis=i,ellipse=h['ellipse'],
                        velocity=velocity,result=result))
            (OUT/'partial.json').write_text(json.dumps(records,indent=2),'utf-8')
            subset=[r for r in records if r['recording']==n and r['t']==t]
            print(n,t,'constant',sum(v['result']['control']['qualified'] for v in subset),
                'phase_only',sum(v['result'].get('arms',{}).get('phase_only',{}).get('qualified',False) for v in subset),
                'joint_pose',sum(v['result'].get('arms',{}).get('joint_pose',{}).get('qualified',False) for v in subset),flush=True)
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in sha.items())
    assert 'torch' not in sys.modules
    (OUT/'results.json').write_text(json.dumps(dict(records=records,source_changed=False,GPU_used=False,
        accepted_hypotheses_are_not_unique_physical_rotors=True,trajectory_FDE_measured=False),indent=2),'utf-8')
