"""Predeclared Stage 1 study. Run freeze BEFORE controlled/raw experiments."""
from pathlib import Path
import argparse,copy,csv,datetime,hashlib,json,os,shutil,sys,time
import numpy as np
from mechanism import *
OUT=ROOT.parent/'output/stage1_observation_errors_20261016'

def save(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def csvsave(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def fingerprints():
    paths=[*(ROOT/'src').rglob('*.py'),*(ROOT/'research/stage1').glob('*.py')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def freeze():
    if (OUT/'protocol.json').exists():raise RuntimeError('Protocol exists; never overwrite a frozen experiment')
    OUT.mkdir(exist_ok=True,parents=True)
    sources=fingerprints()
    for rel in sources:
        dest=OUT/'source_snapshot'/rel;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(ROOT/rel,dest)
    save(OUT/'protocol.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stage='1: diagnose observation errors BEFORE proposing a new estimator',
        primary_seeds=list(range(100,120)),oracle_seeds=[100],worlds=['matched','gain_mismatch'],
        scenarios=list(SCENARIOS),duration_s=14.,dt_s=.02,fault_start_s=5.,fault_end_s=6.,
        methods=list(METHODS),horizons_s=HORIZONS.tolist(),position_noise_px=.75,
        frequency_relative_noise_sigma=.006,ellipse_angle_sigma_deg=.5,ellipse_ratio_sigma=.008,
        model_comparison='Common current CV state and current normal/RPM hold; isolates forcing/learning rather than different position filters.',
        baseline_definitions=dict(fixed='No learning, same default pixel prior',ordinary_rls='Constrained delayed RLS+residual; no quality/excitation gating',
            quality_rls='Same equations as ordinary, only low-quality/fewer-than-four admission disabled',
            v030_rls='Frozen original ProjectedThrustModel, including heuristic excitation gate',
            ordinary_reset_oracle='Diagnostic only: transplant clean learner state when fault ends; unavailable in deployment',
            oracle_current_state='Diagnostic only: supply current truth p/v; all future states withheld'),
        evaluation_windows=dict(pre=[2,4.6],during=[5,6],post_early=[6.4,8],post_late=[9,13.6]),
        recovery_definition='Earliest >=6s when forward 0.5s mean paired 0.4s excess error <=1px for a full following 1s; offline metric, right-censored if not recovered.',
        recovery_threshold_px=1.,recovery_hold_s=1.,recovery_average_s=.5,
        inference_policy='Current/past observations only. Synthetic truth and paired clean model are used only for scoring and explicitly labelled oracle diagnostics.',
        inference_of_persistence='Inspect seed-paired excess 0.4s post-early error (practical threshold 1px), seed-bootstrap 95% CI, normalized parameter/residual drift, and oracle-reset effect. Do not treat correlated frames as independent trials.',
        negative_controls=['ID-only permutation must preserve total thrust and all predictions; no body-moment correspondence claim.',
                           'Before fault start all paired trials must be identical.',
                           'Observation stream must restore exactly after fault; otherwise no claim of persistent learning contamination.'],
        raw_experiment=dict(raw=r'E:\dataset\train\6\media\event.raw',warmup_end_s=24.,fault=[24,25],stop_s=30.,
            conditions=['clean','one_rotor_rectangle','left_pair_rectangle','drop_half'],
            deletion_rectangles='Fixed sensor-coordinate rectangles from past observations at24s. No future box/labels used. Drop-half seeded by batch timestamp.',
            repeats=1,interpretation='One preselected development clip; deleted events omit occluder events; not real physical occlusion or new holdout.'),
        sources=sources,does_not_modify_v030_runtime=True))
    print('Protocol frozen before experiments:',OUT,flush=True)

def verify():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    current=fingerprints()
    for rel,digest in protocol['sources'].items():
        if current.get(rel)!=digest:raise RuntimeError('Frozen experiment code changed: '+rel)
    return protocol

def controlled():
    p=verify();dest=OUT/'controlled';dest.mkdir(exist_ok=True);started=time.perf_counter();completed=0
    # No tuning between worlds or scenarios. Identical seeds couple all arms.
    for mode,seeds in [('estimated',p['primary_seeds']),('oracle_current_state',p['oracle_seeds'])]:
        for world_name in p['worlds']:
            for seed in seeds:
                world=make_world(seed,world_name,p['duration_s'],p['dt_s'])
                directory=dest/f'{mode}/{world_name}/seed_{seed}';directory.mkdir(exist_ok=True,parents=True)
                clean,reset=run_trial(world,'clean',mode,retain_reset=True)
                np.savez_compressed(directory/'truth.npz',times=world['times'],position=world['position'],velocity=world['velocity'],
                                    acceleration=world['acceleration'],true_parameters=world['true_parameters'])
                for scenario in p['scenarios']:
                    result=clean if scenario=='clean' else run_trial(world,scenario,mode,clean_reset=reset)[0]
                    # Prove no pre-fault effects. Oracle reset must also equal
                    # ordinary in a genuinely clean trial, including after6s.
                    prefix=world['times']<p['fault_start_s']
                    np.testing.assert_array_equal(result['prediction'][prefix],clean['prediction'][prefix])
                    np.testing.assert_array_equal(result['parameters'][prefix],clean['parameters'][prefix])
                    if scenario=='id_permutation_only':np.testing.assert_array_equal(result['prediction'],clean['prediction'])
                    if scenario=='clean':np.testing.assert_array_equal(result['prediction'][:,2],result['prediction'][:,5])
                    np.savez_compressed(directory/f'{scenario}.npz',**result)
                    completed+=1
                print(json.dumps(dict(mode=mode,world=world_name,seed=seed,trials=completed,wall_s=round(time.perf_counter()-started,1))),flush=True)
    save(dest/'completion.json',dict(trials=completed,wall_s=time.perf_counter()-started,protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()))

def audit():
    OUT.mkdir(exist_ok=True,parents=True);root=Path(r'E:\dataset')
    suffixes={'.csv','.ulg','.ulog','.bag','.mcap','.mat','.json','.yaml','.yml','.h5','.hdf5','.txt','.npz'}
    skip={'software','tools','.git','node_modules','__pycache__','.venv','site-packages'}
    counts={};candidates=[];json_hits=[]
    for folder,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d.lower() not in skip]
        for name in files:
            path=Path(folder)/name;ext=path.suffix.lower();counts[ext]=counts.get(ext,0)+1
            if ext not in suffixes:continue
            low=name.lower()
            relevant=any(s in low for s in ('rpm','telemetry','imu','ground_truth','groundtruth','tach','attitude','pose','flight_log','sensor_log')) or ext in ('.ulg','.ulog','.bag','.mcap')
            if relevant:candidates.append(dict(path=str(path),bytes=path.stat().st_size))
            if ext=='.json' and path.stat().st_size<100000 and any(s in low for s in ('manifest','sync','calib','sensor','camera')):
                try:
                    text=path.read_text(encoding='utf-8-sig')
                    keys=[s for s in ('rpm','imu','tachometer','ground_truth','physical_sync_verified','camera_matrix','intrinsics','telemetry') if s in text.lower()]
                    if keys:json_hits.append(dict(path=str(path),matched_tokens=keys))
                except (OSError,UnicodeError):pass
    save(OUT/'existing_ground_truth_audit.json',dict(root=str(root),excluded_directories=sorted(skip),file_extension_counts=counts,
        filename_candidates=candidates,small_metadata_token_hits=json_hits,
        caveat='Filename/token inventory only, not proof that files contain independent synchronized usable truth. Public EEPPR is periodic-target truth, not flight pose truth.'))
    print(json.dumps(dict(files=sum(counts.values()),filename_candidates=len(candidates),metadata_hits=len(json_hits))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','controlled','audit']);a=parser.parse_args();globals()[a.action]()
