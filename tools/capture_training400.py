"""Broader RAW-native training, six evaluation recordings excluded entirely.

Predeclared10..40s of every other local experiment recording. Same357 native
pipeline, no NN or per-video parameters. Existing no-index cooperative SDK
reader leaves originalRAW and sidecar directories untouched. Offline capture
is NOT a sustained realtime/FPS certificate. Empty/short recordings are logged,
not replaced by hand-picked favorable intervals.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,traceback,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_rotor_only import make_parser,run,SOURCE_FILES
from conditional_unknown_force import teachers

EVALUATION={'3','4','11','13','15','16'}
TRAINING=('1','2','5','6','7','8','9','10','12','14','17','18','19')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='training400_20261110');parser.add_argument('--start',type=float,default=10.);parser.add_argument('--duration',type=float,default=30.);config=parser.parse_args()
    out=ROOT.parent/'output'/config.batch;out.mkdir(exist_ok=False)
    files=sorted(set(SOURCE_FILES+['tools/capture_training400.py']));hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    raw={n:Path(f'E:/dataset/experiment/{n}/media/event.raw') for n in TRAINING};assert not EVALUATION.intersection(TRAINING)
    protocol=dict(training_source_ids=list(TRAINING),entire_evaluation_source_ids_excluded=sorted(EVALUATION),
        source_sha256=hashes,raws={n:dict(path=str(p),exists=p.exists(),bytes=p.stat().st_size if p.exists() else None) for n,p in raw.items()},
        start_s=config.start,duration_s=config.duration,one_rule_all_sources=True,manual_references_read=False,NN_used=False,
        native_only_original357_pipeline=True,cooperative_reader_build_index=False,
        originals_or_external_sidecars_not_written=True,offline_training_not_realtime_benchmark=True,
        failure_or_short_source_not_replaced_by_selected_interval=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),'utf8')
    (out/'source_snapshot').mkdir()
    for f in files:
        p=out/'source_snapshot'/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())
    records={}
    for n,p in raw.items():
        if not p.exists():records[n]=dict(status='missing_raw',teachers=0);continue
        dest=out/f'recording{n}';started=time.perf_counter();print('TRAIN SOURCE',n,'START fixed',config.start,config.start+config.duration,flush=True)
        args=make_parser().parse_args(['--raw',str(p),'--start',str(config.start),'--duration',str(config.duration),'--output',str(dest),
            '--gpu-python','C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe',
            '--rotor-native-geometry-dll',str(ROOT.parent/'output/uncertainty216_20261108/geometry_kernels.dll')])
        settings=dict(rotor_gpu_front=True,rotor_native_gpu=True,rotor_fast_geometry=True,
            rotor_remote_front=True,rotor_no_phase=True,rotor_local_phase=True,
            rotor_subpixel_motion=True,rotor_entry_init=True,rotor_relative_observation=True,
            rotor_relock_support_guard=True,rotor_border_motion=True,
            rotor_gap_prediction='short',rotor_reference_mode='fixed',rotor_unknown_force_model='plane-adaptive',
            rotor_regional_graph=True,rotor_projected_fov_guard=True,rotor_curvature_hz=0.,
            rotor_prefetch=True,rotor_reader_ready=True,rotor_reader='cooperative',
            rotor_max_backlog_ms=0.,paced=False,no_preview=True,
            rotor_inverse_normal_response=True,rotor_anisotropic_response=True,rotor_periodic_continuation=False)
        for k,v in settings.items():setattr(args,k,v)
        try:
            run(args);rows=[json.loads(s) for s in (dest/'predictions.jsonl').read_text('utf8').splitlines()];samples=teachers(rows,'camera')
            records[n]=dict(status='complete',frames=len(rows),valid=sum(r['valid'] for r in rows),teachers=len(samples),elapsed_s=time.perf_counter()-started)
        except Exception:
            error=traceback.format_exc();(out/f'error_recording{n}.txt').write_text(error,'utf8');records[n]=dict(status='failed_preserved_no_replacement_interval',teachers=0,elapsed_s=time.perf_counter()-started,error=error[-1800:])
        (out/'progress.json').write_text(json.dumps(dict(records=records,total_teachers=sum(r['teachers'] for r in records.values()),current_source=n),indent=2),'utf8');print('TRAIN SOURCE',n,'DONE',records[n],flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items()),'Source files changed during training capture'
    (out/'completion.json').write_text(json.dumps(dict(records=records,total_teachers=sum(r['teachers'] for r in records.values()),manual_references_read=False,evaluation_sources_never_opened=True,RAW_FPS_certificate=False),indent=2),'utf8')
