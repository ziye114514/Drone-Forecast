"""Build a small relocatable live source bundle, using an explicit allowlist.

No SDK imports, RAW/weights reads, subprocesses, package installs or timing.
This prepares source; it does not verify inference or claim a working wheel.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ROOT_FILES=('forecast.py','forecast_runtime.py','forecast_tracks.py','forecast_mainline.py','configs/live_tracks.json','run_live_forecast.py','live_stream.py',
    'live_decode.py','live_process.py','live_motion.py','live_measurement_kf.py',
    'live_covariance.py','live_publication.py','live_frames.py','live_startup.py','live_scale_rollout.py','live_local_flow.py','live_flow_spatial.py','live_flow_acceleration.py','live_pixel_axis.py','live_periodic_packet.py','live_roi_refine.py','live_center_ensemble.py','body_association.py','live_worker.py',
    'LICENSE','THIRD_PARTY_NOTICES.md','docs/LIVE_RUNTIME_SOURCE.md')
RESEARCH_FILES=('research/stage141/native_axis141.py',
    'research/stage79/native79.py')
PACKAGE_FILES=('__init__.py','raw_io.py','geometry.py','detector_backend.py')


def sha(data):return hashlib.sha256(data).hexdigest()


def build(out):
    out=out.resolve()
    if out.exists():raise FileExistsError('Choose a new bundle directory: '+str(out))
    archive=out.with_suffix('.zip')
    if archive.exists():raise FileExistsError(archive)
    files=list(ROOT_FILES)+list(RESEARCH_FILES)
    base=ROOT/'src/event_drone_forecast'
    files+=['src/event_drone_forecast/'+p for p in PACKAGE_FILES]
    files+=[p.relative_to(ROOT).as_posix() for p in sorted((base/'position').glob('*.py'))]
    files+=[p.relative_to(ROOT).as_posix() for p in sorted((base/'_vendor').rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    assert len(files)==len(set(files))
    snapshots={name:(ROOT/name).read_bytes() for name in files}
    for name,data in snapshots.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    out.mkdir(parents=True,exist_ok=False)
    for name,data in snapshots.items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    (out/'live.py').write_text('''"""Portable live-only launcher; config is explicit, no private defaults."""
from forecast import main
if __name__ == '__main__':
    raise SystemExit(main(allowed_profiles=('live',),default_profile='live',require_config=True))
''',encoding='utf-8')
    cfg=json.loads((ROOT/'configs/runtime.example.json').read_text('utf-8'))
    cfg.update(profile='live',tracker_hz=25,publish_hz=5,producer='process',frame_transport='events',publication_clock='independent',body_association='none',noise_model='legacy',startup_sync='none')
    (out/'configs').mkdir()
    (out/'configs/live.example.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (out/'README.md').write_text('''# Portable live RAW source candidate

This is the current position-only paced RAW replay, not the full historical
research/rotor pipeline, a wheel, or a live-camera driver. Default motion and
qualification parameters are unchanged. Detector backend extraction and SDK
ownership cleanup await RAW/GPU regression after the user resumes workloads.

Copy `configs/live.example.json` to your own config and set SDK Python, GPU
Python, compatible ONNX weights + SHA256 and a writable cache directory.
Relative paths resolve from the config directory. No personal paths, RAW,
weights, proprietary SDK, CUDA runtime, Ultralytics or HelixTrack are bundled.
Both Python environments and these third-party dependencies must already be
installed. See THIRD_PARTY_NOTICES.md; Ultralytics and SDK terms are separate.

```text
python live.py --help
python live.py run --config configs/local.json --raw <event.raw> --output <new-output-dir> --start 110 --duration 2 --dry-run
```

Remove `--dry-run` only when workload execution is authorized. `doctor` imports
SDK/GPU dependencies. At build time, no relocated RAW execution is claimed;
later smoke records, if any, are separate from this immutable source manifest.
Default: 25Hz tracking target, 5Hz publication target; invalid heartbeats are
explicit. These targets are not achieved-rate or accuracy guarantees.

`--frame-transport frames` remains optional experimental, process-only. All
native SDK updates are retained; current frames can use shared memory. The
default is events. Optional `--startup-sync reader-ready` holds the prepared
reader until detector warmup and output initialization finish; default none
keeps the original sequence. `--periodic-axis` is observation-only, never a force input.
Source hashes are recorded in protocol.json. This source bundle is separate
from the older full source ZIP and the previously verified 0.3.1 position wheel.
''',encoding='utf-8')
    generated=['live.py','configs/live.example.json','README.md']
    manifest=dict(scope='live-only relocatable source candidate; no new end-to-end verification',
        source_files=[dict(path=n,sha256=sha(data),bytes=len(data)) for n,data in snapshots.items()],
        generated_files=[dict(path=n,sha256=sha((out/n).read_bytes())) for n in generated],
        build_does_not_start_workloads=True,RAW_or_GPU_started=False,weights_or_SDK_included=False,
        wheel_built=False,full_RAW_execution_verified=False,
        research_scripts_included=list(RESEARCH_FILES),
        full_historical_research_pipeline_included=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(sha(z.read(n))==sha((out/n).read_bytes()) for n in z.namelist())
        count=len(z.namelist())
    report=dict(directory=str(out),archive=str(archive),sha256=sha(archive.read_bytes()),
        archive_bytes=archive.stat().st_size,files=count,full_RAW_execution_verified=False,
        GPU_or_RAW_started=False,timing_measured=False)
    out.with_name(out.name+'_delivery.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    build(p.parse_args().output)


if __name__=='__main__':main()
