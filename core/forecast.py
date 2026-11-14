"""Source-checkout CLI; help and doctor do not require the SDK to be installed."""
from pathlib import Path
import argparse
import json
import math
import subprocess
import sys
from forecast_runtime import ROOT, executable, resolve
from forecast_tracks import LIVE_PROFILES,apply_track

PROFILES = {
    'live': [],
    'live-baseline': [],
    'live-accumulated': [],
    'baseline': [],
    'research': ['--pose-solver', 'lm', '--parallel-geometry',
                 '--skip-unobserved-geometry', '--curved-rollout'],
    'rotor-experimental': ['--pose-solver', 'lm', '--parallel-geometry',
                           '--skip-unobserved-geometry', '--curved-rollout', '--rotor-inertia'],
}
CONFIG_KEYS = {'sdk_python', 'gpu_python', 'detector_weights', 'detector_sha256',
               'cache_dir', 'profile', 'preview_window_ms', 'tracker_hz', 'publish_hz', 'producer','body_association','noise_model','publication_clock','frame_transport','startup_sync','scale_rollout','reader_index','periodic_sampling','periodic_packet','center_ensemble_weights','center_ensemble_sha256','local_flow','flow_preparation','flow_acceleration'}


def configuration(path,overrides=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError('Configuration not found: ' + str(path))
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict): raise ValueError('Configuration must be an object')
    overrides=overrides or {}
    runtime_keys={'sdk_python','gpu_python','detector_weights','detector_sha256','cache_dir'}
    if set(overrides)-runtime_keys:raise ValueError('Unknown runtime override')
    for key,value in overrides.items():
        if not isinstance(value,str):raise ValueError(key+' override must be a string')
        if key in ('detector_weights','cache_dir') or (key in ('sdk_python','gpu_python') and ('/' in value or '\\' in value)):
            value=str(Path(value).expanduser().resolve())
        cfg[key]=value
    if set(cfg) - CONFIG_KEYS:
        raise ValueError('Unknown configuration fields: ' + str(set(cfg) - CONFIG_KEYS))
    for key, value in cfg.items():
        if key not in ('preview_window_ms', 'tracker_hz', 'publish_hz') and not isinstance(value, str):
            raise ValueError(key + ' must be a string')
    for key in ('sdk_python', 'gpu_python'):
        if key in cfg:
            value = cfg[key]
            if '/' in value or '\\' in value:
                value = str((path.parent / value).resolve())
            cfg[key] = str(executable(value))
    for key in ('detector_weights', 'cache_dir','center_ensemble_weights'):
        if key in cfg:
            cfg[key] = str((path.parent / Path(cfg[key]).expanduser()).resolve())
    if 'profile' in cfg and cfg['profile'] not in PROFILES:
        raise ValueError('Unknown profile: ' + cfg['profile'])
    return cfg


def probe(python, script):
    result = subprocess.run([str(python), '-X', 'utf8', '-c', script],
                            capture_output=True, text=True, encoding='utf-8',
                            timeout=60, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    lines = result.stdout.splitlines()
    data = None
    for line in reversed(lines):
        try:
            candidate = json.loads(line)
            if isinstance(candidate, dict): data = candidate; break
        except json.JSONDecodeError: pass
    if result.returncode or data is None:
        raise ValueError('Runtime probe failed for ' + str(python) + ':\n' + (result.stderr or result.stdout)[-4000:])
    return data


def doctor(cfg,recipe_id=None):
    sdk = executable(cfg.get('sdk_python', sys.executable))
    runtime = resolve(**{k: cfg[k] for k in ('gpu_python', 'detector_weights', 'detector_sha256', 'cache_dir') if k in cfg})
    sdk_probe = probe(sdk, "import json,sys,importlib.metadata as m; import numpy,scipy,cv2,metavision_core,metavision_sdk_core,metavision_sdk_cv; print(json.dumps(dict(python=sys.version.split()[0],numpy=numpy.__version__,scipy=scipy.__version__,opencv=cv2.__version__,metavision_core=str(metavision_core.__path__))))")
    gpu_probe = probe(runtime['gpu_python'], "import json,sys; import torch,onnxruntime as ort,ultralytics; print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,cuda_available=torch.cuda.is_available(),onnxruntime=ort.__version__,providers=ort.get_available_providers(),ultralytics=ultralytics.__version__)))")
    ready = gpu_probe['cuda_available'] and 'CUDAExecutionProvider' in gpu_probe['providers']
    report = dict(ready=ready, sdk=sdk_probe, gpu=gpu_probe, runtime=runtime,recipe_id=recipe_id,
                  limits='Import/provider/weight checks only; not an inference, SDK ABI or realtime guarantee.')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ready else 1


def command(cfg, args):
    runtime = resolve(**{k: cfg[k] for k in ('gpu_python', 'detector_weights', 'detector_sha256', 'cache_dir') if k in cfg})
    sdk = executable(cfg.get('sdk_python', sys.executable))
    profile = args.profile or cfg.get('profile', 'baseline')
    if profile not in PROFILES: raise ValueError('Unknown profile: ' + profile)
    preview_ms = cfg.get('preview_window_ms', 5.)
    if isinstance(preview_ms, bool) or not isinstance(preview_ms, (int, float)) or not math.isfinite(preview_ms) or not 0 < preview_ms <= 20:
        raise ValueError('preview_window_ms must be in (0, 20]')
    if not math.isfinite(args.start) or not math.isfinite(args.duration) or args.start < 0 or args.duration <= 0 or any(abs(v * 50 - round(v * 50)) > 1e-7 for v in (args.start, args.duration)):
        raise ValueError('Start/duration must align to 20 ms; start >= 0 and duration > 0')
    raw, out = Path(args.raw).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if not raw.is_file() or raw.suffix.lower() != '.raw': raise ValueError('RAW file not found: ' + str(raw))
    if out.exists(): raise ValueError('Output directory already exists: ' + str(out))
    cmd = [str(sdk), '-X', 'utf8', str(ROOT / ('run_live_forecast.py' if profile in LIVE_PROFILES else 'run_optimized_forecast.py')),
           '--raw', str(raw), '--output', str(out), '--start', str(args.start),
           '--duration', str(args.duration), '--preview-window-ms', str(preview_ms)]
    for key, value in runtime.items(): cmd.extend(['--' + key.replace('_', '-'), str(value)])
    cmd.extend(PROFILES[profile])
    if profile in LIVE_PROFILES:
        if getattr(args,'live_recipe_id',None):cmd.extend(['--recipe-id',args.live_recipe_id])
        for key,choices,default in [('body_association',('none','bytetrack'),'none'),('noise_model',('legacy','continuous'),'legacy'),('publication_clock',('tracking','independent'),'independent'),('frame_transport',('events','frames'),'events'),('startup_sync',('none','reader-ready'),'none'),('scale_rollout',('none','compatible'),'none'),('reader_index',('default','disabled'),'default'),('periodic_sampling',('full','fixed-quarter'),'full'),('periodic_packet',('full','latest-pixel'),'full')]:
            value=getattr(args,key,None) or cfg.get(key,default)
            if value not in choices:raise ValueError('Unknown '+key)
            cmd.extend(['--'+key.replace('_','-'),value])
        local_flow=args.local_flow or cfg.get('local_flow','none')
        if local_flow not in ('none','klt'):raise ValueError('Unknown local flow mode')
        cmd.extend(['--local-flow',local_flow])
        preparation=args.flow_preparation or cfg.get('flow_preparation','serial')
        if args.local_flow=='none' and args.flow_preparation is None:preparation='serial'
        if preparation not in ('serial','gpu-overlap') or (preparation!='serial' and local_flow=='none'):
            raise ValueError('GPU-overlap preparation requires a local flow branch')
        cmd.extend(['--flow-preparation',preparation])
        acceleration=args.flow_acceleration or cfg.get('flow_acceleration','none')
        if args.local_flow=='none' and args.flow_acceleration is None:acceleration='none'
        if acceleration not in ('none','velocity-only','position-consistent') or (acceleration!='none' and local_flow=='none'):
            raise ValueError('Flow acceleration requires local flow')
        cmd.extend(['--flow-acceleration',acceleration])
        producer=args.producer or cfg.get('producer','process')
        if producer not in ('process','thread'):raise ValueError('producer must be process or thread')
        transport=args.frame_transport or cfg.get('frame_transport','events')
        if producer!='process' and transport=='frames':
            raise ValueError('Shared frame transport requires --producer process')
        startup=args.startup_sync or cfg.get('startup_sync','none')
        if producer!='process' and startup!='none':
            raise ValueError('Reader-ready startup synchronization requires --producer process')
        if producer!='process' and (args.reader_index or cfg.get('reader_index','default'))!='default':
            raise ValueError('Explicit RAW index configuration requires --producer process')
        if (args.periodic_sampling or cfg.get('periodic_sampling','full'))!='full' and not args.periodic_axis:
            raise ValueError('Fixed pixel sampling requires --periodic-axis')
        if (args.periodic_packet or cfg.get('periodic_packet','full'))!='full' and (not args.periodic_axis or producer!='process'):
            raise ValueError('Periodic packet compression requires --periodic-axis and --producer process')
        ensemble_weight=args.center_ensemble_weights or cfg.get('center_ensemble_weights')
        ensemble_sha=args.center_ensemble_sha256 or cfg.get('center_ensemble_sha256')
        if bool(ensemble_weight)!=bool(ensemble_sha):raise ValueError('Secondary model path and SHA256 are both required')
        if ensemble_weight:
            checked=resolve(runtime['gpu_python'],ensemble_weight,ensemble_sha,runtime['cache_dir'])
            cmd.extend(['--center-ensemble-weights',checked['detector_weights'],'--center-ensemble-sha256',checked['detector_sha256']])
        cmd.extend(['--producer',producer])
        rates={}
        for key, default in [('tracker_hz',25.),('publish_hz',5.)]:
            value = getattr(args,key,None)
            if value is None: value=cfg.get(key,default)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError(key+' must be a positive finite number')
            rates[key]=value
            cmd.extend(['--'+key.replace('_','-'),str(value)])
        if not 0 < rates['publish_hz'] <= rates['tracker_hz'] <= 50:
            raise ValueError('Require 0 < publish Hz <= tracker Hz <= 50')
    elif any(getattr(args,key) is not None for key in ('tracker_hz','publish_hz','producer','body_association','noise_model','publication_clock','frame_transport','startup_sync','scale_rollout','reader_index','periodic_sampling','periodic_packet','center_ensemble_weights','center_ensemble_sha256','local_flow','flow_preparation','flow_acceleration')):
        raise ValueError('Rate options require a live profile')
    if args.periodic_axis: cmd.append('--periodic-axis')
    if args.preview: cmd.append('--preview')
    if args.verify_frames: cmd.append('--verify-frames')
    return cmd


def main(argv=None, *, allowed_profiles=None, default_profile='baseline', require_config=False):
    profiles=tuple(PROFILES) if allowed_profiles is None else tuple(allowed_profiles)
    if not profiles or any(p not in PROFILES for p in profiles) or default_profile not in profiles:
        raise ValueError('Invalid launcher profile configuration')
    parser = argparse.ArgumentParser(description='Event drone RAW forecasting launcher')
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('doctor', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path, required=require_config,
                       default=None if require_config else ROOT / 'configs/local.json',
                       help='JSON config; relative paths resolve from the config directory')
        p.add_argument('--sdk-python',help='override SDK Python from config')
        p.add_argument('--gpu-python',help='override detector Python from config')
        p.add_argument('--detector-weights',help='override ONNX model; relative CLI paths use current directory')
        p.add_argument('--detector-sha256',help='required expected hash for a different detector')
        p.add_argument('--cache-dir',help='override writable cache directory')
        p.add_argument('--profile', choices=profiles)
        p.add_argument('--live-revision',help='choose a recorded accumulated recipe; requires live-accumulated profile')
        if name == 'run':
            p.add_argument('--raw', required=True)
            p.add_argument('--output', required=True)
            p.add_argument('--start', type=float, default=0.)
            p.add_argument('--duration', type=float, default=2.)
            p.add_argument('--tracker-hz',type=float,help='live profile only; default25')
            p.add_argument('--publish-hz',type=float,help='live profile only; default5')
            p.add_argument('--producer',choices=['process','thread'],help='live profile only; default process')
            p.add_argument('--frame-transport',choices=['events','frames'],help='live process only; shared native frames are experimental')
            p.add_argument('--startup-sync',choices=['none','reader-ready'],help='live process only; optional warmup after RAW reader readiness, before replay epoch')
            p.add_argument('--scale-rollout',choices=['none','compatible'],help='live only; optional compatible-width/height scale correction, default none')
            p.add_argument('--reader-index',choices=['default','disabled'],help='live process only; optional disabled HAL background indexing for sequential reads')
            p.add_argument('--periodic-sampling',choices=['full','fixed-quarter'],help='optional periodic observer only; fixed-quarter retains full temporal events at selected pixels')
            p.add_argument('--periodic-packet',choices=['full','latest-pixel'],help='live process periodic observer only; lossless current141 transport compaction')
            p.add_argument('--center-ensemble-weights',help='live only; optional second ONNX detector for equal current-center averaging')
            p.add_argument('--center-ensemble-sha256',help='required SHA256 for the optional second detector')
            p.add_argument('--local-flow',choices=['none','klt'],help='live only; qualified native ON/OFF feature displacement adjusts rollout velocity')
            p.add_argument('--flow-preparation',choices=['serial','gpu-overlap'],help='live flow only; prepare the same current support while waiting for GPU')
            p.add_argument('--flow-acceleration',choices=['none','velocity-only','position-consistent'],help='causal raw flow velocity derivative; image acceleration only')
            p.add_argument('--body-association',choices=['none','bytetrack'],help='live only; current raw boxes with installed ByteTrack association')
            p.add_argument('--noise-model',choices=['legacy','continuous'],help='live only; default legacy')
            p.add_argument('--publication-clock',choices=['tracking','independent'],help='live only; independent clock includes explicit invalid heartbeats')
            p.add_argument('--periodic-axis',action='store_true',help='Native timestamp periodic support diagnostic; no force claim')
            p.add_argument('--preview', action='store_true')
            p.add_argument('--verify-frames', action='store_true')
            p.add_argument('--dry-run', action='store_true', help='Validate and print argv without reading RAW events or starting inference')
    args = parser.parse_args(argv)
    try:
        overrides={key:getattr(args,key) for key in ('sdk_python','gpu_python','detector_weights','detector_sha256','cache_dir') if getattr(args,key,None) is not None}
        cfg = configuration(args.config,overrides)
        profile=args.profile or cfg.get('profile',default_profile)
        if profile not in profiles:raise ValueError('Profile not available in this launcher: '+profile)
        args.profile=profile
        cfg=apply_track(cfg,args)
        if args.action == 'doctor': return doctor(cfg,getattr(args,'live_recipe_id',None))
        cmd = command(cfg, args)
        if args.dry_run:
            print(json.dumps(dict(argv=cmd, raw_events_read=False), ensure_ascii=False, indent=2))
            return 0
        return subprocess.call(cmd, cwd=ROOT, stdout=sys.stdout, stderr=sys.stderr,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (ValueError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print('Configuration/runtime error: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
