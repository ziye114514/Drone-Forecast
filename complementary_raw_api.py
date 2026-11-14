"""Synchronous, parent-safe API for experimental RAW93; stdlib-only import."""
from pathlib import Path
import json,subprocess,time
from raw_forecast_api import ForecastResult,ForecastProcessError,DEFAULT_SDK_PYTHON

ROOT=Path(__file__).resolve().parent


def forecast_raw(raw,output,*,start=0.,duration=0.,preview=False,verify_frames=True,sdk_python=None):
    """Run the fixed55-candidate93 CLI in an isolated child.

    Existing outputs/logs are protected. This API does not turn the sequential
    detector into realtime/asynchronous processing. Old raw_forecast_api keeps85.
    """
    raw=Path(raw).resolve();output=Path(output).resolve()
    for protected in [Path('D:/yolov10-main'),Path('E:/dataset/software/yolov10-main')]:
        if raw.is_relative_to(protected) or output.is_relative_to(protected):raise ValueError('Protected project path')
    if raw.suffix.lower()!='.raw' or not raw.is_file():raise ValueError('Expected existing .raw input')
    if start<0 or duration<0 or round(start*1e6)%20000 or round(duration*1e6)%20000:raise ValueError('start/duration must be nonnegative20ms multiples')
    if output.exists():raise FileExistsError(output)
    python=Path(sdk_python).resolve() if sdk_python is not None else DEFAULT_SDK_PYTHON
    if not python.is_file():raise FileNotFoundError(python)
    output.parent.mkdir(parents=True,exist_ok=True);log_path=output.parent/(output.name+'_process.log')
    args=[str(python),str(ROOT/'run_complementary_forecast.py'),'--raw',str(raw),'--output',str(output),
          '--start',str(float(start)),'--duration',str(float(duration))]
    if preview:args.append('--preview')
    if verify_frames:args.append('--verify-frames')
    started=time.perf_counter()
    with log_path.open('x',encoding='utf-8') as log:
        result=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    elapsed=time.perf_counter()-started
    if result.returncode:raise ForecastProcessError(result.returncode,log_path)
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    completion=json.loads((output/'completion.json').read_text(encoding='utf-8'))
    if protocol.get('complementary_deployment_stage')!=93 or not protocol.get('original49_retained') or not completion.get('source_hashes_unchanged') or completion.get('labels_read'):
        raise RuntimeError('Child completion/provenance check failed; output retained')
    if duration and completion['frames']!=round(duration*50):raise RuntimeError('Incomplete RAW child output')
    observations=output/'observations_forecasts.jsonl'
    if not observations.is_file():raise RuntimeError('Missing child forecasts')
    return ForecastResult(output,observations,completion,protocol,log_path,elapsed)
