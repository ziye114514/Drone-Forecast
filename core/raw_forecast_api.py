"""Parent-process-safe RAW API; SDK and inference guards remain in the child.

This is a synchronous process boundary for the existing85 research entry, not
an asynchronous/realtime pipeline. Each call includes normal cold startup.
"""
from dataclasses import dataclass
from pathlib import Path
import json,subprocess,time

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SDK_PYTHON=ROOT.parent/'.venv-event311/Scripts/python.exe'

class ForecastProcessError(RuntimeError):
    def __init__(self,returncode,log_path):
        self.returncode=returncode;self.log_path=Path(log_path)
        super().__init__(f'RAW forecast process exited {returncode}; see {self.log_path}')

@dataclass(frozen=True)
class ForecastResult:
    output:Path
    observations:Path
    completion:dict
    protocol:dict
    process_log:Path
    elapsed_wall_s:float

def forecast_raw(raw,output,*,start=0.,duration=0.,preview=False,verify_frames=True,sdk_python=None):
    """Run85 in a fresh process and return usable files after its clean exit.

    Only RAW is an external inference input. Same fixed local weights/math85.
    Existing outputs/logs are never overwritten. On child failure retain the
    partial directory and log and raise ForecastProcessError. No SDK imported
    into this caller. This first version has no cancellation/background mode.
    """
    raw=Path(raw).resolve();output=Path(output).resolve()
    if raw.suffix.lower()!='.raw' or not raw.is_file():raise ValueError('Expected existing .raw input')
    if start<0 or duration<0 or round(start*1e6)%20000 or round(duration*1e6)%20000:raise ValueError('start/duration must be nonnegative20ms multiples')
    if output.exists():raise FileExistsError(output)
    protected=Path('D:/yolov10-main')
    if raw.is_relative_to(protected) or output.is_relative_to(protected):raise ValueError('Protected project path')
    python=Path(sdk_python).resolve() if sdk_python is not None else DEFAULT_SDK_PYTHON
    if not python.is_file():raise FileNotFoundError(python)
    output.parent.mkdir(parents=True,exist_ok=True)
    log_path=output.parent/(output.name+'_process.log')
    args=[str(python),str(ROOT/'core/run_batched_forecast.py'),'--raw',str(raw),'--output',str(output),
          '--start',str(float(start)),'--duration',str(float(duration))]
    if preview:args.append('--preview')
    if verify_frames:args.append('--verify-frames')
    started=time.perf_counter()
    with log_path.open('x',encoding='utf-8') as log:
        result=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    elapsed=time.perf_counter()-started
    if result.returncode!=0:raise ForecastProcessError(result.returncode,log_path)
    protocol=json.loads((output/'protocol.json').read_text('utf-8'))
    completion=json.loads((output/'completion.json').read_text('utf-8'))
    if protocol.get('exact_compute_deployment_stage')!=85 or not completion.get('source_hashes_unchanged') or completion.get('labels_read'):
        raise RuntimeError('Child completion/provenance check failed; preserve output for inspection')
    if duration and completion['frames']!=round(duration*50):raise RuntimeError('Incomplete RAW child output')
    observations=output/'observations_forecasts.jsonl'
    if not observations.is_file():raise RuntimeError('Missing child forecasts')
    return ForecastResult(output,observations,completion,protocol,log_path,elapsed)
