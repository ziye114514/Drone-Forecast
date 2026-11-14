"""Keep parallel resource failure evidence without altering frozen inference."""
from pathlib import Path
import datetime,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage123')]
from study123 import OUT,save,sha
def main():
    failed=OUT/'failed_parallel_startup';records=[]
    for case in ['extension1','extension2','natural6']:
        p=failed/case;gpu=p/'gpu_worker.log';text=gpu.read_text('utf-8',errors='replace');assert '1455' in text and 'shm.dll' in text
        data=p/'observations_forecasts.jsonl';records.append(dict(case=case,forecast_bytes=data.stat().st_size if data.exists() else 0,
            gpu_log=str(gpu),gpu_log_sha256=sha(gpu),error='WinError1455: pagefile/commit allocation failure during Torch DLL import; no inference',
            parent_log=str(failed/(case+'_parent.log'))))
    assert all(r['forecast_bytes']==0 for r in records)
    save(OUT/'startup_failure_audit123.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempts=records,
        action='Retain all failed files; sequential retry of identical frozen cases. No OS memory/settings changes, no forecast source changes.',
        classification='Execution resource failure, not detection/forecast failure; all successfully decoded natural frames remain later evaluation denominators.'))
    print(json.dumps(records),flush=True)
if __name__=='__main__':main()
