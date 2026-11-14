"""Keep original strict JSON IPC; preserve any offending line on a fresh run."""
from pathlib import Path
import argparse, datetime, hashlib, json
import study55 as current

HERE = Path(__file__).resolve()
SUPPLEMENT = current.OUT/'ipc_observation_supplement.json'

def freeze():
    current.verify()
    assert not SUPPLEMENT.exists()
    failed = current.OUT/'off_clear6'
    target = current.OUT/'failed_off_clear6_initial_json'
    assert failed.is_dir() and not target.exists()
    assert failed.resolve().parent == target.resolve().parent == current.OUT.resolve()
    assert not (failed/'completion.json').exists()
    manifest = {p.relative_to(failed).as_posix():current.sha(p) for p in failed.rglob('*') if p.is_file()}
    failed.rename(target)
    current.save(SUPPLEMENT, dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        helper_sha256=current.sha(HERE), diagnostic_sha256=current.sha(HERE.with_name('ipc_diagnostic55.py')),
        original_protocol_sha256=current.sha(current.OUT/'protocol.json'),
        failed_run=str(target), failed_files_sha256=manifest,
        original_batch_session=12452, original_batch_exit_code=1,
        original_exception='json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)',
        diagnosis='Original offending stdout line was not retained; both stderr logs empty. A fresh one-step diagnostic with exact original parser succeeded. Cause remains unresolved; do not label inference or ROI accuracy as failed.',
        change='Only add exception logging around unchanged json.loads. No skipped lines, parsing leniency, predictions, thresholds or mathematical changes. Preserve any next failure and stop. Full ROI-off must still exactly regress54.',
        original_failed_output_preserved=True))
    print('Original failed run retained; strict-parser observation supplement frozen', flush=True)

def verify():
    current.verify()
    supplement = current.load(SUPPLEMENT)
    assert supplement['helper_sha256'] == current.sha(HERE)
    assert supplement['diagnostic_sha256'] == current.sha(HERE.with_name('ipc_diagnostic55.py'))
    assert supplement['original_protocol_sha256'] == current.sha(current.OUT/'protocol.json')
    for name, value in supplement['failed_files_sha256'].items():
        assert current.sha(Path(supplement['failed_run'])/name) == value

def run(case):
    verify()
    def observed_run(raw, output, start=0., duration=0., precision=32, preview=False,
                     verify_frames=False, geometry_roi='body', event_capacity_mib=32):
        namespace = current.runner.build(geometry_roi, event_capacity_mib)
        def receive(worker, kind):
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError(kind+' worker stopped; inspect its log')
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                with (current.OUT/'ipc_observation_failures.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(case=case, kind=kind, raw_line=line,
                                                 process_poll=worker.poll()), ensure_ascii=False)+'\n')
                raise
        namespace['receive'] = receive
        try:
            return namespace['run'](raw, output, start, duration, precision, preview, verify_frames)
        finally:
            namespace.clear()
    current.runner.run = observed_run
    current.run(case)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run', 'verify'])
    parser.add_argument('--case', choices=list(current.CASES))
    args = parser.parse_args()
    if args.action == 'run':
        run(args.case)
    else:
        globals()[args.action]()
