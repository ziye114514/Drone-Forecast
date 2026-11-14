"""Record exact IPC lines without changing parsing, inference or frozen55 sources."""
from pathlib import Path
import argparse, datetime, hashlib, json, sys, traceback
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_roi_forecast as runner

def main(output):
    output = Path(output).resolve()
    assert not output.exists()
    record = output.with_name(output.name + '_receive.jsonl')
    assert not record.exists()
    namespace = runner.build('off', 32)
    def receive(worker, kind):
        line = worker.stdout.readline()
        with record.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(kind=kind, line=line, process_poll=worker.poll()), ensure_ascii=False) + '\n')
        if not line:
            raise RuntimeError(kind + ' worker stopped; inspect its log')
        return json.loads(line)
    namespace['receive'] = receive
    metadata = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    original_protocol_sha256=hashlib.sha256((output.parent/'protocol.json').read_bytes()).hexdigest(),
                    design='Exact original parser; log raw IPC lines. One actual RAW20ms step; diagnostic only, not a replacement formal experiment.')
    output.with_name(output.name+'_protocol.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    try:
        namespace['run']('E:/dataset/train/6/media/event.raw', output, 22., .02, 32, False, True)
    except Exception:
        output.with_name(output.name+'_failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        namespace.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
