"""A real RAW prefix must reproduce the already-published full-run prefix."""
from pathlib import Path
import argparse,json,subprocess,sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();root=Path(__file__).resolve().parents[1]
    full=out/'runs/train3';prefix=out/'causality_prefix_check'
    manifest=json.loads((full/'run_manifest.json').read_text())
    if not (prefix/'summary.json').exists():
        subprocess.run([sys.executable,str(root/'run.py'),'--raw',manifest['input_raw'],
                        '--output',str(prefix),'--config',str(out/'frozen_config.json'),'--duration','4'],check=True)
    result=dict(raw=manifest['input_raw'],prefix_seconds=4,future_suffix_seconds_omitted=56)
    for filename,name in [('observations.jsonl','observations'),('forecasts.jsonl','forecasts')]:
        short=[json.loads(s) for s in (prefix/filename).read_text().splitlines()]
        long=[json.loads(s) for s in (full/filename).read_text().splitlines()]
        result[name+'_exactly_equal']=short==long[:len(short)]
        result['prefix_frames']=len(short)
    result['interpretation']='Real RAW prefix regression, not an exhaustive proof for every sequence.'
    (out/'causality_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    if not result['observations_exactly_equal'] or not result['forecasts_exactly_equal']:
        raise SystemExit('Causality regression failed')


if __name__=='__main__':main()
