"""Freeze source/config, then run RAW-only inference and separate evaluation.

Labels enter only the scoring subprocess after each prediction file is closed.
All sequences were previously examined; this is an engineering regression suite,
not a newly unseen benchmark. Defaults point to this workstation's recordings.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.config import Config

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
RAWS={6:'E:/dataset/train/6/media/event.raw',2:'E:/dataset/train/2/aligned/event.raw',
      3:'E:/dataset/train/3/aligned/event.raw',4:'E:/dataset/train/4/aligned/event.raw'}


def source_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'src').rglob('*.py'))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=WORK/'output/raw_only_project_integrated')
    parser.add_argument('--sequences',type=int,nargs='+',default=[6,2,3,4])
    parser.add_argument('--skip-complete',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    config=out/'frozen_config.json';protocol=out/'experiment_protocol.json'
    hashes=source_hashes()
    if protocol.exists():
        prior=json.loads(protocol.read_text(encoding='utf-8'))
        if prior['source_files']!=hashes:raise RuntimeError('Source changed after freeze; use a new experiment directory.')
    else:
        Config().save(config)
        prior=dict(version='0.2.0',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   source_files=hashes,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                   development_sequence='train6',evaluation_sequences=['train2','train3','train4'],
                   status='Previously examined recordings; frozen-configuration engineering regression, not unseen-test evidence.',
                   input_policy='Inference subprocess reads RAW only. Human annotations enter separate evaluator.',
                   baseline_results=str(WORK/'output/raw_only_project'),
                   benchmark_policy='Compare both each-version coverage and intersection of available label timestamps.')
        protocol.write_text(json.dumps(prior,ensure_ascii=False,indent=2),encoding='utf-8')
    if hashlib.sha256(config.read_bytes()).hexdigest()!=prior['config_sha256']:
        raise RuntimeError('Frozen configuration changed.')
    for sid in args.sequences:
        run=out/'runs'/f'train{sid}'
        if (run/'summary.json').exists() and args.skip_complete:
            print(f'SKIP completed train{sid}',flush=True)
        else:
            if (run/'run_manifest.json').exists():raise RuntimeError(f'Partial/existing run: {run}; inspect before retry.')
            print(f'BEGIN train{sid}',flush=True)
            subprocess.run([sys.executable,str(ROOT/'core/run.py'),'--raw',RAWS[sid],
                            '--output',str(run),'--config',str(config)],check=True)
        labels=WORK/'output/raw_only_project/evaluation_data'/f'train{sid}.json'
        subprocess.run([sys.executable,str(ROOT/'tools/evaluate.py'),'--run',str(run),'--labels',str(labels)],check=True)
    if source_hashes()!=hashes:raise RuntimeError('Sources mutated during experiment.')
    print('SUITE COMPLETE',flush=True)


if __name__=='__main__':main()
