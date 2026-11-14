"""Check public recipe selection, baseline access and override precedence."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path)
args=parser.parse_args()
OUT=args.output_dir.resolve() if args.output_dir else WORK/'output/live_recipe_checks_20261026';OUT.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'configs/live_tracks.json').read_text('utf-8'))
recommended=manifest['recommended'];recipe=manifest['tracks'][recommended]
stub=OUT/'dry_run_only.raw';stub.write_bytes(b'');results={}
try:
    cases=[('recommended','forecast_mainline.py',[]),('baseline','forecast_mainline.py',['--profile','live-baseline']),
           ('old_live','forecast.py',['--profile','live']),
           ('no_flow','forecast_mainline.py',['--local-flow','none']),
           ('override','forecast_mainline.py',['--detector-weights',str(ROOT/'models/live_position_baseline.onnx'),
                '--detector-sha256','e2531556ad0431d904fb0d1d432727a2e2dd9eaed9e098a6ed3d9e55f474db0b'])]
    for name,entry,extra in cases:
        args=[sys.executable,'-X','utf8',str(ROOT/entry),'run','--config',str(ROOT/'configs/local.json'),
            '--raw',str(stub),'--output',str(OUT/(name+'_not_created')),'--dry-run']+extra
        r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,encoding='utf-8');assert r.returncode==0,r.stderr
        d=json.loads(r.stdout);assert not d['raw_events_read'];argv=d['argv']
        def value(flag):return argv[argv.index(flag)+1] if flag in argv else None
        results[name]=dict(model_sha256=value('--detector-sha256'),scale=value('--scale-rollout'),
            transport=value('--frame-transport'),startup=value('--startup-sync'),recipe=value('--recipe-id'),local_flow=value('--local-flow'),flow_preparation=value('--flow-preparation'),flow_acceleration=value('--flow-acceleration'))
        assert not (OUT/(name+'_not_created')).exists()
    assert results['recommended']['model_sha256']==recipe['sha256']
    assert results['recommended']['scale']==recipe['parameters']['scale_rollout'] and results['recommended']['recipe']==recommended
    assert results['recommended']['local_flow']==recipe['parameters'].get('local_flow','none')
    assert results['recommended']['flow_preparation']==recipe['parameters'].get('flow_preparation','serial')
    assert results['recommended']['flow_acceleration']==recipe['parameters'].get('flow_acceleration','none')
    assert results['baseline']['model_sha256']==results['old_live']['model_sha256']=='e2531556ad0431d904fb0d1d432727a2e2dd9eaed9e098a6ed3d9e55f474db0b'
    assert results['baseline']['scale']==results['old_live']['scale']=='none'
    assert results['baseline']['local_flow']==results['old_live']['local_flow']=='none'
    assert results['baseline']['flow_preparation']==results['old_live']['flow_preparation']=='serial'
    assert results['no_flow']['local_flow']=='none' and results['no_flow']['flow_preparation']=='serial'
    assert results['no_flow']['flow_acceleration']==results['baseline']['flow_acceleration']==results['old_live']['flow_acceleration']=='none'
    assert results['override']['model_sha256']==results['baseline']['model_sha256']
    assert results['override']['recipe']==recommended+':model_override'
    report=dict(results=results,RAW_or_GPU_started=False,baseline_separate_and_exact_model=True)
    (OUT/'entry_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
finally:stub.unlink()
