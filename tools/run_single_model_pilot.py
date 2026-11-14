"""Single merged detector plus independently qualified scale rollout."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path.insert(0,str(ROOT))
from forecast import configuration
OUT=WORK/'output/single_model_optimized_RAW_20261026';OUT.mkdir(exist_ok=False)
cfg=configuration(ROOT/'configs/local.json')
export=json.loads((WORK/'output/equal_weight_detector_20261026/export.json').read_text('utf-8'))
cfg.update(profile='live',frame_transport='frames',startup_sync='reader-ready',reader_index='disabled',
    detector_weights=export['model'],detector_sha256=export['sha256'],scale_rollout='compatible')
cfg.pop('center_ensemble_weights',None);cfg.pop('center_ensemble_sha256',None)
config=OUT/'runtime.private.json';config.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
for case,start,raw in [('side',110.,'E:/dataset/train/6/media/event.raw'),('train2',40.,'E:/dataset/train/2/aligned/event.raw')]:
    args=[sys.executable,'-X','utf8',str(ROOT/'forecast.py'),'run','--config',str(config),
        '--raw',raw,'--output',str(OUT/case),'--start',str(start),'--duration','4',
        '--profile','live','--preview','--verify-frames']
    result=subprocess.run(args,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise SystemExit(result.returncode)
    print('Completed single-model '+case,flush=True)
