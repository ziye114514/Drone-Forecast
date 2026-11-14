from pathlib import Path
import sys,json,hashlib,datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from event_drone_forecast.config import Config

root=Path(__file__).resolve().parents[1]
output=root.parent/'output/raw_only_project'
output.mkdir(exist_ok=True,parents=True)
Config().save(output/'frozen_config.json')
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src').rglob('*.py')}
record=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            development_sequence='train6',heldout_sequences=['train2','train3','train4'],
            input_policy='Prediction reads RAW only. Evaluation human boxes are not passed to inference.',
            output_units='pixels, seconds; no metric 3-D truth available',
            source_files=files,config_sha256=hashlib.sha256((output/'frozen_config.json').read_bytes()).hexdigest(),
            limitations=['Dataset names these recordings train; held out only from this prototype development.',
                'Target visibility and camera motion differ across sequences; single primary target assumption.',
                'Sparse labels permit endpoint error only; no dense future ADE claim.',
                'No RPM/attitude ground truth or wind/metric-scale verification.'])
(output/'experiment_protocol.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(config=str(output/'frozen_config.json'),sources=len(files))))
