"""Record a verifier-only repair; frozen forecasting sources stay unchanged."""
from pathlib import Path
import datetime,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'research/stage120')]
import study120 as s
p=json.loads((s.OUT/'protocol120.json').read_text('utf-8'));new=s.sources();old=p['sources'];key=Path(s.__file__).relative_to(ROOT).as_posix()
assert {k:v for k,v in old.items() if k!=key}=={k:v for k,v in new.items() if k!=key}
snapshot=ROOT/'research/stage120/study120_validation_v1.py';assert s.sha(snapshot)==old[key]
s.save(s.OUT/'verification_amendment120.json',dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    failure='First actual RAW clear execution exited0 and emitted200 frames. Checker incorrectly required every startup tick with any coefficients to have all64 models initialized; some models require warmup.',
    repair='Still compare every present old coefficient/prediction at every tick exactly. Require union64 after whole positive sequence; negative may initialize none. No inference/observation/metric change or rerun of full clear.',
    original_source_snapshot=str(snapshot),original_hash=old[key],new_hash=new[key],all_forecasting_sources_unchanged=True))
p['sources']=new;p['verification_amendment']='verification_amendment120.json';s.save(s.OUT/'protocol120.json',p)
s.check('clear6');s.run('prefix_clear6');s.run('negative5')
