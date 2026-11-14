"""Actual-observation numerical migration check in a fresh isolated process."""
from pathlib import Path
import argparse,json,sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--package-root',type=Path,default=ROOT/'src')
p.add_argument('--references',type=Path,default=ROOT.parent/'output/portable_position_core_20261025/actual_observation_references.json')
p.add_argument('--output',type=Path)
args=p.parse_args();package_root=args.package_root.resolve()
# The exact package must be supplied; do not silently fall through to an old
# editable install, research scripts or the current working directory.
sys.path=[str(package_root)]+[x for x in sys.path if x and not Path(x).resolve().is_relative_to(ROOT)]
def guard(event,items):
    if event=='open' and items and isinstance(items[0],str):
        path=Path(items[0]).resolve()
        if path.is_relative_to(ROOT/'research') or path.is_relative_to(ROOT/'external'):
            raise RuntimeError('Portable core read historical source: '+str(path))
sys.addaudithook(guard)
from event_drone_forecast.position import LiveMotion,predictions
import event_drone_forecast.position as module
assert Path(module.__file__).resolve().is_relative_to(package_root)
data=json.loads(args.references.read_text(encoding='utf-8'));count=0
for case in data:
    model=LiveMotion(25.,case['noise_model'])
    for row in case['rows']:
        c,valid,d=model.step(row['t'],row['body']);preferred,cv=predictions(c,.03)
        assert c==row['coefficient'],(case['case'],case['noise_model'],row['t'],'coefficient')
        assert valid==row['valid'],(case['case'],row['t'],'validity')
        assert d==row['diagnostic'],(case['case'],row['t'],'diagnostic')
        assert preferred==row['preferred'] and cv==row['cv'],(case['case'],row['t'],'forecast')
        count+=1
unexpected=[n for n in sys.modules if n in ('center107','robust23','turn154','metavision_core','metavision_sdk_cv')]
assert not unexpected,unexpected
result=dict(rows=count,noise_models=['legacy','continuous'],all_coefficients_and_forecasts_exact=True,
            all_validity_and_diagnostics_exact=True,research_or_external_source_reads=False,
            SDK_imported=False,module=str(module.__file__),package_root=str(package_root))
if args.output:args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
