"""Serialize NumPy scalar diagnostics without changing frozen scoring arithmetic."""
import argparse,datetime,importlib.util,json,sys
import numpy as np
from common40 import ROOT,OUT,sha,save
original_dumps=json.dumps
def scalar(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)
def main(stage):
    supplement=OUT/'scoring_serialization_supplement.json'
    if not supplement.exists():save(supplement,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=sha(__import__('pathlib').Path(__file__)),reason='Frozen40 scorer completed arithmetic but JSON dump rejected np.int64 sum of Boolean successes. Add only serializer for NumPy scalars/arrays, not changing predictions, thresholds, samples or arithmetic. Same serializer needed41 current-localization summary. No inference rerun.'))
    else:assert json.loads(supplement.read_text(encoding='utf-8'))['helper_sha256']==sha(__import__('pathlib').Path(__file__))
    def dumps(*args,**kwargs):kwargs.setdefault('default',scalar);return original_dumps(*args,**kwargs)
    json.dumps=dumps
    folder=ROOT/f'research/stage{stage}';sys.path.insert(0,str(folder));spec=importlib.util.spec_from_file_location(f'score_bridge_{stage}',folder/f'score{stage}.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,choices=[40,41],required=True);main(p.parse_args().stage)
