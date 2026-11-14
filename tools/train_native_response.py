"""Fit one shared unknown-force prior using completed native-rotor observations.

For reproduction or a new independent test, split by entire recording FIRST.
This tool never reads manual evaluation boxes. Existing output is not replaced.
"""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
from conditional_unknown_force import teachers,fit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--residual-frame',choices=['disk','camera'],default='camera');a=p.parse_args()
    if a.output.exists():p.error('Output already exists; preserve the old model and choose a new path')
    groups=[];manifest=[]
    for path in a.predictions:
        raw=path.read_bytes();rr=[json.loads(s) for s in raw.decode('utf8').splitlines()];g=teachers(rr,a.residual_frame)
        groups.append(g);manifest.append(dict(filename=path.name,sha256=hashlib.sha256(raw).hexdigest(),teachers=len(g)))
    if sum(map(len,groups))<10:p.error('Fewer than10 qualified native teachers; do not fit an unsupported response prior')
    m=fit(groups,a.residual_frame);m.update(training_manifest=manifest,training_scope='native weak teachers, not independent force truth',evaluation_policy='Hold entire recordings out before fitting; do not evaluate on these training sources as unseen.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(m,ensure_ascii=False,indent=2),'utf8');print(dict(output=str(a.output),teachers=m['training_samples']))
