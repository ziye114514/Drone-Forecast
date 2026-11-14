"""One-time mechanical extraction of existing math, not an algorithm rewrite.

Captures actual-observation reference outputs BEFORE changing the imports.
Copies only the existing functions/classes and frozen parameters; FilterPy
author arithmetic remains in unmodified, licensed source snapshots.
"""
from pathlib import Path
import ast,copy,hashlib,json,pprint,shutil,sys
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path.insert(0,str(ROOT))
OUT=WORK/'output/portable_position_core_20261025';OUT.mkdir(exist_ok=False)
BACKUP=OUT/'before';BACKUP.mkdir()
for name in ['live_motion.py','live_measurement_kf.py','live_covariance.py','live_gain.py','run_live_forecast.py','pyproject.toml']:
    shutil.copy2(ROOT/name,BACKUP/name)
from live_motion import LiveMotion,predictions
import center107,robust23

references=[]
for case in ['clear','side','train2','negative']:
    source=WORK/f'output/live_body_association_v2_20261025/{case}/tracking.jsonl'
    inputs=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
    for noise in ['legacy','continuous']:
        motion=LiveMotion(25.,noise);seq=[]
        for row in inputs:
            c,valid,d=motion.step(row['t'],row['body']);pref,cv=predictions(c,.03)
            seq.append(dict(t=row['t'],body=row['body'],coefficient=c,valid=valid,diagnostic=d,preferred=pref,cv=cv))
        references.append(dict(case=case,noise_model=noise,rows=seq))
(OUT/'actual_observation_references.json').write_text(json.dumps(references,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

PACKAGE=ROOT/'src/event_drone_forecast';DEST=PACKAGE/'position';DEST.mkdir(exist_ok=False)
VENDOR=PACKAGE/'_vendor';VENDOR.mkdir(exist_ok=False)
(VENDOR/'__init__.py').write_text('"""Pinned third-party sources; original licenses retained."""\n',encoding='utf-8')
for name in ['filterpy_linear_upstream','filterpy_process_upstream']:
    shutil.copytree(ROOT/'external'/name,VENDOR/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))

def select(path,names):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==len(names),(path,names)
    return nodes

class LocalCovarianceImport(ast.NodeTransformer):
    def visit_ImportFrom(self,node):
        if node.module=='live_covariance':node.module='covariance';node.level=1
        return node

def emit(name,header,nodes):
    (DEST/name).write_text(header+'\n\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n',encoding='utf-8')

kf=select(ROOT/'research/stage107/center107.py',['coefficient','SimpleKF'])
kf+=select(ROOT/'live_measurement_kf.py',['MeasurementKF'])
kf=[LocalCovarianceImport().visit(copy.deepcopy(n)) for n in kf]
emit('kalman.py','"""Unchanged107 KF state and existing measurement update; imports only moved."""\nimport numpy as np\nfrom ._filterpy import predict,update\nPARAMS='+pprint.pformat(center107.PARAMS,sort_dicts=False),kf)
emit('robust.py','"""Existing23 robust acceleration math; frozen parameters, no history imports."""\nimport numpy as np\nfrom scipy.optimize import least_squares\nPARAMS='+pprint.pformat(robust23.PARAMS,sort_dicts=False),select(ROOT/'research/stage23/robust23.py',['cap','robust_acceleration']))
shutil.copy2(ROOT/'research/stage154/turn154.py',DEST/'rollout.py')
emit('motion.py','"""Current live motion state; mechanically extracted, same numerical rules."""\nfrom collections import deque\nimport copy,math\nimport numpy as np\nfrom .kalman import MeasurementKF\nfrom .robust import robust_acceleration,cap\nfrom .rollout import forecast as curved_forecast',select(ROOT/'live_motion.py',['LiveMotion','predictions']))

cov=(ROOT/'live_covariance.py').read_text(encoding='utf-8')
begin=cov.index('ROOT=');end=cov.index('snapshot=json.loads')
cov=cov[:begin]+"from .kalman import SimpleKF,predict,update,coefficient,PARAMS\nVENDOR=Path(__file__).resolve().parents[1]/'_vendor/filterpy_process_upstream'\n"+cov[end:]
(DEST/'covariance.py').write_text(cov,encoding='utf-8')
filter_source=(ROOT/'research/stage107/center107.py').read_text(encoding='utf-8')
begin=filter_source.index('snapshot=json.loads');end=filter_source.index('\nPARAMS=')
preface='"""Execute pinned unmodified MIT FilterPy author functions, independent of research paths."""\nfrom pathlib import Path\nimport ast,hashlib,json\nimport numpy as np\nVENDOR=Path(__file__).resolve().parents[1]/\'_vendor/filterpy_linear_upstream\'\n'
(DEST/'_filterpy.py').write_text(preface+filter_source[begin:end]+'\n',encoding='utf-8')
(DEST/'__init__.py').write_text('"""SDK-independent current image-position forecasting core."""\nfrom .motion import LiveMotion,predictions\n__all__=["LiveMotion","predictions"]\n',encoding='utf-8')
manifest={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'research/stage107/center107.py',ROOT/'research/stage23/robust23.py',ROOT/'research/stage154/turn154.py',BACKUP/'live_motion.py',BACKUP/'live_measurement_kf.py']}
(OUT/'extraction_manifest.json').write_text(json.dumps(dict(method='Copy own function/class ASTs without arithmetic changes; freeze actual current parameter dictionaries; relocate pinned author sources',sources=manifest,reference_rows=sum(len(r['rows']) for r in references)),indent=2),encoding='utf-8')
print('Generated portable package and original reference rows:',sum(len(r['rows']) for r in references),flush=True)
