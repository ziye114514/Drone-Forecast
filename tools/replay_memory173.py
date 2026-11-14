"""Reuse existing bounded force-memory credit on native rotor observations."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from coherent_gap_observer import CoherentGapObserver
from adaptive_force_memory import AdaptiveForceMemory
from physical_publication import evaluate
from score_rotor164 import score
OUT=ROOT.parent/'output/gap173_20261107/memory_trial'

if __name__=='__main__':
    OUT.mkdir(exist_ok=False);summary=dict(parameters_searched=False,NN_used=False,labels_runtime_inputs=False,cases={})
    for name in ('side','train2'):
        source=ROOT.parent/f'output/gap173_20261107/verified/{name}_coherent_gap.jsonl'
        rs=[json.loads(r) for r in source.read_text('utf-8').splitlines()];memory=AdaptiveForceMemory(allow_shorter=True);output=[]
        for old in rs:
            r=copy.deepcopy(old);body=r['body'];shift=body.get('measurement_reference_shift_xy')
            if shift is not None:memory.rebase(shift)
            current=bool(body.get('detected') and not body.get('center_provisional'))
            supported=bool(r['valid']) # Existing recent-measurement expiry remains active.
            predictions,meta=memory.step(r['t'],r['coefficient'],np.array(body['center']) if current else None,
                r['physical'],r['force'],valid=r['valid'],native_net_motion_supported=supported)
            r['adaptive_force_memory']=meta;r['physical']=predictions
            if r['coefficient']:
                expected=evaluate(r,[.1,.2,.4])
                for key in predictions:np.testing.assert_allclose(predictions[key],expected[key],atol=1e-8)
            output.append(r)
        dest=OUT/f'{name}.jsonl';dest.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in output),'utf-8')
        summary['cases'][name]=dict(rows=len(output),active=sum(bool(r['adaptive_force_memory']['applied']) for r in output),
            updated=sum(bool(r['adaptive_force_memory']['updated']) for r in output))
    # Freeze all candidate outputs before opening development references.
    for name,r in summary['cases'].items():
        labels=ROOT.parent/('output/dense_labels_side_110_1195/labels.json' if name=='side' else 'output/dense_labels_train2_40_46/labels.json')
        r['score']=score(OUT/f'{name}.jsonl',labels,np.arange(110.4,114.0001,.2) if name=='side' else np.arange(43.2,44.4001,.2))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),'utf-8')
    print(json.dumps({n:dict(mean=r['score']['error']['mean'],n=r['score']['available'],active=r['active']) for n,r in summary['cases'].items()}))
