"""Publish declared chronological calibration/test partitions once."""
from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT),str(ROOT/'tools')]
from score_nn_rotor166 import QUERIES
from compare_final_pipeline import reference_at
BASE=ROOT.parent/'output';OUT=BASE/'decision169_20261105'

if __name__=='__main__':
    schedule={};train={};test={}
    for name in ['NN_clear','NN_side','NN_train2','pure_side','pure_train2']:
        kind=name.removeprefix('NN_').removeprefix('pure_')
        cut=28.3 if kind=='clear' else 112. if kind=='side' else 44.
        queries=np.arange(110.4,113.2001,.2) if name=='pure_side' else QUERIES[kind]
        if kind=='clear':
            with zipfile.ZipFile(ROOT.parent/'archive/output_history_20261028.zip') as z:d=json.loads(z.read('paper_reproduction_v030/evaluation_data/train6.json').decode('utf-8'))
            truth=[dict(r,box=r.get('bbox')) for r in d['labels']]
        else:
            folder='dense_labels_side_110_1195' if kind=='side' else 'dense_labels_train2_40_46'
            truth=sorted(json.loads((BASE/folder/'labels.json').read_text('utf-8'))['records'].values(),key=lambda r:r['t'])
        train[name]=[];test[name]=[];schedule[name]=dict(cutoff_t=cut,train_queries=[],test_queries=[],cross_boundary_excluded=[])
        for q in queries:
            q=float(q);target_t=q+.4
            if kind=='clear':
                exact=[r for r in truth if abs(r['t']-target_t)<1e-7 and r.get('state')=='present' and r.get('box')]
                target=((np.asarray(exact[0]['box'][:2])+exact[0]['box'][2:])/2).tolist() if exact else None
            else:
                target=reference_at(truth,target_t);target=None if target is None else np.asarray(target).tolist()
            record=dict(query_t=q,target_t=target_t,target=target)
            if target_t<=cut+1e-8:
                schedule[name]['train_queries'].append(q)
                if target is not None:train[name].append(record)
            elif q>=cut-1e-8:schedule[name]['test_queries'].append(q);test[name].append(record)
            else:schedule[name]['cross_boundary_excluded'].append(q)
    for name,data in [('schedule.json',schedule),('calibration_targets.json',train),('heldout_targets.json',test)]:
        p=OUT/name
        if p.exists():raise FileExistsError(p)
        p.write_text(json.dumps(data,indent=2),'utf-8')
    print(json.dumps({n:dict(train=len(train[n]),test=len(test[n]),cut=schedule[n]['cutoff_t']) for n in schedule}))
